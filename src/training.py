"""Training / validation loops.

- `train_restoration`: autoencoders (Task 1, and the Task 2 specialists).
- `train_classifier`:  the 4-class corruption classifier (Task 2), with exactly balanced batches.
Each is used both by Optuna trials (short, with pruning) and by the final training run (long, with
checkpointing), so the tuned config is trained the same way.
"""
import copy
import math
import time

import mlflow
import numpy as np
import optuna
import torch
from torch.utils.data import DataLoader

from src import config as C
from src.data import pet
from src.losses import restoration_loss
from src.metrics import objective_score, per_image_metrics
from src.tracking import TrainingDiverged


def make_train_loader(batch_size: int, conditions=(0, 1, 2, 3), num_workers: int = 2) -> DataLoader:
    """Runtime-corrupted training images. Each load picks a condition uniformly from `conditions`."""
    images = pet.CachedImages("trainval", pet.load_split()["train"])
    ds = pet.PetTrainDataset(images, conditions=conditions)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers,
                      persistent_workers=num_workers > 0, pin_memory=True)


def load_val_tensors(conditions=(0, 1, 2, 3)):
    """The fixed validation set (manifest) as CPU tensors, built once. Filtered to `conditions`."""
    ds = pet.load_manifest_dataset("val")
    keep = [i for i, e in enumerate(ds.entries) if e["condition"] in conditions]
    xs, ys, labels = zip(*[ds[i][:3] for i in keep])
    return torch.stack(xs), torch.stack(ys), torch.tensor(labels)


@torch.no_grad()
def predict(model, x: torch.Tensor, device, batch_size: int = 128) -> torch.Tensor:
    """Run the model over a CPU tensor in batches; returns CPU float32 outputs."""
    model.eval()
    outs = []
    for i in range(0, len(x), batch_size):
        with torch.autocast("cuda", dtype=torch.float16):
            outs.append(model(x[i:i + batch_size].to(device)).float().cpu())
    return torch.cat(outs)


def validate(model, val, device) -> dict:
    x, y, labels = val
    m = per_image_metrics(predict(model, x, device), y)
    out = {"val_l1": m["l1"].mean().item(), "val_ssim": m["ssim"].mean().item(), "val_psnr": m["psnr"].mean().item()}
    out["val_score"] = objective_score(out["val_l1"], out["val_ssim"])
    for c in labels.unique().tolist():  # per-condition SSIM, useful to watch in MLflow
        out[f"val_ssim_{C.CONDITIONS[c]}"] = m["ssim"][labels == c].mean().item()
    return out


def train_restoration(model, cfg: dict, epochs: int, device, *, conditions=(0, 1, 2, 3), trial=None,
                      step_offset: int = 0, patience: int | None = None, num_workers: int = 2,
                      log_every: int = 1, sample_fn=None, metric_prefix: str = "") -> dict:
    """Train with loss = alpha*L1 + (1-alpha)*(1-SSIM), AdamW, cosine LR decay, AMP.

    cfg needs: lr, batch_size, alpha, weight_decay (optional).
    trial:     Optuna trial -> report val_score every epoch and prune if the pruner says so.
               step_offset shifts the reported step (several models trained in one trial).
    metric_prefix: prefix for MLflow metric names (e.g. "salt/") when several models share one run.
    patience:  early stopping on val_score (validation set only, never test).
    sample_fn: optional callback(model, epoch) to log sample images.
    Logs per-epoch metrics to the active MLflow run. Returns history + best state (by val_score).
    """
    loader = make_train_loader(cfg["batch_size"], conditions, num_workers)
    val = load_val_tensors(conditions)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-5))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler()
    best = {"val_score": math.inf}
    history, since_best = [], 0

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        losses = []
        for x, y, _ in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                pred = model(x)
            loss = restoration_loss(pred, y, cfg["alpha"])  # computed in float32
            if not torch.isfinite(loss):
                raise TrainingDiverged(f"non-finite loss at epoch {epoch}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        sched.step()

        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "lr": sched.get_last_lr()[0],
               **validate(model, val, device), "epoch_s": time.perf_counter() - t0}
        history.append(row)
        if epoch % log_every == 0 or epoch == epochs:
            mlflow.log_metrics({metric_prefix + k: v for k, v in row.items() if k != "epoch"}, step=epoch)
        if sample_fn is not None:
            sample_fn(model, epoch)

        if row["val_score"] < best["val_score"]:
            best = {**row, "state": copy.deepcopy(model.state_dict())}
            since_best = 0
        else:
            since_best += 1

        if trial is not None:
            trial.report(row["val_score"], epoch + step_offset)
            if trial.should_prune():
                raise optuna.TrialPruned(f"pruned at epoch {epoch}")
        if patience is not None and since_best >= patience:
            print(f"early stopping at epoch {epoch} (best epoch {best['epoch']})")
            break
        print(f"epoch {epoch:3d}  loss {row['train_loss']:.4f}  val_score {row['val_score']:.4f}  "
              f"ssim {row['val_ssim']:.4f}  psnr {row['val_psnr']:.2f}  ({row['epoch_s']:.1f}s)", flush=True)

    return {"history": history, "best": best}


# ------------------------------------------------------------------ classifier (Task 2)

def make_balanced_loader(batch_size: int, seed: int = C.SEED, num_workers: int = 2) -> DataLoader:
    """Training images where every batch has exactly batch_size/4 of each condition."""
    from src.data.samplers import BalancedBatchSampler
    images = pet.CachedImages("trainval", pet.load_split()["train"])
    sampler = BalancedBatchSampler(len(images), batch_size, seed=seed)
    return DataLoader(pet.PetTrainDataset(images), batch_sampler=sampler, num_workers=num_workers,
                      persistent_workers=num_workers > 0, pin_memory=True)


@torch.no_grad()
def classifier_logits(model, x: torch.Tensor, device, batch_size: int = 256) -> torch.Tensor:
    model.eval()
    outs = []
    for i in range(0, len(x), batch_size):
        with torch.autocast("cuda", dtype=torch.float16):
            outs.append(model(x[i:i + batch_size].to(device)).float().cpu())
    return torch.cat(outs)


def validate_classifier(model, val, device) -> dict:
    from sklearn.metrics import f1_score
    x, _, labels = val
    logits = classifier_logits(model, x, device)
    pred = logits.argmax(1)
    return {"val_loss": torch.nn.functional.cross_entropy(logits, labels).item(),
            "val_acc": (pred == labels).float().mean().item(),
            "val_macro_f1": f1_score(labels, pred, average="macro")}


def train_classifier(model, cfg: dict, epochs: int, device, *, trial=None, patience: int | None = None,
                     num_workers: int = 2) -> dict:
    """Cross-entropy, AdamW (lr, weight_decay from cfg), cosine LR decay, AMP, balanced batches.

    Model selection metric = validation cross-entropy (lower is better). We use the loss rather than
    accuracy because this network becomes the Task 3 gate, where its probabilities (not only its
    top choice) are used as mixing weights, so they should be well calibrated.
    """
    loader = make_balanced_loader(cfg["batch_size"], num_workers=num_workers)
    val = load_val_tensors()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler()
    best, history, since_best = {"val_loss": math.inf}, [], 0

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        loader.batch_sampler.set_epoch(epoch)  # new (still reproducible) shuffle every epoch
        model.train()
        losses, correct, seen = [], 0, 0
        for x, _, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits.float(), y)
            if not torch.isfinite(loss):
                raise TrainingDiverged(f"non-finite loss at epoch {epoch}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)
        sched.step()

        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "train_acc": correct / seen,
               "lr": sched.get_last_lr()[0], **validate_classifier(model, val, device),
               "epoch_s": time.perf_counter() - t0}
        history.append(row)
        mlflow.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=epoch)
        if row["val_loss"] < best["val_loss"]:
            best, since_best = {**row, "state": copy.deepcopy(model.state_dict())}, 0
        else:
            since_best += 1
        print(f"epoch {epoch:3d}  loss {row['train_loss']:.4f}  val_loss {row['val_loss']:.4f}  "
              f"val_acc {row['val_acc']:.4f}  f1 {row['val_macro_f1']:.4f}  ({row['epoch_s']:.1f}s)", flush=True)
        if trial is not None:
            trial.report(row["val_loss"], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned(f"pruned at epoch {epoch}")
        if patience is not None and since_best >= patience:
            print(f"early stopping at epoch {epoch} (best epoch {best['epoch']})")
            break
    return {"history": history, "best": best}
