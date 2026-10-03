"""Training / validation loop for restoration autoencoders (Task 1, and the Task 2 specialists).

One function, `train_restoration`, is used both by Optuna trials (short, with pruning) and by the
final training run (long, with checkpointing), so the tuned config is trained the same way.
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
                      patience: int | None = None, num_workers: int = 2, log_every: int = 1,
                      sample_fn=None) -> dict:
    """Train with loss = alpha*L1 + (1-alpha)*(1-SSIM), AdamW, cosine LR decay, AMP.

    cfg needs: lr, batch_size, alpha, weight_decay (optional).
    trial:     Optuna trial -> report val_score every epoch and prune if the pruner says so.
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
            mlflow.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=epoch)
        if sample_fn is not None:
            sample_fn(model, epoch)

        if row["val_score"] < best["val_score"]:
            best = {**row, "state": copy.deepcopy(model.state_dict())}
            since_best = 0
        else:
            since_best += 1

        if trial is not None:
            trial.report(row["val_score"], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned(f"pruned at epoch {epoch}")
        if patience is not None and since_best >= patience:
            print(f"early stopping at epoch {epoch} (best epoch {best['epoch']})")
            break
        print(f"epoch {epoch:3d}  loss {row['train_loss']:.4f}  val_score {row['val_score']:.4f}  "
              f"ssim {row['val_ssim']:.4f}  psnr {row['val_psnr']:.2f}  ({row['epoch_s']:.1f}s)", flush=True)

    return {"history": history, "best": best}
