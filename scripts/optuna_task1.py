"""Task 1 Optuna study: universal denoising autoencoder.

Search space (PDF: at least lr, batch size, bottleneck dim, encoder channels, dropout, alpha):
    lr             log-uniform [1e-4, 3e-3]
    batch_size     {16, 32, 64}            (<= 64 fits easily in 4 GB, measured peak 1.7 GB)
    bottleneck_dim {256, 512, 1024, 2048}  (latent 8x8 x 4/8/16/32 channels = 192x .. 24x compression)
    base_channels  {24, 32, 48, 64}        (encoder widths b, 2b, 4b, 8b)
    dropout        uniform [0.0, 0.3]      (on the latent code)
    alpha          uniform [0.5, 0.95]     (loss = alpha*L1 + (1-alpha)*(1-SSIM))
Objective (minimize): best validation score over the trial's epochs, where
    score = 0.5 * val_L1 + 0.5 * (1 - val_SSIM)   -- fixed, does NOT depend on alpha.
Short trials (default 15 epochs), MedianPruner, OOM/NaN -> pruned. Resumable (SQLite storage).

Run:  .venv\\Scripts\\python scripts/optuna_task1.py --n-trials 30
Outputs: artifacts/optuna/task1.db, artifacts/results/task1_optuna_{summary.json,trials.csv},
         docs/figures/task1_optuna_*.png, configs/task1.yaml (best config for the final training).
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
import mlflow
import optuna
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C  # noqa: E402
from src.models.autoencoder import ConvAutoencoder, count_parameters  # noqa: E402
from src.tracking import create_study, safe_objective, setup_mlflow, study_summary, trial_run  # noqa: E402
from src.training import train_restoration  # noqa: E402
from src.utils import get_device, seed_everything  # noqa: E402

STUDY = "task1"
SEARCH_SPACE = {
    "lr": "log-uniform [1e-4, 3e-3]",
    "batch_size": [16, 32, 64],
    "bottleneck_dim": [256, 512, 1024, 2048],
    "base_channels": [24, 32, 48, 64],
    "dropout": "uniform [0.0, 0.3]",
    "alpha": "uniform [0.5, 0.95]",
}


def suggest(trial: optuna.Trial) -> dict:
    return {
        "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", SEARCH_SPACE["batch_size"]),
        "bottleneck_dim": trial.suggest_categorical("bottleneck_dim", SEARCH_SPACE["bottleneck_dim"]),
        "base_channels": trial.suggest_categorical("base_channels", SEARCH_SPACE["base_channels"]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        "alpha": trial.suggest_float("alpha", 0.5, 0.95),
    }


def make_objective(epochs: int, device):
    def objective(trial: optuna.Trial) -> float:
        cfg = suggest(trial)
        seed_everything(C.SEED)  # same init/data order for every trial -> fair comparison
        model = ConvAutoencoder(cfg["base_channels"], cfg["bottleneck_dim"], cfg["dropout"]).to(device)
        with trial_run(trial, {**cfg, "epochs": epochs, "params_M": count_parameters(model) / 1e6}):
            result = train_restoration(model, cfg, epochs, device, trial=trial)
            best = result["best"]
            mlflow.log_metrics({"best_val_score": best["val_score"], "best_val_ssim": best["val_ssim"],
                                "best_val_psnr": best["val_psnr"], "best_epoch": best["epoch"]})
            for k in ("val_ssim", "val_psnr", "val_l1", "epoch"):
                trial.set_user_attr(f"best_{k}", best[k])
            return best["val_score"]
    return objective


def save_outputs(study: optuna.Study, epochs: int) -> dict:
    summary = {"study": STUDY, "direction": "minimize",
               "objective": "0.5*val_L1 + 0.5*(1 - val_SSIM), best epoch of each trial",
               "epochs_per_trial": epochs, "search_space": SEARCH_SPACE, **study_summary(study)}
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    (C.RESULTS / "task1_optuna_summary.json").write_text(json.dumps(summary, indent=2))
    study.trials_dataframe().to_csv(C.RESULTS / "task1_optuna_trials.csv", index=False)

    C.FIGURES.mkdir(parents=True, exist_ok=True)
    for name, plot in [("history", optuna.visualization.matplotlib.plot_optimization_history),
                       ("importance", optuna.visualization.matplotlib.plot_param_importances)]:
        try:
            ax = plot(study)
            ax.figure.set_size_inches(8, 5)
            ax.figure.tight_layout()
            ax.figure.savefig(C.FIGURES / f"task1_optuna_{name}.png", dpi=120)
            plt.close("all")
        except Exception as e:  # importance needs >1 completed trial; don't fail the study over a plot
            print(f"could not draw {name} plot: {e}")

    # Config for the final training run (best params + the final schedule).
    best = study.best_params
    config = {"model": {"base_channels": best["base_channels"], "bottleneck_dim": best["bottleneck_dim"],
                        "dropout": round(best["dropout"], 4)},
              "train": {"lr": float(f"{best['lr']:.3g}"), "batch_size": best["batch_size"],
                        "alpha": round(best["alpha"], 4), "weight_decay": 1e-5, "epochs": 100, "patience": 20},
              "source": f"Optuna study '{STUDY}', best trial #{study.best_trial.number} "
                        f"(val score {study.best_value:.4f})"}
    Path(C.ROOT / "configs").mkdir(exist_ok=True)
    (C.ROOT / "configs" / "task1.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=30, help="total trials wanted in the study (resumes)")
    ap.add_argument("--epochs", type=int, default=15, help="epochs per trial")
    ap.add_argument("--smoke", action="store_true",
                    help="2 trials x 2 epochs, in-memory study, nothing saved (pipeline check only)")
    args = ap.parse_args()

    device = get_device()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if args.smoke:
        setup_mlflow("smoke-tests")
        study = optuna.create_study(direction="minimize")
        with mlflow.start_run(run_name="smoke-task1"):
            study.optimize(safe_objective(make_objective(2, device)), n_trials=2)
        print("smoke OK:", study_summary(study))
        return
    study = create_study(STUDY, n_startup_trials=5, n_warmup_steps=3)
    remaining = args.n_trials - len([t for t in study.trials if t.state.is_finished()])
    print(f"study '{STUDY}': {len(study.trials)} trials so far, running {max(remaining, 0)} more")

    setup_mlflow("task1-universal-ae")
    if remaining > 0:
        with mlflow.start_run(run_name=f"optuna-{STUDY}"):
            mlflow.log_params({"n_trials_target": args.n_trials, "epochs_per_trial": args.epochs,
                               "search_space": json.dumps(SEARCH_SPACE)})
            study.optimize(safe_objective(make_objective(args.epochs, device)), n_trials=remaining,
                           callbacks=[lambda s, t: print(f"trial {t.number}: {t.state.name} "
                                                         f"value={t.value} params={t.params}", flush=True)])
            summary = save_outputs(study, args.epochs)
            mlflow.log_metrics({k: summary[k] for k in ("completed", "pruned", "failed")})
            mlflow.log_artifact(str(C.RESULTS / "task1_optuna_summary.json"))
    else:
        summary = save_outputs(study, args.epochs)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
