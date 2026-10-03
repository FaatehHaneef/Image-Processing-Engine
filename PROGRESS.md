# Progress

Resume guide: read `CLAUDE.md`, then `docs/explanation.md`, then `docs/implementation_plan.md`, then this file.

## Done
- **Phase 0, setup and planning (2026-10-03)**
  - Read `docs/assignment.pdf` (PDFs were already in `docs/`).
  - Read-only dataset inspection: `scripts/inspect_datasets.py`, results in `docs/dataset_findings.md`.
  - Environment and git checks (see Known issues).
  - Wrote `docs/explanation.md` (knowledge base) and `docs/implementation_plan.md` (7 phases), plus `docs/ai_use_log.md`.

- **Plan approved + environment (2026-10-03)**
  - NVIDIA driver updated by user to 616.92 (CUDA 13.4).
  - `.venv` (Python 3.11.7) with torch 2.14.1+cu126, torchvision 0.29.1+cu126. Verified: `torch.cuda.is_available()` = True, CUDA 12.6, cuDNN 9.10, RTX 3050 Ti 4 GiB, GPU matmul + fp16 autocast conv OK.
  - Approved decisions recorded in `docs/explanation.md` section 13; CLAUDE.md section 0 updated to 7 phases; plan updated (AI-log entry per phase, deadline reminder, committed artifacts).

- **Phase 1, foundation (2026-10-03)**
  - `requirements.txt` pinned; packages installed in `.venv`.
  - `src/`: config, corruptions (NumPy-only), Pet data + cache, FS2K data + paired augmentation, balanced sampler, SSIM/L1 loss, metrics + fixed Optuna objective, MLflow (SQLite) + Optuna helpers (OOM/divergence -> pruned).
  - `scripts/prepare_data.py` built: Pet split 2,944/736; cache (181 + 180 MB, gitignored); val manifest 736 (184/class); test manifest 36,690; FS2K split 899/159/1,046 (val styles 54/52/53). Re-run verified byte-identical.
  - `docs/figures/corruption_grid.png`; 22 pytest tests pass.
  - Benchmark (logged to MLflow `phase1-benchmark`): AE 2.8-2.9 s/epoch, GAN 7.9 s/epoch (2.71 GB peak). Plan estimates updated.
  - Fixed `.gitignore`: `data/` was also hiding `src/data/` -> now `/data/`.

## In progress
- Waiting for go-ahead to start Phase 2.

## Next
- Phase 2: Task 1 universal autoencoder (model, Optuna, final training, evaluation, ONNX export + verify tooling).

## Known issues / decisions pending
- Global Python 3.11.7 has `torch 2.11.0+cpu`. Always use `.venv`.
- `gh` CLI not installed (push works via Git Credential Manager).
- MLflow UI: `.venv\Scripts\mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`
- After a fresh clone, run `scripts/prepare_data.py` once to rebuild the (gitignored) image cache.
- Reminder for user: confirm the real deadline with the instructor.

## Environment snapshot (2026-10-03)
- Windows 11, Python 3.11.7, Node 22.19.0, npm 10.9.3, git 2.50.1
- GPU: RTX 3050 Ti Laptop 4 GB, driver 616.92 (CUDA 13.4); `.venv` torch 2.14.1+cu126
- Docker 28.3.2 engine running, Compose v2.38.2, 7.61 GiB RAM; C: 115 GB free
- Repo: https://github.com/FaatehHaneef/Image-Processing-Engine (main)
