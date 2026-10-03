# Progress

Resume guide: read `CLAUDE.md`, then `docs/explanation.md`, then `docs/implementation_plan.md`, then this file.

## Done
- **Phase 0, setup and planning (2026-10-03)**
  - Read `docs/assignment.pdf` (PDFs were already in `docs/`).
  - Read-only dataset inspection: `scripts/inspect_datasets.py`, results in `docs/dataset_findings.md`.
  - Environment and git checks (see Known issues).
  - Wrote `docs/explanation.md` (knowledge base) and `docs/implementation_plan.md` (7 phases), plus `docs/ai_use_log.md`.

## In progress
- Waiting for approval of the plan.

## Next
- Phase 1: foundation (venv + CUDA PyTorch, data pipeline, corruptions, manifests, shared utilities).

## Known issues / decisions pending
- GPU driver 512.77 (CUDA 11.6) is too old for current PyTorch CUDA 12/13 builds. Options: update the driver (recommended), or use torch 2.7.1+cu118. Needs a decision before Phase 1.
- Global Python 3.11.7 has `torch 2.11.0+cpu`. Always use `.venv`.
- `gh` CLI not installed (push works via Git Credential Manager).
- CLAUDE.md section 0 lists 10 phases; the approved plan may use 7. Update CLAUDE.md after approval.
- Open design choices (see `docs/explanation.md` section 12): validation manifest design, direct resize vs aspect-preserving, committing small artifacts (splits/manifests/Optuna DBs).

## Environment snapshot (2026-10-03)
- Windows 11, Python 3.11.7, Node 22.19.0, npm 10.9.3, git 2.50.1
- GPU: RTX 3050 Ti Laptop 4 GB, driver 512.77
- Docker 28.3.2 engine running, Compose v2.38.2, 7.61 GiB RAM; C: 115 GB free
- Repo: https://github.com/FaatehHaneef/Image-Processing-Engine (main)
