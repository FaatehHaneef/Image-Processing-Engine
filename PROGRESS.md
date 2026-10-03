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

## In progress
- Waiting for go-ahead to start Phase 1.

## Next
- Phase 1: foundation (venv + CUDA PyTorch, data pipeline, corruptions, manifests, shared utilities).

## Known issues / decisions pending
- Global Python 3.11.7 has `torch 2.11.0+cpu`. Always use `.venv`.
- `gh` CLI not installed (push works via Git Credential Manager).
- `requirements.txt` not written yet (Phase 1 pins torch 2.14.1+cu126, torchvision 0.29.1+cu126 plus the rest).
- Reminder for user: confirm the real deadline with the instructor.

## Environment snapshot (2026-10-03)
- Windows 11, Python 3.11.7, Node 22.19.0, npm 10.9.3, git 2.50.1
- GPU: RTX 3050 Ti Laptop 4 GB, driver 616.92 (CUDA 13.4); `.venv` torch 2.14.1+cu126
- Docker 28.3.2 engine running, Compose v2.38.2, 7.61 GiB RAM; C: 115 GB free
- Repo: https://github.com/FaatehHaneef/Image-Processing-Engine (main)
