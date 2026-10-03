# AI-use log

Running record for the report's AI-use appendix: what AI generated, and how it was checked or corrected.

| Date | Tool | What was produced | How it was verified / corrected |
|---|---|---|---|
| 2026-10-03 | Claude Code (Claude Opus 5.5) | Requirement summary (`docs/explanation.md`), phased plan (`docs/implementation_plan.md`), read-only dataset inspection script (`scripts/inspect_datasets.py`) and findings (`docs/dataset_findings.md`), environment/git checks | Requirements cross-checked against `docs/assignment.pdf`. Dataset facts come from running the script on the real files (counts match the official lists; FS2K style counts match the README except a 1-image swap between styles 2 and 3 in the README table). *To do (student): review and confirm.* |
| 2026-10-03 | Claude Code (Claude Opus 5.5) | Recorded the approved design decisions (`docs/explanation.md` section 13), updated the CLAUDE.md phase list and the plan, created `.venv` with torch 2.14.1+cu126 | `nvidia-smi` checked for CUDA >= 12 before installing; `torch.cuda.is_available()` and a small GPU tensor operation run in `.venv`. *To do (student): review the decisions text.* |
