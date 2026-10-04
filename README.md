# Image Processing Engine

Four generative models for image restoration and face-to-sketch generation, served through one web app.

| Workspace | Task | Model |
|---|---|---|
| **Universal Restoration** | 1 | One convolutional denoising autoencoder (compressed 8x8 latent bottleneck, no skip connections) that removes salt-and-pepper noise, Gaussian blur and rectangular occlusion |
| **Hard-Routed Restoration** | 2 | A CNN classifier picks the corruption, then exactly one specialist autoencoder (or an identity bypass for clean images) restores it |
| **Soft Mixture-of-Experts Restoration** | 3 | A gate (initialized from the classifier) blends identity + the three specialists with continuous weights; fine-tuned end to end |
| **Face-to-Sketch Generator** | 4 | Style-conditioned pix2pix cGAN (U-Net generator + PatchGAN discriminator) trained on FS2K; three sketch styles |

Stack: PyTorch (training) -> ONNX -> FastAPI + ONNX Runtime (CPU) backend -> React + Tailwind frontend, all started with Docker Compose. Hyperparameters tuned with Optuna, experiments tracked with MLflow.

---

## Run the app (Docker)

Requirements: [Git](https://git-scm.com/) and [Docker Desktop](https://www.docker.com/products/docker-desktop/) (running). No Python or Node needed.

**1. Clone**
```bash
git clone https://github.com/FaatehHaneef/Image-Processing-Engine.git
cd Image-Processing-Engine
```

**2. Get the one large model file** (168 MB, too large for the git repository; all other models are already included in `models/onnx/`)

Windows (PowerShell):
```powershell
curl.exe -L -o models/onnx/task4_generator.onnx https://github.com/FaatehHaneef/Image-Processing-Engine/releases/download/models-v1/task4_generator.onnx
```
macOS / Linux:
```bash
curl -L -o models/onnx/task4_generator.onnx https://github.com/FaatehHaneef/Image-Processing-Engine/releases/download/models-v1/task4_generator.onnx
```
(Or download it from the [models-v1 release](https://github.com/FaatehHaneef/Image-Processing-Engine/releases/tag/models-v1) and put it in `models/onnx/`.) Without it, the three restoration workspaces work and Face-to-Sketch reports that its model is missing.

**3. Start**
```bash
docker compose up --build
```
The first build takes a few minutes (it downloads the base images and installs dependencies). Then open **http://localhost:8080**.

The top-right corner shows **MODELS LOADED** when all 7 models are available (the **System** page lists each one). Stop with `Ctrl+C`, then `docker compose down`.

### What you can do in the app
- Upload a JPG/PNG (up to 10 MB), pick one of the bundled samples, or (Face-to-Sketch) use the webcam.
- Apply salt-and-pepper, Gaussian blur or occlusion at the test severities (low / medium / high), or mark an upload as already corrupted.
- See the restored image, an error map (when the clean original is known), classifier probabilities (Hard-Routed), mixture weights and a routing diagram (Soft-MoE), the inference time, and download the result.

---

## Repository layout

```
src/                 shared training code: data pipeline, corruptions, models, losses, metrics, Optuna/MLflow helpers
scripts/             prepare_data, optuna_task*, train_task*, evaluate_task*, export_onnx, verify_onnx, ...
configs/             final hyperparameters chosen by Optuna (task1.yaml, task2_*.yaml, task3.yaml, task4.yaml)
artifacts/           splits/ and manifests/ (fixed data splits and test corruptions), optuna/ (studies), results/ (metrics, tables)
models/onnx/         exported ONNX models (task4_generator.onnx via the release, see above)
backend/             FastAPI app, sample images, tests, Dockerfile
frontend/            React + Tailwind app, nginx config, Dockerfile
docs/                assignment, design decisions (explanation.md), API contract, dataset findings, figures, AI-use log
tests/               pytest unit tests for data, corruptions, losses and models
docker-compose.yml
```

---

## Reproduce training and evaluation (optional)

Training runs locally on an NVIDIA GPU (developed on a 4 GB RTX 3050 Ti), not in Docker.

**Setup** (Python 3.11):
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (macOS/Linux: source .venv/bin/activate)
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```
**Data** (not included in the repository): download the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) into `data/oxford-iiit-pet/` (`images/`, `annotations/`) and [FS2K](https://github.com/DengPingFan/FS2K) into `data/fs2k/FS2K/` (`photo/`, `sketch/`, `anno_train.json`, `anno_test.json`). Then:
```bash
python scripts/prepare_data.py          # split (seed 42), image cache, validation + test corruption manifests, FS2K split
python -m pytest                        # unit tests
```
**Per task** (each Optuna script writes `configs/<task>.yaml`, which the training script reads):
```bash
python scripts/optuna_task1.py --n-trials 30            && python scripts/train_task1.py            && python scripts/evaluate_task1.py
python scripts/optuna_task2_classifier.py --n-trials 25 && python scripts/train_task2_classifier.py
python scripts/optuna_task2_specialists.py --n-trials 20 && python scripts/train_task2_specialists.py && python scripts/evaluate_task2.py
python scripts/optuna_task3.py --n-trials 20            && python scripts/train_task3.py            && python scripts/evaluate_task3.py
python scripts/optuna_task4.py --n-trials 20            && python scripts/train_task4.py            && python scripts/evaluate_task4.py
python scripts/export_onnx.py --all && python scripts/verify_onnx.py --all
```
Evaluation scripts use the official test sets exactly once and refuse to overwrite existing test results without `--force`.

**Experiment tracking:** `mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`, then open http://127.0.0.1:5000 (the app's **Experiments** link points there).

**Run the app without Docker** (development):
```bash
python -m uvicorn backend.app.main:app --port 8000      # terminal 1
cd frontend && npm install && npm run dev               # terminal 2 -> http://localhost:5173
```

---

## Data and attribution

- **Oxford-IIIT Pet Dataset**: O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar, "Cats and Dogs", CVPR 2012. License CC BY-SA 4.0. The 16 sample images in `backend/samples/pets/` are resized copies of test-set images.
- **FS2K**: D.-P. Fan et al., "Facial-Sketch Synthesis: A New Challenge", Machine Intelligence Research, 2022 (code under the MIT license). The 6 sample faces in `backend/samples/faces/` are resized copies of FS2K test photos from its free-stock-photo subset.
- The landing-page illustrations are design artwork from the Google Stitch mock-up, not model outputs.
