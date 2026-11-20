# Project checkpoint — 20 November 2026 (India)

## Current project

**Plant Disease Detection V3** — unified Streamlit + FastAPI dashboard with **Potato, Tomato and Grapes**.

- GitHub repository: https://github.com/Eren-r/Plant-Disease-Detection-V2
- Working remote branch: `feature/v3-grapes` (not merged to `main`)
- Draft pull request: https://github.com/Eren-r/Plant-Disease-Detection-V2/pull/1
- Windows project folder: `D:\PROGRAMMING\Projects\plant disease detection`
- User confirmed that the **non-containerized unified dashboard works locally**, with three-crop selection.
- Earlier the local `feature/v3-grapes` branch diverged from origin (two unique local commits). A safe `feature/v3-grapes-synced` branch tracking origin was advised. **Preserve all local commits; do not reset or force push.**

## Models and data

**Potato / Tomato V2**
- Existing six-class EfficientNet-B0 model and calibration/OOD files are unchanged.
- Local checkpoint: `models/checkpoints/efficientnet_b0_best.pth`.
- V2 routes `/predict` and `/explain` are retained.

**Grapes V3**
- Four disease/status classes: Black Rot, Esca, Healthy, Leaf Blight.
- Local checkpoint: `models/grapes/efficientnet_b0_grapes_best.pth`.
- User trained on RTX 3050 GPU, best epoch **3**; model successfully predicted via standalone Streamlit and was subsequently integrated into the main dashboard.
- Provisional GVLiD v5 source: **3,477** images; **572** quarantined without deleting originals; **2,905** retained.
- Provisional split: **2,029** train / **440** validation / **436** test; grouping present but **0 verified source-group rows**.
- Test accuracy **0.9908256880733946** (~99.08%); test macro F1 **0.9839968774395005** (~98.40%).
- Grapes output remains **experimental** and *uncalibrated*; no grape-specific OOD rejection or Grad-CAM; field validation and agronomist review still pending.
- Data stays locally under `data/grapes/`; local weights and training results are not part of the GitHub repository.

## Unified application

- FastAPI `app/api.py` has `GET /supported-crops` and `POST /predict-crop` (explicit crop form field), while preserving V2 routes.
- Streamlit `app/streamlit_app.py` has Potato/Tomato/Grapes selection.
- Grapes API loads the local checkpoint lazily through `v3/grapes/service.py`.
- The user confirmed the **working unified Python dashboard** after resolving divergent Git branches. If stale routes reappear, inspect `git status -sb` and Python's actual route list.

## Docker work completed this session

Updated on remote `feature/v3-grapes`:
- `Dockerfile`: includes app, src, v3 and knowledge base; supports CPU wheels by default or CUDA GPU wheels via build arg.
- `docker-compose.yml`: two services (`api`, `streamlit`) using the same CPU image; mounts host `./models/grapes` **read-only** into `/app/models/grapes`; loopback host ports 8000 and 8501, overridable.
- `docker-compose.gpu.yml`: optional NVIDIA GPU build/Compose override.
- `.dockerignore`: excludes local Grapes weights, datasets, sensitive local state and training output from Docker build context; includes the versioned V2 checkpoint.
- `DOCKER_V3.md`: Windows Docker Desktop setup and troubleshooting.
- `.github/workflows/docker-v3.yml`: builds CPU Docker image, boots FastAPI/Streamlit, verifies health and supported-crop endpoint, stops services.
- Sidebar changed to say CPU or CUDA depending on backend.
- **GitHub Docker V3 smoke-test run 44785166740 passed** and **Python CI run 44785166821 passed** at commit `12550480cc07efe2a737c9f49af77312fd292663`. CI used an empty Grapes model mount; it **did not test real Grapes inference**.
- Docker configuration has NOT yet been confirmed running on the user's Windows PC. GPU Docker mode has NOT been verified.

## Tomorrow: concrete next steps

1. **Preserve working local state**: `git status -sb`; verify current synced branch and local untracked files. Reconcile any local differences safely, without deleting data, weights or the two earlier local commits.
2. Verify Docker Desktop uses WSL2 Linux containers:
   ```powershell
   docker --version
   docker compose version
   docker info --format "{{.OSType}}"
   ```
3. Verify required files locally:
   ```powershell
   Test-Path ".\models\checkpoints\efficientnet_b0_best.pth"
   Test-Path ".\models\grapes\efficientnet_b0_grapes_best.pth"
   Test-Path ".\reports\results\efficientnet_b0_calibration.json"
   Test-Path ".\reports\results\efficientnet_b0_ood.json"
   ```
4. Stop native FastAPI/Streamlit processes on ports 8000/8501, or choose overrides. From project root:
   ```powershell
   docker compose config --quiet
   docker compose up --build -d
   docker compose ps
   docker compose logs --tail 100 api
   ```
5. Verify `http://localhost:8501` and `http://localhost:8000/supported-crops`, including `grapes_checkpoint_available=true`, then run a real inference for **each** Potato, Tomato and Grapes.
6. Only after CPU Docker works, optionally try GPU mode (Docker Desktop WSL2 NVIDIA prerequisite). Command:
   `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d`.
7. Once Docker end-to-end succeeds, decide on model/checkpoint backups and continue remaining roadmap: Grapes Grad-CAM, calibration, grape-specific unknown-image detection, independent vineyard validation, then Pomegranate, Papaya and Mango.

**Important:** This Markdown checkpoint is the saved *project status*, not a backup of the binary .pth model files. The user's local model weights and datasets must be preserved/backed up separately on their PC. Do not claim they are on GitHub.
