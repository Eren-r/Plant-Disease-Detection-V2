# Docker — Unified Plant Disease Detection V3

This container stack runs **one dashboard** with the existing calibrated
Potato/Tomato V2 classifier and the **separate provisional Grapes V3**
EfficientNet-B0 checkpoint. The existing `/predict`, `/explain` and V2
knowledge base remain intact. The dashboard uses `/predict-crop` for explicit
Potato/Tomato/Grapes routing.

> **Important:** Docker does not train the model. The trained Grapes `.pth`
> must already exist on your PC. Grapes probabilities are uncalibrated, and
> independent vineyard validation is not completed. Treat all predictions
> as screening or research outputs, not definitive diagnoses.

## 1. Windows PC prerequisites

- Docker Desktop running **Linux containers** using its WSL 2 backend.
- At least a few GB of available disk space (PyTorch Docker images are large).
- Your project folder at `D:\PROGRAMMING\Projects\plant disease detection`.
- Your trained Grapes checkpoint at
  `models\grapes\efficientnet_b0_grapes_best.pth`.
- The existing V2 checkpoint
  `models\checkpoints\efficientnet_b0_best.pth` and two
  `reports\results\efficientnet_b0_{calibration,ood}.json` files.
- Internet access when building the image for the first time.

You **do not need your external SSD**. Docker binds the local
`models/grapes` directory into the API container read-only.
No dataset images, Grapes weights, training reports or virtual environment
are copied into the Docker image.

## 2. Pull the Docker files safely

Your local working branch may be `feature/v3-grapes-synced` (tracking
`origin/feature/v3-grapes`). Do not discard the two unmerged commits on
your older local `feature/v3-grapes` branch.

```powershell
cd "D:\PROGRAMMING\Projects\plant disease detection"
git status -sb
git fetch origin
git pull --ff-only origin feature/v3-grapes
```

If Git refuses because you have local modifications or a divergent branch,
preserve those changes first. Do not run `git reset --hard`.

## 3. Verify files and Docker Desktop

```powershell
docker --version
docker compose version
docker info --format "{{.OSType}}"
Test-Path ".\models\checkpoints\efficientnet_b0_best.pth"
Test-Path ".\models\grapes\efficientnet_b0_grapes_best.pth"
Test-Path ".\reports\results\efficientnet_b0_calibration.json"
Test-Path ".\reports\results\efficientnet_b0_ood.json"
docker compose config --quiet
```

Docker must report `linux`. All four `Test-Path` commands should be
`True`. Missing Grapes weights must be supplied locally; they are not in Git.

## 4. Build and start the containers (recommended CPU mode)

Stop your existing local Python-based FastAPI/Streamlit servers on
ports 8000 and 8501, or choose different host ports as shown below.

```powershell
docker compose up --build -d
docker compose ps
docker compose logs --tail 50 api
```

Open:

- Main dashboard: http://localhost:8501
- API documentation: http://localhost:8000/docs
- API crop information: http://localhost:8000/supported-crops

Verify from PowerShell:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/supported-crops | ConvertTo-Json
```

Expected `/supported-crops`: `crops` includes `Potato`, `Tomato`,
`Grapes` and `grapes_checkpoint_available` is `true`.

**If other local services already use 8000 and 8501**, optionally select
different host ports **in the same PowerShell session** before starting:

```powershell
$env:PLANT_HOST_API_PORT = "8011"
$env:PLANT_HOST_UI_PORT = "8510"
docker compose up --build -d
```

Then use http://localhost:8510 for the dashboard and
http://localhost:8011/supported-crops for API crop info.
Inside Docker, Streamlit always calls `http://api:8000`; it does not use
the Windows host port.

The Compose stack uses **one image** for two services:

```text
Windows project folder
    models/grapes/efficientnet_b0_grapes_best.pth
                        |
                        | read-only bind mount
                        v
Docker Compose
    api       FastAPI :8000 --> V2 six-class + Grapes four-class model
    streamlit Streamlit :8501 --> api:8000/predict-crop
```

## 5. Optional NVIDIA GPU mode (Windows + WSL2)

Only try this after CPU mode works and Docker Desktop's WSL2 NVIDIA
GPU support is properly configured. CUDA support in native Windows
PyTorch **does not automatically mean** the Docker engine can see the GPU.

Confirm GPU access from a Linux container (requires a suitable NVIDIA
driver and Docker Desktop GPU support):

```powershell
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

If that command fails, stay in CPU mode. Do not attempt to repair
the application by retraining models.

To rebuild with CUDA-enabled PyTorch and reserve one GPU for the API:

```powershell
docker compose down
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d
docker compose -f docker-compose.yml -f docker-compose.gpu.yml ps
```

The API uses CUDA only if `torch.cuda.is_available()` inside the container.
The GPU image requires much more download and disk space, and your RTX 3050
4 GB GPU has limited VRAM. If it runs out of VRAM, use CPU mode.

Confirm which device FastAPI uses via `/health` (V2) and the
`device` returned in grape prediction JSON.

## 6. Test each crop in the one dashboard

Open http://localhost:8501 (or your overridden host UI port).

- Select **Potato**; upload a potato leaf, then check V2 prediction and
  the existing Grad-CAM.
- Select **Tomato**; upload a tomato leaf and check the V2 output.
- Select **Grapes**; upload a **confirmed grape leaf**, then check that
  its prediction uses the separate Grapes V3 model. The dashboard does not
  show the V2 OOD or Grad-CAM controls for Grapes.

Because the crop is user-selected rather than automatically classified,
check the selection before uploading. Grapes training results were
provisional and need independent field review.

## 7. Troubleshooting, logs and stopping

```powershell
docker compose ps
docker compose logs --tail 100 api
docker compose logs --tail 100 streamlit
docker compose exec api python -c "import torch; print(torch.cuda.is_available())"
docker compose images
docker compose down
```

- If `/supported-crops` returns **404**, check you rebuilt the latest
  feature branch: `docker compose up --build -d`.
- If it returns `grapes_checkpoint_available: false`, verify the host
  path. Container models must be in `/app/models/grapes`, not `D:\...`.
- If `/predict-crop` returns **503** for Grapes, the checkpoint may be
  absent, corrupted, incompatible, or unreadable by the container.
- If the API stays unhealthy, check `docker compose logs api`. The V2
  checkpoint and calibration/OOD JSON files are required for API startup.
- If ports are occupied, stop the local Python processes or override the
  `PLANT_HOST_API_PORT` and `PLANT_HOST_UI_PORT` environment variables.
- If a GPU-enabled build fails or runs out of VRAM, return to the standard
  CPU command after `docker compose down`.

When using the GPU override, include both `-f` options with `down`
if you want Compose to refer to the identical merged configuration.

The trained models and dataset **stay on your PC** after `docker compose down`.
Do **not** use `docker compose down -v` if you later add important named
volumes. Never delete `models/grapes` or the Git checkout merely to rebuild.
