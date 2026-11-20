# Phase 1 — Grapes: V3 Crop Expansion

**Status:** Starter code prepared. **No grape model has been trained yet.** The real GVLiD dataset must be downloaded, audited, and supplied to the scripts on the development machine.

## Scope

- Keep existing V2 Potato/Tomato files and checkpoints unchanged.
- Train a standalone 4-class Grapes EfficientNet-B0: `Grapes_Black_rot`, `Grapes_Esca`, `Grapes_healthy`, `Grapes_Leaf_blight`.
- Once independently evaluated, integrate through a crop-specific model registry rather than replacing the existing six-class V2 model.
- This module does **not** identify which crop is pictured. In a later phase, implement crop identification or require the user to select Grapes explicitly; do not feed arbitrary crop images directly into this 4-class classifier.

## Data source and licence

GVLiD v5, DOI `10.17632/wkymf8bhcg.5`, 3,477 published images (verify local counts), CC BY 4.0. Download from https://data.mendeley.com/datasets/wkymf8bhcg/5 . Cite and retain the original source attribution; ensure commercial-use compliance with its licence and any third-party material terms.

The published class names are Healthy, Black rot, Esca and Leaf blight. Place the extracted image folders under a raw dataset path and preserve the source archive. The preparation script recognizes folder names `healthy`, `black_rot`, `esca`, `leaf_blight` (case, spaces and underscores may vary). It recursively searches for image files. If dataset folder naming differs, inspect and update `ALIASES` first. **Never rename disease labels based on guesswork.**

## Folder layout (example: external SSD drive E:)

```text
E:/AgriVision_AI/
  datasets/raw/gvlid/                   <- manually extracted original images
  datasets/manifests/grapes_v1/         <- generated audit + manifest
  models/grapes/efficientnet_b0_grapes_best.pth
  reports/grapes_v1/                    <- training/evaluation reports
```

### 1. Clone or open the existing project on your computer

Use the existing project root, in an activated Python 3.11 environment with `torch`, `torchvision`, `numpy`, `scikit-learn`, `Pillow`, `pytest` installed. Your existing V2 environment contains these. For an RTX 3050 4 GB GPU, begin with `--batch-size 8` or `16`, and `--workers 0` on Windows.

The feature branch is already published on GitHub. On a **fresh clone**, check it out:

```powershell
git clone https://github.com/Eren-r/Plant-Disease-Detection-V2.git
cd Plant-Disease-Detection-V2
git fetch origin
git switch --track origin/feature/v3-grapes
```

If you already have a local copy with unpushed changes, **do not reset it**. First run `git status`, `git branch -vv` and `git fetch origin` and reconcile changes safely. The new modules are committed on the feature branch. Existing V2 modules remain untouched.

### 2. Download the dataset to the external SSD

Extract GVLiD into `E:\AgriVision_AI\datasets\raw\gvlid` without deleting or editing source images.

### 3. Prepare and audit — **required before training**

```powershell
python -m v3.grapes.prepare `
  --raw-root "E:\AgriVision_AI\datasets\raw\gvlid" `
  --out-dir "E:\AgriVision_AI\datasets\manifests\grapes_v1"
```

Review `audit.json`, `rejected.csv`, and `label_conflicts.csv`. If the script rejects images or identifies conflicting labels, fix the mapping/provenance with manual review, not by silently deleting files. The manifest contains paths relative to `--raw-root` and group-disjoint train/validation/test splits.

**Stronger field evaluation:** provide `--groups-csv` with `relative_path,source_group` populated by actual **vineyard/farm/vine or capture-session identifiers**, when source metadata is verified. The script keeps each group in one split. Do not publish private farm IDs, GPS coordinates or personally identifiable metadata without permission. Without source metadata, splitting only protects against detected near/exact duplicates and **is not farm-independent**.

### 4. Run preparer tests

```powershell
python -m pytest tests/test_grapes_v3.py -q
```

### 5. Optional quick training smoke test

```powershell
python -m v3.grapes.train `
  --raw-root "E:\AgriVision_AI\datasets\raw\gvlid" `
  --manifest "E:\AgriVision_AI\datasets\manifests\grapes_v1\manifest.csv" `
  --checkpoint "E:\AgriVision_AI\models\grapes\grapes_smoke.pth" `
  --out-dir "E:\AgriVision_AI\reports\grapes_smoke" `
  --epochs 1 --batch-size 8
```

A smoke test is **not** a final trained or validated model. It tests imports, the GPU environment, data loading and the end-to-end loop. The initial pretrained ImageNet weights require download/caching on first use.

### 6. Full training (after audit approval)

```powershell
python -m v3.grapes.train `
  --raw-root "E:\AgriVision_AI\datasets\raw\gvlid" `
  --manifest "E:\AgriVision_AI\datasets\manifests\grapes_v1\manifest.csv" `
  --checkpoint "E:\AgriVision_AI\models\grapes\efficientnet_b0_grapes_best.pth" `
  --out-dir "E:\AgriVision_AI\reports\grapes_v1" `
  --epochs 15 --batch-size 8
```

Check `grapes_test_results.json` including per-class precision, recall, F1, and the confusion matrix. **Do not claim the same accuracy as V2**; this is a different dataset and problem. Checkpoint and test output remain on SSD until reviewed.

### 7. Run a prediction on a known grape leaf

After a successful full training run:

```powershell
python -m v3.grapes.predict `
  --image "E:\AgriVision_AI\datasets\raw\gvlid\healthy\example.jpg" `
  --checkpoint "E:\AgriVision_AI\models\grapes\efficientnet_b0_grapes_best.pth"
```

Replace the example image with a **real file path** from the manifest. The prediction CLI assumes a grape leaf is already confirmed; calibration, OOD gating and integration into the main FastAPI service are later tasks.

### 8. Launch standalone Grapes browser demo (after training)

This experimental Streamlit page **does not modify** the existing V2 FastAPI or Streamlit endpoints.

```powershell
python -m streamlit run v3/grapes/demo.py --server.port 8502
```

Open `http://localhost:8502`, paste the full trained grape checkpoint path such as `E:\\AgriVision_AI\\models\\grapes\\efficientnet_b0_grapes_best.pth`, and upload a **known grape leaf**. The output is a four-class experimental inference, not a crop detector. **No grape-specific confidence calibration or OOD rejection is implemented yet.**

### 9. Commit code and metadata to GitHub

```powershell
git add v3/ knowledge_base/grapes.json tests/test_grapes_v3.py README_GRAPES_V3.md
git commit -m "Add grape dataset audit and separate EfficientNet-B0 training pipeline"
git push origin feature/v3-grapes
```

Draft pull request #1 already exists. Review tests and data provenance, and merge only when ready. **Do not commit raw images, private farm metadata or large model checkpoints to normal Git history.** Add `datasets/`, `*.pth`, and raw image folders to `.gitignore` where appropriate; don't accidentally exclude `knowledge_base/grapes.json`. Use DVC, artifact storage, or a GitHub Release for an approved model if distribution is required. Keep the existing Potato/Tomato disease database intact, and merge the expert-reviewed grape knowledge base via a crop-aware service in a later integration phase.

## Dataset conflict handling — November 2026 revision

The first audit of the local GVLiD extraction flagged 399 cross-label candidate pairs with
`--near-distance 3`. A repeat audit with `--near-distance 0` flagged 309 pairs.
Among the 399 candidates, a separate SHA-256 investigation found 18 duplicate pair
**entries** (not necessarily 18 distinct SHA groups), predominantly Black Rot versus Esca,
and 364 candidate pairs between Esca and Leaf Blight.

The original dHash-only check generated many weak candidates; even distance 0
does not prove two photographs are identical. The revised pipeline distinguishes:

- `exact_label_conflicts.csv`: every file belonging to a byte-identical
  SHA-256 group whose files carry conflicting class labels. These are blocking
  issues by default.
- `quarantined.csv`: only exact cross-label SHA groups, explicitly excluded
  from the manifest with `--quarantine-exact-conflicts`. The raw files are
  not deleted or renamed. Review this list with dataset provenance before
  approving any training run.
- `weak_dhash_candidates.csv`: suspect cross-label pairs flagged by dHash
  but not supported by the independent pHash + RGB test. Keep under review;
  a collision alone is not ground truth.
- `near_label_conflicts.csv`: cross-label pairs supported by pHash distance
  and low normalized 32×32 RGB difference. These **still require human review**,
  and block the manifest.
- `manifest.csv`: produced only when no strong unresolved conflicts remain
  and exact cross-label issues have been reviewed/quarantined.

Run from the project root (the data stays under the existing local project):

```powershell
python -m pytest tests/test_grapes_v3.py -q

python -m v3.grapes.prepare `
  --raw-root "data\grapes\raw" `
  --out-dir "data\grapes\manifests\gvlid_v2" `
  --quarantine-exact-conflicts
```

Review `audit.json`, `quarantined.csv`, `near_label_conflicts.csv`
and `weak_dhash_candidates.csv`. If it still stops, **do not train**
or bypass the new checks: review the listed candidates, confirm provenance,
and agree on an explicit source-preserving resolution first. For training,
change `--manifest` to the new `gvlid_v2\manifest.csv` path.

This is an internal duplicate-aware split, **not** proof of field-independent
validation. Verified farm/vine/session grouping and independent field testing
remain necessary before company deployment.


### Provisional baseline when cross-label *near* candidates remain unresolved

For the local GVLiD dataset, the revised audit reported **381** cross-label near
candidates involving **554** distinct images after exact conflicts were quarantined.
These are candidate similarities, **not confirmed incorrect labels**.

A separate, explicitly opt-in **provisional** path can exclude both ends of
every flagged cross-label pair, together with their known same-label duplicate
and source-group mates. **Raw photos are never modified.** The affected images
are documented in `quarantined.csv`. This option intentionally sacrifices
coverage and may bias performance; use **only for an internal GPU prototype**.

```powershell
python -m pytest tests/test_grapes_v3.py -q
python -m v3.grapes.prepare `
  --raw-root "data\grapes\raw" `
  --out-dir "data\grapes\manifests\gvlid_provisional_v1" `
  --quarantine-exact-conflicts `
  --quarantine-near-conflicts
```

Inspect `audit.json` for images retained per class, overall group count,
exclusion counts and split sizes. If it fails because a class has insufficient
independent groups, **do not reduce the grouping constraints**. Request source
review instead. Keep `quarantined.csv` and all source conflict reports.

Once the manifest exists, a one-epoch *internal smoke test* can use
`data\grapes\manifests\gvlid_provisional_v1\manifest.csv`.
The resulting weights and metrics must be labelled **provisional**, and
cannot be used to claim field generalization or production readiness.
Independent expert review of the original Esca / Leaf Blight labels,
better provenance and a separate field test are necessary.


## Unified main dashboard (Potato + Tomato + Grapes)

The `feature/v3-grapes` branch now integrates the locally trained grape model
into the existing FastAPI + Streamlit dashboard. **It does not replace the V2
model**. The old `POST /predict` and `POST /explain` routes retain their V2
contracts. The new `POST /predict-crop` route accepts a form field
`crop` with one of `Potato`, `Tomato`, or `Grapes`, and an uploaded
image under `file`. `GET /supported-crops` reports support and whether the
local Grapes checkpoint exists.

**Local Windows setup — from the main project directory:**

```powershell
cd "D:\PROGRAMMING\Projects\plant disease detection"
git status
git pull --ff-only origin feature/v3-grapes
.\.venv\Scripts\Activate.ps1
Test-Path ".\models\grapes\efficientnet_b0_grapes_best.pth"
python -m pytest tests/test_crop_dashboard.py tests/test_grapes_dashboard_service.py tests/test_api.py -q
```

The checkpoint test must return `True`. If the file is elsewhere, set
`GRAPES_MODEL_PATH` in the terminal **before starting FastAPI**, for example:

```powershell
$env:GRAPES_MODEL_PATH = "D:\PROGRAMMING\Projects\plant disease detection\models\grapes\efficientnet_b0_grapes_best.pth"
```

Run both services from the main project root, in separate activated terminals:

```powershell
python -m uvicorn app.api:app --reload --port 8000
```

```powershell
python -m streamlit run app/streamlit_app.py --server.port 8501
```

Open `http://localhost:8501`. In the **Select the crop to analyze**
dropdown, choose **Potato**, **Tomato**, or **Grapes**, upload a matching
leaf image, and click **Analyze Leaf**. The single dashboard now routes
Grapes to the V3 four-class checkpoint; Potato/Tomato remain on the
calibrated six-class V2 model. For Potato/Tomato, an apparent selected/predicted
crop mismatch is reported explicitly instead of silently showing a wrong-crop
disease. For Grapes, only verified grape leaves should be uploaded because
the crop is assumed from user selection, not automatically detected.

Grapes results have **uncalibrated** scores and a visibly experimental
warning; V2-only OOD and Grad-CAM are not claimed for the grape model.
The model loads lazily only when Grapes is first used, so the V2 application
can still start if the local grape checkpoint is missing. A missing checkpoint
returns HTTP 503 on a Grapes prediction.

Troubleshooting:

- If the frontend gets HTTP 404 at `/predict-crop`, restart the **FastAPI**
  server after pulling this branch. The old backend does not have that route.
- If Grapes returns HTTP 503, verify `Test-Path
  ".\models\grapes\efficientnet_b0_grapes_best.pth"` and the
  `GRAPES_MODEL_PATH` environment variable in the API terminal.
- If a Potato/Tomato choice gets HTTP 422, the existing six-class V2 model
  predicted the other crop. Inspect the image and selection; this is not
  automatic crop verification.
- If ports are in use, stop the existing Streamlit `8501` and FastAPI
  `8000` processes before restarting.
- The separate `v3/grapes/demo.py` at `8502` is now optional.

**Validation reminder:** The provisional grape holdout reported about 99.08%
accuracy and 98.40% macro F1 on its internal grouped split, but 572 images
were quarantined and verified independent farm grouping was unavailable.
These are **not** field-ready or production-certification metrics. Review
the original label conflicts, collect independent vineyard data, and validate
crop selection, calibration and OOD policies before production deployment.
