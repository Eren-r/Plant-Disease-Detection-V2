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