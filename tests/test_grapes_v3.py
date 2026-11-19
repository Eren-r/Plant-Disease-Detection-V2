"""Synthetic, no-dataset tests for the V3 grape module."""
import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from v3.grapes.prepare import CLASSES, label_for, prepare


def test_alias_mapping():
    assert label_for(Path("data/black_rot/sample.jpg")) == "Grapes_Black_rot"
    assert label_for(Path("data/Leaf blight/sample.jpg")) == "Grapes_Leaf_blight"
    assert label_for(Path("data/unknown/sample.jpg")) is None


def test_group_disjoint_manifest(tmp_path):
    root = tmp_path / "raw"
    for label_index, label in enumerate(CLASSES):
        folder = root / label
        folder.mkdir(parents=True)
        for index in range(30):
            pixels = np.random.default_rng(label_index * 1000 + index).integers(
                0, 255, (88, 88, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(folder / f"{index:03d}.png")
    out = tmp_path / "manifests"
    audit = prepare(root, out, radius=0)
    assert audit["images"] == 120
    with (out / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["split"] for row in rows} == {"train", "val", "test"}
    splits = {}
    for row in rows:
        assert splits.setdefault(row["group_id"], row["split"]) == row["split"]


def test_conflicting_duplicate_labels_rejected(tmp_path):
    root = tmp_path / "raw"
    pixels = np.random.default_rng(7).integers(0, 255, (80, 80, 3), dtype=np.uint8)
    for name in CLASSES:
        folder = root / name
        folder.mkdir(parents=True)
        Image.fromarray(pixels).save(folder / "same.png")
    with pytest.raises(ValueError, match="label conflicts"):
        prepare(root, tmp_path / "manifest", radius=0)
