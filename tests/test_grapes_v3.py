"""Synthetic regression tests for the conservative Grapes V3 preparation pipeline."""
import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from v3.grapes.prepare import CLASSES, label_for, prepare


def make_dataset(tmp_path):
    root = tmp_path / "raw"
    for class_number, label in enumerate(CLASSES):
        folder = root / label
        folder.mkdir(parents=True)
        for i in range(30):
            rng = np.random.default_rng(class_number * 1000 + i)
            Image.fromarray(rng.integers(0, 255, (88, 88, 3),
                                        dtype=np.uint8)).save(folder / f"img{i:03d}.png")
    return root


def test_alias_mapping():
    assert label_for(Path("data/black_rot/sample.jpg")) == "Grapes_Black_rot"
    assert label_for(Path("data/Leaf blight/sample.jpg")) == "Grapes_Leaf_blight"
    assert label_for(Path("data/unknown/sample.jpg")) is None


def test_group_disjoint_manifest(tmp_path):
    root = make_dataset(tmp_path)
    audit = prepare(root, tmp_path / "out", radius=0)
    assert audit["images"] == 120
    with (tmp_path / "out" / "manifest.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert {r["split"] for r in rows} == {"train", "val", "test"}
    groups = {}
    for row in rows:
        assert groups.setdefault(row["group_id"], row["split"]) == row["split"]


def test_exact_conflicts_require_opt_in_quarantine(tmp_path):
    root = make_dataset(tmp_path)
    original = root / CLASSES[0] / "img000.png"
    copy = root / CLASSES[1] / "wrong_label.png"
    copy.write_bytes(original.read_bytes())
    with pytest.raises(ValueError, match="label conflicts"):
        prepare(root, tmp_path / "strict", radius=0)
    audit = prepare(root, tmp_path / "quarantined", radius=0,
                    quarantine_exact_conflicts=True)
    assert audit["images"] == 119
    assert audit["quarantined_cross_label_sha_files"] == 2
    assert original.exists() and copy.exists()
    with (tmp_path / "quarantined" / "manifest.csv").open(
            newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    present = {r["relative_path"] for r in rows}
    assert original.relative_to(root).as_posix() not in present
    assert copy.relative_to(root).as_posix() not in present


def test_dhash_collision_does_not_force_label_conflict(tmp_path):
    root = make_dataset(tmp_path)
    Image.new("RGB", (88, 88), (20, 20, 20)).save(root / CLASSES[0] / "dark.png")
    Image.new("RGB", (88, 88), (230, 230, 230)).save(root / CLASSES[1] / "light.png")
    audit = prepare(root, tmp_path / "colors", radius=0)
    assert audit["images"] == 122
    assert audit["weak_dhash_candidates_for_review"] >= 1


def test_opt_in_near_quarantine_retains_all_classes(tmp_path):
    root = make_dataset(tmp_path)
    # Two byte-different but visually identical files under conflicting labels.
    # The strict near-match policy must fail; opt-in provisional quarantine
    # excludes both and leaves raw files untouched.
    first = root / CLASSES[0] / "img000.png"
    second = root / CLASSES[1] / "near_conflict.png"
    with Image.open(first) as image:
        image.convert("RGB").save(second, format="JPEG", quality=97)
    with pytest.raises(ValueError, match="high-similarity"):
        prepare(root, tmp_path / "near_strict", radius=3)
    audit = prepare(root, tmp_path / "near_quarantine", radius=3,
                    quarantine_near_conflicts=True)
    assert audit["provisional_near_candidate_quarantine_used"]
    assert audit["quarantined_near_candidate_files"] >= 2
    assert audit["images"] < 121
    assert first.exists() and second.exists()
    with (tmp_path / "near_quarantine" / "manifest.csv").open(
            newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    included = {r["relative_path"] for r in rows}
    assert first.relative_to(root).as_posix() not in included
    assert second.relative_to(root).as_posix() not in included
