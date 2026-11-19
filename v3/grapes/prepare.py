"""Audit four-class grape images; write a grouped, reproducible manifest.

From project root:
python -m v3.grapes.prepare --raw-root E:/AgriVision_AI/datasets/raw/gvlid --out-dir E:/AgriVision_AI/datasets/manifests/grapes_v1
Optionally supply --groups-csv with relative_path,source_group columns (verified farm/vine/session IDs).
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit

CLASSES = ("Grapes_Black_rot", "Grapes_Esca", "Grapes_healthy", "Grapes_Leaf_blight")
ALIASES = {"blackrot": CLASSES[0], "grapesblackrot": CLASSES[0],
           "esca": CLASSES[1], "grapesesca": CLASSES[1],
           "healthy": CLASSES[2], "grapeshealthy": CLASSES[2],
           "leafblight": CLASSES[3], "grapesleafblight": CLASSES[3]}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def normalized(value):
    return re.sub(r"[^a-z0-9]", "", value.lower())


def label_for(relative_path):
    found = {ALIASES[normalized(part)] for part in relative_path.parts[:-1]
             if normalized(part) in ALIASES}
    return next(iter(found)) if len(found) == 1 else None


def dhash(image):
    arr = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    value = 0
    for flag in (arr[:, 1:] > arr[:, :-1]).flat:
        value = (value << 1) | int(flag)
    return value


class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            if self.size[a] < self.size[b]:
                a, b = b, a
            self.parent[b] = a
            self.size[a] += self.size[b]


class BKTree:
    def __init__(self):
        self.root = None

    def add(self, value, index):
        if self.root is None:
            self.root = [value, [index], {}]
            return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                node[1].append(index)
                return
            children = node[2]
            if distance not in children:
                children[distance] = [value, [index], {}]
                return
            node = children[distance]

    def query(self, value, radius):
        stack = [self.root] if self.root is not None else []
        hits = []
        while stack:
            node = stack.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= radius:
                hits.extend(node[1])
            for edge, child in node[2].items():
                if distance - radius <= edge <= distance + radius:
                    stack.append(child)
        return hits


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def source_groups(csv_path):
    if csv_path is None:
        return {}
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"relative_path", "source_group"}.issubset(reader.fieldnames or []):
            raise ValueError("Groups CSV requires relative_path and source_group")
        return {r["relative_path"].replace("\\", "/"): r["source_group"] for r in reader}


def best_group_split(labels, groups, fraction, seed):
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    if len(set(groups.tolist())) < 3:
        raise ValueError("Too few independent groups for splitting")
    candidates = GroupShuffleSplit(n_splits=40, test_size=fraction, random_state=seed)
    desired = set(labels.tolist())
    best = None
    for kept, holdout in candidates.split(np.zeros(len(labels)), labels, groups):
        if set(labels[kept]) != desired or set(labels[holdout]) != desired:
            continue
        score = sum(abs(np.mean(labels[holdout] == label) - np.mean(labels == label))
                    for label in desired)
        score += abs(len(holdout) / len(labels) - fraction)
        if best is None or score < best[0]:
            best = (score, kept, holdout)
    if best is None:
        raise ValueError("Cannot make class-complete grouped split; inspect labels and grouping")
    return best[1], best[2]


def prepare(raw_root, out_dir, radius=3, groups_csv=None):
    raw_root = Path(raw_root)
    out_dir = Path(out_dir)
    if not raw_root.is_dir():
        raise FileNotFoundError(raw_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = source_groups(groups_csv)
    records, rejected = [], []
    images = sorted(p for p in raw_root.rglob("*")
                    if p.is_file() and p.suffix.lower() in EXTENSIONS)
    for file in images:
        rel = file.relative_to(raw_root).as_posix()
        label = label_for(Path(rel))
        if label is None:
            rejected.append({"relative_path": rel, "reason": "unknown_or_ambiguous_folder_label"})
            continue
        try:
            with Image.open(file) as image:
                image.load()
                if min(image.size) < 64:
                    raise ValueError("image smaller than 64px")
                phash, (width, height) = dhash(image), image.size
            digest = hashlib.sha256(file.read_bytes()).hexdigest()
        except (OSError, ValueError) as exc:
            rejected.append({"relative_path": rel, "reason": "invalid_image: " + str(exc)[:100]})
            continue
        records.append({"relative_path": rel, "label": label, "sha256": digest,
                        "dhash": f"{phash:016x}", "width": width, "height": height,
                        "source_group": metadata.get(rel, "")})
    write_csv(out_dir / "rejected.csv", rejected, ["relative_path", "reason"])
    if rejected:
        raise ValueError(f"{len(rejected)} rejected images: inspect rejected.csv first")
    counts = Counter(r["label"] for r in records)
    if any(counts[cls] == 0 for cls in CLASSES):
        raise ValueError("One or more grape labels absent. Verify folder names.")
    dsu, index, exact, sources, near_pairs = DSU(len(records)), BKTree(), {}, {}, []
    for i, record in enumerate(records):
        if record["sha256"] in exact:
            dsu.union(i, exact[record["sha256"]])
        exact.setdefault(record["sha256"], i)
        h = int(record["dhash"], 16)
        for j in index.query(h, radius):
            dsu.union(i, j)
            if records[i]["label"] != records[j]["label"]:
                near_pairs.append((records[j]["relative_path"], record["relative_path"]))
        index.add(h, i)
        group = record["source_group"]
        if group:
            if group in sources:
                dsu.union(i, sources[group])
            sources.setdefault(group, i)
    conflicts = []
    for i, record in enumerate(records):
        for j in range(i):
            if record["sha256"] == records[j]["sha256"] and record["label"] != records[j]["label"]:
                conflicts.append({"image_a": records[j]["relative_path"], "image_b": record["relative_path"]})
    conflicts.extend({"image_a": a, "image_b": b} for a, b in near_pairs)
    write_csv(out_dir / "label_conflicts.csv", conflicts, ["image_a", "image_b"])
    if conflicts:
        raise ValueError(f"{len(conflicts)} label conflicts: inspect label_conflicts.csv")
    groups = [f"g{dsu.find(i):06d}" for i in range(len(records))]
    labels = [r["label"] for r in records]
    remaining, test = best_group_split(labels, groups, .15, 42)
    train_local, val_local = best_group_split(
        [labels[i] for i in remaining], [groups[i] for i in remaining], .15 / .85, 84)
    assignment = {int(i): "test" for i in test}
    assignment.update({int(remaining[i]): "train" for i in train_local})
    assignment.update({int(remaining[i]): "val" for i in val_local})
    for i, r in enumerate(records):
        r.update({"group_id": groups[i], "split": assignment[i]})
    splits = defaultdict(set)
    for r in records:
        splits[r["group_id"]].add(r["split"])
    assert all(len(x) == 1 for x in splits.values()), "Group leakage found"
    write_csv(out_dir / "manifest.csv", records, [
        "relative_path", "label", "split", "group_id", "sha256", "dhash", "width",
        "height", "source_group"])
    audit = {"source": "GVLiD Mendeley Data v5", "doi": "10.17632/wkymf8bhcg.5",
             "license": "CC BY 4.0; retain attribution", "images": len(records),
             "class_counts": dict(counts), "splits": dict(Counter(r["split"] for r in records)),
             "groups": len(splits), "near_duplicate_radius": radius,
             "farm_independent": bool(metadata),
             "warning": "Source groups must be verified; content-only duplicate groups do not demonstrate field-level generalization."}
    (out_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--groups-csv", type=Path)
    parser.add_argument("--near-distance", type=int, default=3)
    args = parser.parse_args()
    if not 0 <= args.near_distance <= 8:
        parser.error("--near-distance must be from 0 to 8")
    prepare(args.raw_root, args.out_dir, args.near_distance, args.groups_csv)


if __name__ == "__main__":
    main()
