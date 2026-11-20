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
from scipy.fft import dctn
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



def phash(image):
    """Second perceptual hash used only with an RGB image-difference check."""
    arr = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    coefficients = dctn(arr, type=2, norm="ortho")[:8, :8].flatten()
    median = np.median(coefficients[1:])
    result = 0
    for bit in coefficients > median:
        result = (result << 1) | int(bit)
    return result


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


def prepare(raw_root, out_dir, radius=3, groups_csv=None,
            quarantine_exact_conflicts=False, phash_distance=5, rgb_mae_max=0.06):
    """Build a training manifest after explicitly handling cross-label exact duplicates.

    Never edits/deletes raw photos. Weak dHash candidates are REVIEW FLAGS,
    not proof of duplication. Strong near matches require both pHash and RGB agreement.
    """
    raw_root, out_dir = Path(raw_root), Path(out_dir)
    if not raw_root.is_dir():
        raise FileNotFoundError(raw_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = source_groups(groups_csv)
    records, rejected, pixels = [], [], []
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
                rgb = image.convert("RGB")
                h_d, h_p, (width, height) = dhash(rgb), phash(rgb), image.size
                small = np.asarray(rgb.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.uint8)
            digest = hashlib.sha256(file.read_bytes()).hexdigest()
        except (OSError, ValueError) as exc:
            rejected.append({"relative_path": rel, "reason": "invalid_image: " + str(exc)[:100]})
            continue
        records.append({"relative_path": rel, "label": label, "sha256": digest,
                        "dhash": f"{h_d:016x}", "phash": f"{h_p:016x}",
                        "width": width, "height": height,
                        "source_group": metadata.get(rel, "")})
        pixels.append(small)
    write_csv(out_dir / "rejected.csv", rejected, ["relative_path", "reason"])
    if rejected:
        raise ValueError(f"{len(rejected)} rejected images: inspect rejected.csv")
    original_counts = Counter(r["label"] for r in records)
    if any(original_counts[cls] == 0 for cls in CLASSES):
        raise ValueError("One or more grape labels absent; verify dataset folders")

    sha_groups = defaultdict(list)
    for i, record in enumerate(records):
        sha_groups[record["sha256"]].append(i)
    cross_sha = {sha: idx for sha, idx in sha_groups.items()
                 if len({records[i]["label"] for i in idx}) > 1}
    exact_rows = []
    for sha, idx in cross_sha.items():
        labels = ";".join(sorted({records[i]["label"] for i in idx}))
        for i in idx:
            exact_rows.append({"relative_path": records[i]["relative_path"],
                               "label": records[i]["label"], "sha256": sha,
                               "conflicting_labels": labels})
    write_csv(out_dir / "exact_label_conflicts.csv", exact_rows,
              ["relative_path", "label", "sha256", "conflicting_labels"])
    if cross_sha and not quarantine_exact_conflicts:
        raise ValueError(f"{len(cross_sha)} SHA groups have cross-label label conflicts "
                         f"({len(exact_rows)} source files). Review exact_label_conflicts.csv; "
                         "use --quarantine-exact-conflicts to exclude those groups from "
                         "the manifest without deleting source images.")

    quarantine_hashes = set(cross_sha)
    quarantined = [{"relative_path": r["relative_path"], "label": r["label"],
                    "reason": "cross_label_exact_sha256"}
                   for r in records if r["sha256"] in quarantine_hashes]
    write_csv(out_dir / "quarantined.csv", quarantined,
              ["relative_path", "label", "reason"])
    kept_idx = [i for i, r in enumerate(records) if r["sha256"] not in quarantine_hashes]
    kept = [records[i] for i in kept_idx]
    features = [pixels[i] for i in kept_idx]
    counts = Counter(r["label"] for r in kept)
    if any(counts[cls] == 0 for cls in CLASSES):
        raise ValueError("Some classes have no images after quarantine")

    dsu, seen_sha, seen_source = DSU(len(kept)), {}, {}
    for i, row in enumerate(kept):
        sha, source = row["sha256"], row["source_group"]
        if sha in seen_sha:
            dsu.union(i, seen_sha[sha])
        seen_sha.setdefault(sha, i)
        if source:
            if source in seen_source:
                dsu.union(i, seen_source[source])
            seen_source.setdefault(source, i)

    dindex, pindex = BKTree(), BKTree()
    weak_candidates, strong_conflicts = [], []
    for i, row in enumerate(kept):
        dh, ph = int(row["dhash"], 16), int(row["phash"], 16)
        dh_hits = set(dindex.query(dh, radius))
        ph_hits = set(pindex.query(ph, phash_distance))
        for j in sorted(dh_hits | ph_hits):
            previous = kept[j]
            pdist = (ph ^ int(previous["phash"], 16)).bit_count()
            ddist = (dh ^ int(previous["dhash"], 16)).bit_count()
            rgb_diff = float(np.abs(features[i].astype(np.float32) -
                                    features[j].astype(np.float32)).mean() / 255.0)
            is_strong = pdist <= phash_distance and rgb_diff <= rgb_mae_max
            if is_strong:
                if row["label"] == previous["label"]:
                    dsu.union(i, j)
                else:
                    strong_conflicts.append({
                        "image_a": previous["relative_path"], "image_b": row["relative_path"],
                        "label_a": previous["label"], "label_b": row["label"],
                        "dhash_distance": ddist, "phash_distance": pdist,
                        "rgb_mae": round(rgb_diff, 5)})
            elif j in dh_hits and row["label"] != previous["label"]:
                weak_candidates.append({
                    "image_a": previous["relative_path"], "image_b": row["relative_path"],
                    "dhash_distance": ddist, "phash_distance": pdist,
                    "rgb_mae": round(rgb_diff, 5)})
        dindex.add(dh, i)
        pindex.add(ph, i)

    write_csv(out_dir / "near_label_conflicts.csv", strong_conflicts,
              ["image_a", "image_b", "label_a", "label_b",
               "dhash_distance", "phash_distance", "rgb_mae"])
    write_csv(out_dir / "weak_dhash_candidates.csv", weak_candidates,
              ["image_a", "image_b", "dhash_distance", "phash_distance", "rgb_mae"])
    if strong_conflicts:
        raise ValueError(f"{len(strong_conflicts)} high-similarity cross-label pairs need review; "
                         "inspect near_label_conflicts.csv")

    groups = [f"g{dsu.find(i):06d}" for i in range(len(kept))]
    labels = [row["label"] for row in kept]
    remaining, test = best_group_split(labels, groups, 0.15, 42)
    train_local, val_local = best_group_split(
        [labels[i] for i in remaining], [groups[i] for i in remaining],
        0.15 / 0.85, 84)
    assignment = {int(i): "test" for i in test}
    assignment.update({int(remaining[i]): "train" for i in train_local})
    assignment.update({int(remaining[i]): "val" for i in val_local})
    for i, row in enumerate(kept):
        row.update({"group_id": groups[i], "split": assignment[i]})
    split_groups = defaultdict(set)
    for row in kept:
        split_groups[row["group_id"]].add(row["split"])
    assert all(len(splits) == 1 for splits in split_groups.values())
    write_csv(out_dir / "manifest.csv", kept,
              ["relative_path", "label", "split", "group_id", "sha256", "dhash",
               "phash", "width", "height", "source_group"])
    audit = {
        "source": "GVLiD Mendeley Data v5", "doi": "10.17632/wkymf8bhcg.5",
        "license": "CC BY 4.0; attribution required",
        "total_valid_source_images": len(records), "images": len(kept),
        "original_class_counts": dict(original_counts), "class_counts": dict(counts),
        "splits": dict(Counter(r["split"] for r in kept)),
        "groups": len(split_groups),
        "quarantined_cross_label_sha_files": len(quarantined),
        "cross_label_sha_groups": len(cross_sha),
        "strong_cross_label_pairs": len(strong_conflicts),
        "weak_dhash_candidates_for_review": len(weak_candidates),
        "verified_source_group_rows": sum(bool(r["source_group"]) for r in kept),
        "warning": ("Internal grouped evaluation only; weak similarity candidates require "
                    "visual sampling and farm-level validation remains outstanding.")
    }
    (out_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--groups-csv", type=Path)
    parser.add_argument("--near-distance", type=int, default=3)
    parser.add_argument("--quarantine-exact-conflicts", action="store_true")
    parser.add_argument("--near-phash-distance", type=int, default=5)
    parser.add_argument("--near-rgb-mae", type=float, default=0.06)
    args = parser.parse_args()
    if not 0 <= args.near_distance <= 8:
        parser.error("--near-distance must be from 0 to 8")
    if not 0 <= args.near_phash_distance <= 12 or not 0 <= args.near_rgb_mae <= 0.1:
        parser.error("Invalid near similarity thresholds")
    prepare(args.raw_root, args.out_dir, args.near_distance, args.groups_csv,
            args.quarantine_exact_conflicts, args.near_phash_distance,
            args.near_rgb_mae)


if __name__ == "__main__":
    main()
