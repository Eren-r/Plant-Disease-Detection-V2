"""Summarize and prioritize cross-label grape near-duplicate candidates.

Does not modify datasets, labels, checkpoints or the training manifest.

python -m v3.grapes.triage_near_conflicts \
  --conflicts data/grapes/manifests/gvlid_v2/near_label_conflicts.csv \
  --out-dir reports/grapes/near_triage
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def triage(conflicts, out_dir, examples_per_class_pair=15):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with Path(conflicts).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = {"image_a", "image_b", "label_a", "label_b",
                  "phash_distance", "rgb_mae"}
        if not fields.issubset(reader.fieldnames or []):
            raise ValueError("Expected near_label_conflicts.csv with " + ", ".join(sorted(fields)))
        source_rows = list(reader)
    pairs = {}
    for row in source_rows:
        a = row["image_a"].replace("\\", "/")
        b = row["image_b"].replace("\\", "/")
        if a == b:
            continue
        key = tuple(sorted((a, b)))
        if key not in pairs:
            pairs[key] = {
                "image_a": a, "label_a": row["label_a"],
                "image_b": b, "label_b": row["label_b"],
                "dhash_distance": row.get("dhash_distance", ""),
                "phash_distance": row["phash_distance"],
                "rgb_mae": row["rgb_mae"]
            }
    rows = sorted(pairs.values(),
                  key=lambda r: (float(r["rgb_mae"]), int(r["phash_distance"])))
    by_class = defaultdict(list)
    frequency = Counter()
    for row in rows:
        by_class[tuple(sorted((row["label_a"], row["label_b"])))].append(row)
        frequency.update((row["image_a"], row["image_b"]))
    sample = []
    for classes, matches in sorted(by_class.items(), key=lambda x: -len(x[1])):
        used = set()
        for row in matches:
            if sum(1 for r in sample if
                   tuple(sorted((r["label_a"], r["label_b"]))) == classes) >= examples_per_class_pair:
                break
            if row["image_a"] in used and row["image_b"] in used:
                continue
            used.update((row["image_a"], row["image_b"]))
            sample.append(row)
    queue_fields = [
        "image_a", "label_a", "image_b", "label_b",
        "dhash_distance", "phash_distance", "rgb_mae",
        "manual_decision", "reviewer", "notes"
    ]
    write_csv(out_dir / "manual_review_queue.csv",
              [dict(row, manual_decision="", reviewer="", notes="") for row in rows],
              queue_fields)
    write_csv(out_dir / "representative_pairs.csv", sample, queue_fields[:7])
    summary = {
        "reported_pair_rows": len(source_rows),
        "distinct_pair_rows": len(rows),
        "unique_images_in_pairs": len(frequency),
        "pair_counts_by_label": {
            " / ".join(key): len(value) for key, value in by_class.items()
        },
        "top_repeated_images": [
            {"image": name, "pair_count": count}
            for name, count in frequency.most_common(20)
        ],
        "representative_pair_count": len(sample),
        "status": "manual_review_required",
        "warning": "Do not automatically delete images, relabel conditions, or bypass grouping."
    }
    (out_dir / "triage_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print("Review queue:", out_dir / "manual_review_queue.csv")
    print("Representative pairs:", out_dir / "representative_pairs.csv")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conflicts", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--examples-per-class-pair", type=int, default=15)
    args = parser.parse_args()
    if not 1 <= args.examples_per_class_pair <= 100:
        parser.error("--examples-per-class-pair must be between 1 and 100")
    triage(args.conflicts, args.out_dir, args.examples_per_class_pair)


if __name__ == "__main__":
    main()
