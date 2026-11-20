"""Investigate cross-label duplicate candidates without modifying original images.

    python -m v3.grapes.review_conflicts --raw-root data/grapes/raw --conflicts data/grapes/manifests/gvlid_v1/label_conflicts.csv --out-dir reports/grapes/conflict_review
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import io
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from v3.grapes.prepare import dhash, label_for


def safe_path(root: Path, rel: str) -> Path:
    root = root.resolve()
    path = (root / rel).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Path outside raw dataset: " + rel)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def inspect_file(path: Path):
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    with Image.open(path) as image:
        finger = dhash(image)
    return {"sha256": digest, "dhash": finger}


def thumbnail(path: Path):
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((220, 180))
        stream = io.BytesIO()
        image.save(stream, "JPEG", quality=75)
    return "data:image/jpeg;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def review(raw_root: Path, conflicts_csv: Path, out_dir: Path, max_preview: int = 80):
    raw_root = Path(raw_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with Path(conflicts_csv).open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not {"image_a", "image_b"}.issubset(reader.fieldnames or []):
            raise ValueError("Expected image_a and image_b columns in conflict CSV")
        pairs = list(reader)

    cache, detailed = {}, []
    for item in pairs:
        a, b = item["image_a"], item["image_b"]
        first, second = safe_path(raw_root, a), safe_path(raw_root, b)
        if a not in cache:
            cache[a] = inspect_file(first)
        if b not in cache:
            cache[b] = inspect_file(second)
        x, y = cache[a], cache[b]
        kind = ("exact_byte_duplicate" if x["sha256"] == y["sha256"]
                else "perceptual_candidate_requires_review")
        detailed.append({
            "image_a": a, "label_a": label_for(Path(a)) or "unrecognized",
            "image_b": b, "label_b": label_for(Path(b)) or "unrecognized",
            "match_type": kind, "dhash_distance": (x["dhash"] ^ y["dhash"]).bit_count(),
            "same_sha256": x["sha256"] == y["sha256"],
        })

    header = ["image_a", "label_a", "image_b", "label_b", "match_type", "dhash_distance", "same_sha256"]
    with (out_dir / "conflict_review.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        writer.writerows(detailed)

    summary = {
        "total_conflict_pairs": len(detailed),
        "exact_byte_duplicate_pairs_across_labels": sum(r["same_sha256"] for r in detailed),
        "perceptual_near_match_candidates": sum(not r["same_sha256"] for r in detailed),
        "unique_affected_images": len({s for r in detailed for s in (r["image_a"], r["image_b"])}),
        "warning": ("Different SHA256 hashes do not prove images are different, and low dHash "
                    "distance does not prove they are identical. Visually review candidates "
                    "and source labels before making training/split decisions."),
    }
    (out_dir / "conflict_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    cards = []
    prioritized = sorted(detailed, key=lambda r: (not r["same_sha256"], r["dhash_distance"]))
    for idx, r in enumerate(prioritized[:max_preview], 1):
        a, b = r["image_a"], r["image_b"]
        image_a = thumbnail(safe_path(raw_root, a))
        image_b = thumbnail(safe_path(raw_root, b))
        cards.append(
            '<section><h3>Pair ' + str(idx) + " — " + html.escape(r["match_type"]) +
            " (dHash distance " + str(r["dhash_distance"]) + ")</h3>" +
            '<div class="pair"><figure><img src="' + image_a + '"><figcaption>' +
            html.escape(a) + " | " + html.escape(r["label_a"]) + "</figcaption></figure>" +
            '<figure><img src="' + image_b + '"><figcaption>' +
            html.escape(b) + " | " + html.escape(r["label_b"]) +
            "</figcaption></figure></div></section>"
        )
    page = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>GVLiD cross-label review</title>'
            '<style>body{font-family:system-ui;margin:2rem;max-width:1050px}section{border:1px solid #ddd;'
            'border-radius:10px;padding:12px;margin:12px 0}.pair{display:flex;gap:20px;flex-wrap:wrap}'
            'figure{margin:5px;max-width:46%;overflow-wrap:anywhere}img{max-width:220px;max-height:180px}'
            'figcaption{font-size:12px}</style><h1>Grapes cross-label conflict review</h1><p>'
            'Do not automatically remove or relabel these images. Investigate source provenance.'
            '</p>' + "".join(cards) + "</html>")
    (out_dir / "conflict_gallery.html").write_text(page, encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Detailed report:", out_dir / "conflict_review.csv")
    print("Open in browser:", out_dir / "conflict_gallery.html")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--conflicts", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-preview", type=int, default=80)
    args = parser.parse_args()
    if args.max_preview < 0:
        parser.error("--max-preview must be non-negative")
    review(args.raw_root, args.conflicts, args.out_dir, args.max_preview)


if __name__ == "__main__":
    main()
