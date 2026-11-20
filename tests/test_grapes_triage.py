"""Tests for risk-free grape cross-label triage reports."""
import csv
import json

from v3.grapes.triage_near_conflicts import triage


def test_triage_deduplicates_pairs_and_preserves_source(tmp_path):
    source = tmp_path / "near_label_conflicts.csv"
    fields = ["image_a", "image_b", "label_a", "label_b",
              "dhash_distance", "phash_distance", "rgb_mae"]
    with source.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "image_a": "Esca/a.jpg", "image_b": "Leaf_blight/b.jpg",
            "label_a": "Grapes_Esca", "label_b": "Grapes_Leaf_blight",
            "dhash_distance": 0, "phash_distance": 1, "rgb_mae": .01})
        writer.writerow({
            "image_a": "Leaf_blight/b.jpg", "image_b": "Esca/a.jpg",
            "label_a": "Grapes_Leaf_blight", "label_b": "Grapes_Esca",
            "dhash_distance": 0, "phash_distance": 1, "rgb_mae": .01})
    output = tmp_path / "triage"
    summary = triage(source, output, 10)
    assert summary["reported_pair_rows"] == 2
    assert summary["distinct_pair_rows"] == 1
    assert summary["unique_images_in_pairs"] == 2
    assert source.is_file()
    with (output / "manual_review_queue.csv").open(
        newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 1 and rows[0]["manual_decision"] == ""
    assert (output / "representative_pairs.csv").exists()
    assert json.loads((output / "triage_summary.json").read_text())["status"] == "manual_review_required"
