import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

from src.data.dataset_6class import create_dataloaders


DATASET_ROOT = "data_split_grouped_6class"

CHECKPOINT = Path(
    "models/checkpoints/efficientnet_b0_best.pth"
)

CALIBRATION_FILE = Path(
    "reports/results/efficientnet_b0_calibration.json"
)

OUTPUT = Path(
    "reports/results/efficientnet_b0_review_thresholds.json"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CONFIDENCE_THRESHOLDS = [
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.92,
    0.95,
    0.97,
    0.98,
    0.99,
]

MARGIN_THRESHOLDS = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
]


def build_model(num_classes):
    model = models.efficientnet_b0(
        weights=None
    )

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        num_classes,
    )

    return model


def collect_validation_predictions(
    model,
    loader,
    device,
    temperature,
    dataset,
):
    model.eval()

    records = []
    sample_index = 0

    with torch.inference_mode():

        for images, labels in loader:

            images = images.to(
                device,
                non_blocking=True,
            )

            outputs = model(images)

            calibrated_logits = (
                outputs / temperature
            )

            probabilities = torch.softmax(
                calibrated_logits,
                dim=1,
            )

            top_probabilities, top_indices = torch.topk(
                probabilities,
                k=2,
                dim=1,
            )

            for i in range(len(labels)):

                actual = labels[i].item()
                predicted = top_indices[i, 0].item()

                confidence = (
                    top_probabilities[i, 0].item()
                )

                second_confidence = (
                    top_probabilities[i, 1].item()
                )

                margin = (
                    confidence
                    - second_confidence
                )

                image_path = dataset.samples[
                    sample_index
                ][0]

                records.append(
                    {
                        "image_path": image_path,
                        "actual_class": dataset.classes[
                            actual
                        ],
                        "predicted_class": dataset.classes[
                            predicted
                        ],
                        "correct": actual == predicted,
                        "confidence": confidence,
                        "second_confidence": second_confidence,
                        "margin": margin,
                    }
                )

                sample_index += 1

    return records


def evaluate_confidence_rule(
    records,
    confidence_threshold,
):
    """
    Flag a prediction when confidence is below
    the supplied threshold.
    """

    total = len(records)

    errors = sum(
        not record["correct"]
        for record in records
    )

    reviewed = 0
    reviewed_errors = 0

    for record in records:

        needs_review = (
            record["confidence"]
            < confidence_threshold
        )

        if needs_review:

            reviewed += 1

            if not record["correct"]:
                reviewed_errors += 1

    missed_errors = (
        errors - reviewed_errors
    )

    return {
        "confidence_threshold": confidence_threshold,
        "reviewed_samples": reviewed,
        "review_rate": reviewed / total,
        "reviewed_errors": reviewed_errors,
        "missed_errors": missed_errors,
        "error_capture_rate": (
            reviewed_errors / errors
            if errors > 0
            else 0.0
        ),
    }


def evaluate_margin_rule(
    records,
    margin_threshold,
):
    """
    Flag a prediction when the gap between the
    top-1 and top-2 probabilities is small.
    """

    total = len(records)

    errors = sum(
        not record["correct"]
        for record in records
    )

    reviewed = 0
    reviewed_errors = 0

    for record in records:

        needs_review = (
            record["margin"]
            < margin_threshold
        )

        if needs_review:

            reviewed += 1

            if not record["correct"]:
                reviewed_errors += 1

    missed_errors = (
        errors - reviewed_errors
    )

    return {
        "margin_threshold": margin_threshold,
        "reviewed_samples": reviewed,
        "review_rate": reviewed / total,
        "reviewed_errors": reviewed_errors,
        "missed_errors": missed_errors,
        "error_capture_rate": (
            reviewed_errors / errors
            if errors > 0
            else 0.0
        ),
    }


def evaluate_combined_rule(
    records,
    confidence_threshold,
    margin_threshold,
):
    """
    Flag a prediction when either:
        confidence is low
    OR
        top-1/top-2 margin is small.
    """

    total = len(records)

    errors = sum(
        not record["correct"]
        for record in records
    )

    reviewed = 0
    reviewed_errors = 0

    for record in records:

        needs_review = (
            record["confidence"]
            < confidence_threshold
            or
            record["margin"]
            < margin_threshold
        )

        if needs_review:

            reviewed += 1

            if not record["correct"]:
                reviewed_errors += 1

    missed_errors = (
        errors - reviewed_errors
    )

    return {
        "confidence_threshold": confidence_threshold,
        "margin_threshold": margin_threshold,
        "reviewed_samples": reviewed,
        "review_rate": reviewed / total,
        "reviewed_errors": reviewed_errors,
        "missed_errors": missed_errors,
        "error_capture_rate": (
            reviewed_errors / errors
            if errors > 0
            else 0.0
        ),
    }


def main():

    print("=" * 70)
    print("VALIDATION REVIEW-THRESHOLD ANALYSIS")
    print("=" * 70)

    calibration = json.loads(
        CALIBRATION_FILE.read_text(
            encoding="utf-8"
        )
    )

    temperature = calibration["temperature"]

    print(
        f"Using temperature: "
        f"{temperature:.6f}"
    )

    (
        _train_dataset,
        val_dataset,
        _test_dataset,
        _train_loader,
        val_loader,
        _test_loader,
        _class_weights,
    ) = create_dataloaders(
        DATASET_ROOT
    )

    model = build_model(
        len(val_dataset.classes)
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    records = collect_validation_predictions(
        model,
        val_loader,
        DEVICE,
        temperature,
        val_dataset,
    )

    errors = [
        record
        for record in records
        if not record["correct"]
    ]

    print(
        f"Validation samples: "
        f"{len(records)}"
    )

    print(
        f"Validation errors: "
        f"{len(errors)}"
    )

    print("\nACTUAL VALIDATION ERRORS")
    print("-" * 70)

    for index, error in enumerate(
        errors,
        start=1,
    ):
        print(
            f"{index}. "
            f"Confidence={error['confidence']:.4f} | "
            f"Margin={error['margin']:.4f} | "
            f"Actual={error['actual_class']} | "
            f"Predicted={error['predicted_class']}"
        )

    # ---------------------------------------------------------
    # Confidence-only rules
    # ---------------------------------------------------------

    confidence_results = []

    print("\nCONFIDENCE-ONLY RULES")
    print("-" * 70)

    for threshold in CONFIDENCE_THRESHOLDS:

        result = evaluate_confidence_rule(
            records,
            threshold,
        )

        confidence_results.append(result)

        print(
            f"Confidence < {threshold:.2f} | "
            f"Review rate: {result['review_rate']:.3f} | "
            f"Error capture: {result['error_capture_rate']:.3f} | "
            f"Missed errors: {result['missed_errors']}"
        )

    # ---------------------------------------------------------
    # Margin-only rules
    # ---------------------------------------------------------

    margin_results = []

    print("\nMARGIN-ONLY RULES")
    print("-" * 70)

    for threshold in MARGIN_THRESHOLDS:

        result = evaluate_margin_rule(
            records,
            threshold,
        )

        margin_results.append(result)

        print(
            f"Margin < {threshold:.2f} | "
            f"Review rate: {result['review_rate']:.3f} | "
            f"Error capture: {result['error_capture_rate']:.3f} | "
            f"Missed errors: {result['missed_errors']}"
        )

    # ---------------------------------------------------------
    # Combined rules
    # ---------------------------------------------------------

    combined_results = []

    for confidence_threshold in (
        CONFIDENCE_THRESHOLDS
    ):

        for margin_threshold in (
            MARGIN_THRESHOLDS
        ):

            result = evaluate_combined_rule(
                records,
                confidence_threshold,
                margin_threshold,
            )

            combined_results.append(result)

    full_capture = [
        result
        for result in combined_results
        if result["error_capture_rate"] >= 1.0
    ]

    full_capture.sort(
        key=lambda result: (
            result["review_rate"],
            result["confidence_threshold"],
            result["margin_threshold"],
        )
    )

    print(
        "\nBEST COMBINED RULES CAPTURING "
        "100% OF VALIDATION ERRORS"
    )
    print("-" * 70)

    if not full_capture:

        print(
            "No tested rule captured all "
            "validation errors."
        )

    else:

        for result in full_capture[:10]:

            print(
                f"Confidence < "
                f"{result['confidence_threshold']:.2f} "
                f"OR Margin < "
                f"{result['margin_threshold']:.2f} | "
                f"Review rate: "
                f"{result['review_rate']:.3f} | "
                f"Captured: "
                f"{result['reviewed_errors']}/"
                f"{len(errors)} | "
                f"Missed: "
                f"{result['missed_errors']}"
            )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(
            {
                "model": "EfficientNet-B0",
                "dataset": DATASET_ROOT,
                "temperature": temperature,
                "validation_samples": len(records),
                "validation_errors": len(errors),
                "validation_errors_detail": errors,
                "confidence_only": confidence_results,
                "margin_only": margin_results,
                "combined_rules": combined_results,
                "best_full_capture_rules": full_capture[:20],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("THRESHOLD ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"Saved: {OUTPUT}"
    )


if __name__ == "__main__":
    main()