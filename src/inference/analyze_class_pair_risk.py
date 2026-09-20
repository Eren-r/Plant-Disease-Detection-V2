import json
from collections import Counter
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
    "reports/results/efficientnet_b0_class_pair_risk.json"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def build_model(num_classes):
    model = models.efficientnet_b0(
        weights=None
    )

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        num_classes,
    )

    return model


def main():

    print("=" * 70)
    print("EFFICIENTNET-B0 ACTUAL → PREDICTED PAIR ANALYSIS")
    print("=" * 70)

    calibration = json.loads(
        CALIBRATION_FILE.read_text(
            encoding="utf-8"
        )
    )

    temperature = calibration["temperature"]

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

    class_names = val_dataset.classes

    model = build_model(
        len(class_names)
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

    # Actual class -> predicted class
    pair_total = Counter()
    pair_wrong = Counter()

    # Total examples per actual class
    actual_total = Counter()

    errors = []

    sample_index = 0

    with torch.inference_mode():

        for images, labels in val_loader:

            images = images.to(
                DEVICE,
                non_blocking=True,
            )

            outputs = model(images)

            probabilities = torch.softmax(
                outputs / temperature,
                dim=1,
            )

            top_probabilities, top_indices = torch.topk(
                probabilities,
                k=2,
                dim=1,
            )

            for i in range(len(labels)):

                actual_index = labels[i].item()
                predicted_index = top_indices[i, 0].item()

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

                actual_class = class_names[
                    actual_index
                ]

                predicted_class = class_names[
                    predicted_index
                ]

                pair = (
                    actual_class,
                    predicted_class,
                )

                pair_total[pair] += 1
                actual_total[actual_class] += 1

                correct = (
                    actual_index == predicted_index
                )

                if not correct:

                    pair_wrong[pair] += 1

                    errors.append(
                        {
                            "actual": actual_class,
                            "predicted": predicted_class,
                            "confidence": confidence,
                            "second_confidence": second_confidence,
                            "margin": margin,
                            "image_path": val_dataset.samples[
                                sample_index
                            ][0],
                        }
                    )

                sample_index += 1

    pair_results = []

    for pair, total in pair_total.items():

        wrong = pair_wrong[pair]

        pair_results.append(
            {
                "actual_class": pair[0],
                "predicted_class": pair[1],
                "occurrences": total,
                "errors": wrong,
                "pair_error_rate": (
                    wrong / total
                    if total > 0
                    else 0.0
                ),
                "is_error_pair": wrong > 0,
            }
        )

    # Put actual mistakes first, then highest error rate.
    pair_results.sort(
        key=lambda item: (
            item["errors"],
            item["pair_error_rate"],
            item["occurrences"],
        ),
        reverse=True,
    )

    print("\nACTUAL → PREDICTED PAIRS WITH ERRORS")
    print("-" * 70)

    error_pairs = [
        item
        for item in pair_results
        if item["is_error_pair"]
    ]

    if not error_pairs:
        print("No validation errors found.")

    else:

        for result in error_pairs:

            print(
                f"{result['actual_class']:<30} "
                f"→ {result['predicted_class']:<30} "
                f"Occurrences: {result['occurrences']:3d} | "
                f"Errors: {result['errors']:2d} | "
                f"Error rate: "
                f"{result['pair_error_rate']:.4f}"
            )

    # ---------------------------------------------------------
    # Class-level error rates
    # ---------------------------------------------------------

    class_error_rates = []

    for class_name in class_names:

        total = actual_total[class_name]

        errors_for_class = sum(
            item["errors"]
            for item in pair_results
            if item["actual_class"] == class_name
        )

        class_error_rates.append(
            {
                "class": class_name,
                "samples": total,
                "errors": errors_for_class,
                "error_rate": (
                    errors_for_class / total
                    if total > 0
                    else 0.0
                ),
            }
        )

    print("\nACTUAL-CLASS ERROR RATES")
    print("-" * 70)

    for result in class_error_rates:

        print(
            f"{result['class']:<35} "
            f"Samples: {result['samples']:3d} | "
            f"Errors: {result['errors']:2d} | "
            f"Error rate: {result['error_rate']:.4f}"
        )

    # ---------------------------------------------------------
    # Explicit Tomato Early/Late comparison
    # ---------------------------------------------------------

    late_to_early = next(
        (
            item
            for item in pair_results
            if item["actual_class"]
            == "Tomato_Late_blight"
            and item["predicted_class"]
            == "Tomato_Early_blight"
        ),
        None,
    )

    early_to_late = next(
        (
            item
            for item in pair_results
            if item["actual_class"]
            == "Tomato_Early_blight"
            and item["predicted_class"]
            == "Tomato_Late_blight"
        ),
        None,
    )

    print("\nTOMATO EARLY/LATE BIDIRECTIONAL ANALYSIS")
    print("-" * 70)

    print("Late Blight → Early Blight:")

    if late_to_early:
        print(
            f"  Occurrences: "
            f"{late_to_early['occurrences']}"
        )
        print(
            f"  Errors:      "
            f"{late_to_early['errors']}"
        )
        print(
            f"  Error rate:  "
            f"{late_to_early['pair_error_rate']:.4f}"
        )
    else:
        print("  No occurrences.")

    print("\nEarly Blight → Late Blight:")

    if early_to_late:
        print(
            f"  Occurrences: "
            f"{early_to_late['occurrences']}"
        )
        print(
            f"  Errors:      "
            f"{early_to_late['errors']}"
        )
        print(
            f"  Error rate:  "
            f"{early_to_late['pair_error_rate']:.4f}"
        )
    else:
        print("  No occurrences.")

    print("\nACTUAL VALIDATION ERRORS")
    print("-" * 70)

    for index, error in enumerate(
        errors,
        start=1,
    ):

        print(
            f"{index}. "
            f"{error['actual']} → "
            f"{error['predicted']} | "
            f"Confidence: {error['confidence']:.4f} | "
            f"Margin: {error['margin']:.4f}"
        )

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
                "pair_results": pair_results,
                "class_error_rates": class_error_rates,
                "validation_errors": errors,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("PAIR ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"Saved: {OUTPUT}"
    )


if __name__ == "__main__":
    main()