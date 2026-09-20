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

OUTPUT = Path(
    "reports/results/efficientnet_b0_calibration.json"
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


def collect_validation_logits(
    model,
    loader,
    device,
):
    model.eval()

    all_logits = []
    all_labels = []

    with torch.inference_mode():

        for images, labels in loader:

            images = images.to(
                device,
                non_blocking=True,
            )

            logits = model(images)

            all_logits.append(
                logits.cpu()
            )

            all_labels.append(
                labels.cpu()
            )

    return (
        torch.cat(all_logits),
        torch.cat(all_labels),
    )


def expected_calibration_error(
    probabilities,
    labels,
    number_of_bins=15,
):
    confidences, predictions = probabilities.max(
        dim=1
    )

    accuracies = (
        predictions == labels
    ).float()

    ece = torch.zeros(
        1,
        dtype=torch.float32,
    )

    bin_boundaries = torch.linspace(
        0.0,
        1.0,
        number_of_bins + 1,
    )

    for lower, upper in zip(
        bin_boundaries[:-1],
        bin_boundaries[1:],
    ):
        in_bin = (
            (confidences > lower)
            & (confidences <= upper)
        )

        count = in_bin.sum()

        if count.item() == 0:
            continue

        bin_accuracy = (
            accuracies[in_bin].mean()
        )

        bin_confidence = (
            confidences[in_bin].mean()
        )

        ece += (
            count.float()
            / len(labels)
        ) * torch.abs(
            bin_confidence
            - bin_accuracy
        )

    return float(ece.item())


def calculate_metrics(
    logits,
    labels,
    temperature=1.0,
):
    calibrated_logits = (
        logits / temperature
    )

    probabilities = torch.softmax(
        calibrated_logits,
        dim=1,
    )

    predictions = probabilities.argmax(
        dim=1
    )

    accuracy = (
        predictions == labels
    ).float().mean().item()

    nll = nn.functional.cross_entropy(
        calibrated_logits,
        labels,
    ).item()

    ece = expected_calibration_error(
        probabilities,
        labels,
    )

    confidence = probabilities.max(
        dim=1
    ).values.mean().item()

    return {
        "accuracy": accuracy,
        "nll": nll,
        "ece": ece,
        "mean_confidence": confidence,
    }


def fit_temperature(
    logits,
    labels,
):
    log_temperature = torch.nn.Parameter(
        torch.zeros(1)
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.01,
        max_iter=100,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()

        temperature = torch.exp(
            log_temperature
        )

        loss = criterion(
            logits / temperature,
            labels,
        )

        loss.backward()

        return loss

    optimizer.step(closure)

    temperature = torch.exp(
        log_temperature
    ).item()

    return temperature


def main():

    print("=" * 70)
    print("EFFICIENTNET-B0 CONFIDENCE CALIBRATION")
    print("=" * 70)

    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Validation dataset: {DATASET_ROOT}")

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

    print(
        f"Validation samples: "
        f"{len(val_dataset)}"
    )

    logits, labels = (
        collect_validation_logits(
            model,
            val_loader,
            DEVICE,
        )
    )

    print("\nBEFORE CALIBRATION")
    print("-" * 70)

    before = calculate_metrics(
        logits,
        labels,
        temperature=1.0,
    )

    print(
        f"Accuracy:         "
        f"{before['accuracy']:.4f}"
    )

    print(
        f"NLL:              "
        f"{before['nll']:.4f}"
    )

    print(
        f"ECE:              "
        f"{before['ece']:.4f}"
    )

    print(
        f"Mean confidence:  "
        f"{before['mean_confidence']:.4f}"
    )

    print(
        "\nFitting temperature..."
    )

    temperature = fit_temperature(
        logits,
        labels,
    )

    print(
        f"Learned temperature: "
        f"{temperature:.6f}"
    )

    print("\nAFTER CALIBRATION")
    print("-" * 70)

    after = calculate_metrics(
        logits,
        labels,
        temperature=temperature,
    )

    print(
        f"Accuracy:         "
        f"{after['accuracy']:.4f}"
    )

    print(
        f"NLL:              "
        f"{after['nll']:.4f}"
    )

    print(
        f"ECE:              "
        f"{after['ece']:.4f}"
    )

    print(
        f"Mean confidence:  "
        f"{after['mean_confidence']:.4f}"
    )

    results = {
        "model": "EfficientNet-B0",
        "dataset": DATASET_ROOT,
        "checkpoint": str(CHECKPOINT),
        "temperature": temperature,
        "before_calibration": before,
        "after_calibration": after,
        "class_names": class_names,
        "validation_samples": len(
            val_dataset
        ),
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(
            results,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("CALIBRATION COMPLETE")
    print("=" * 70)

    print(
        f"Saved: {OUTPUT}"
    )


if __name__ == "__main__":
    main()