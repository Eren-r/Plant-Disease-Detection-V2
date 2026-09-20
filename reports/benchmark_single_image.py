import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

from src.data.dataset_6class import create_dataloaders


DATASET_ROOT = "data_split_grouped_6class"

RESNET_CHECKPOINT = Path(
    "models/checkpoints/resnet18_robust_best.pth"
)

EFFICIENTNET_CHECKPOINT = Path(
    "models/checkpoints/efficientnet_b0_best.pth"
)

OUTPUT = Path(
    "reports/results/single_image_benchmark.json"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

WARMUP_RUNS = 20
TIMED_RUNS = 100


def build_resnet18(num_classes):
    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes,
    )

    return model


def build_efficientnet_b0(num_classes):
    model = models.efficientnet_b0(weights=None)

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        num_classes,
    )

    return model


def benchmark(model, image, name):
    model = model.to(DEVICE)
    model.eval()

    image = image.unsqueeze(0).to(
        DEVICE,
        non_blocking=True,
    )

    # Warm-up
    with torch.inference_mode():

        for _ in range(WARMUP_RUNS):
            _ = model(image)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    timings = []

    with torch.inference_mode():

        for _ in range(TIMED_RUNS):

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()

            _ = model(image)

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            elapsed = (
                time.perf_counter() - start
            ) * 1000

            timings.append(elapsed)

    mean_latency = sum(timings) / len(timings)

    sorted_timings = sorted(timings)

    p50 = sorted_timings[
        len(sorted_timings) // 2
    ]

    p95_index = int(
        len(sorted_timings) * 0.95
    ) - 1

    p95 = sorted_timings[
        max(0, p95_index)
    ]

    p99_index = int(
        len(sorted_timings) * 0.99
    ) - 1

    p99 = sorted_timings[
        max(0, p99_index)
    ]

    result = {
        "model": name,
        "device": str(DEVICE),
        "batch_size": 1,
        "mean_latency_ms": mean_latency,
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
        "p99_latency_ms": p99,
    }

    print(f"\n{name}")
    print("-" * 60)
    print(f"Mean latency: {mean_latency:.3f} ms")
    print(f"P50 latency:  {p50:.3f} ms")
    print(f"P95 latency:  {p95:.3f} ms")
    print(f"P99 latency:  {p99:.3f} ms")

    return result


def main():
    print("=" * 70)
    print("SINGLE-IMAGE MODEL LATENCY BENCHMARK")
    print("=" * 70)

    print(f"Device: {DEVICE}")

    if torch.cuda.is_available():
        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    (
        _train_dataset,
        _val_dataset,
        test_dataset,
        _train_loader,
        _val_loader,
        _test_loader,
        _class_weights,
    ) = create_dataloaders(DATASET_ROOT)

    num_classes = len(test_dataset.classes)

    # Use one real test image.
    image, _label = test_dataset[0]

    results = []

    # ---------------------------------------------------------
    # ResNet18 Robust
    # ---------------------------------------------------------

    resnet = build_resnet18(num_classes)

    checkpoint = torch.load(
        RESNET_CHECKPOINT,
        map_location=DEVICE,
        weights_only=True,
    )

    resnet.load_state_dict(
        checkpoint["model_state_dict"]
    )

    results.append(
        benchmark(
            resnet,
            image,
            "ResNet18 Robust",
        )
    )

    del resnet

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # EfficientNet-B0
    # ---------------------------------------------------------

    efficientnet = build_efficientnet_b0(
        num_classes
    )

    checkpoint = torch.load(
        EFFICIENTNET_CHECKPOINT,
        map_location=DEVICE,
        weights_only=True,
    )

    efficientnet.load_state_dict(
        checkpoint["model_state_dict"]
    )

    results.append(
        benchmark(
            efficientnet,
            image,
            "EfficientNet-B0",
        )
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(
            {
                "benchmark_type": "single_image",
                "warmup_runs": WARMUP_RUNS,
                "timed_runs": TIMED_RUNS,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)

    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()