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
    "reports/results/model_benchmark.json"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BENCHMARK_IMAGES = 256
WARMUP_BATCHES = 5


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


def count_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


def get_checkpoint_size(path):
    return path.stat().st_size / (1024 ** 2)


def benchmark_model(model, test_loader, name):
    model = model.to(DEVICE)
    model.eval()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    total_images = 0
    total_time = 0.0

    warmup_done = False
    warmup_count = 0

    with torch.no_grad():

        for images, _labels in test_loader:

            images = images.to(
                DEVICE,
                non_blocking=True,
            )

            # Warm-up GPU / kernels.
            if not warmup_done:

                _ = model(images)

                if DEVICE.type == "cuda":
                    torch.cuda.synchronize()

                warmup_count += 1

                if warmup_count >= WARMUP_BATCHES:
                    warmup_done = True

                continue

            if total_images >= BENCHMARK_IMAGES:
                break

            remaining = (
                BENCHMARK_IMAGES -
                total_images
            )

            images = images[:remaining]

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()

            _ = model(images)

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start

            total_time += elapsed
            total_images += images.size(0)

    latency_per_image_ms = (
        total_time / total_images
    ) * 1000

    throughput = (
        total_images / total_time
    )

    peak_memory_gb = None

    if DEVICE.type == "cuda":
        peak_memory_gb = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 3)
        )

    print(f"\n{name}")
    print("-" * 60)
    print(f"Parameters:           {count_parameters(model):,}")
    print(
        f"Latency / image:      "
        f"{latency_per_image_ms:.3f} ms"
    )
    print(
        f"Throughput:           "
        f"{throughput:.2f} images/sec"
    )

    if peak_memory_gb is not None:
        print(
            f"Peak GPU memory:      "
            f"{peak_memory_gb:.3f} GB"
        )

    return {
        "model": name,
        "parameters": count_parameters(model),
        "latency_ms_per_image": latency_per_image_ms,
        "throughput_images_per_sec": throughput,
        "peak_gpu_memory_gb": peak_memory_gb,
    }


def main():

    print("=" * 70)
    print("MODEL PRODUCTION BENCHMARK")
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
        test_loader,
        _class_weights,
    ) = create_dataloaders(DATASET_ROOT)

    num_classes = len(test_dataset.classes)

    print(
        f"Test images available: "
        f"{len(test_dataset)}"
    )

    print(
        f"Benchmark images: "
        f"{BENCHMARK_IMAGES}"
    )

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

    resnet_result = benchmark_model(
        resnet,
        test_loader,
        "ResNet18 Robust",
    )

    resnet_result["checkpoint_size_mb"] = (
        get_checkpoint_size(
            RESNET_CHECKPOINT
        )
    )

    del resnet

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

    efficientnet_result = benchmark_model(
        efficientnet,
        test_loader,
        "EfficientNet-B0",
    )

    efficientnet_result["checkpoint_size_mb"] = (
        get_checkpoint_size(
            EFFICIENTNET_CHECKPOINT
        )
    )

    results = {
        "device": str(DEVICE),
        "benchmark_images": BENCHMARK_IMAGES,
        "models": [
            resnet_result,
            efficientnet_result,
        ],
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
    print("BENCHMARK COMPLETE")
    print("=" * 70)

    print("\nCHECKPOINT SIZE")

    print(
        f"ResNet18 Robust: "
        f"{resnet_result['checkpoint_size_mb']:.2f} MB"
    )

    print(
        f"EfficientNet-B0: "
        f"{efficientnet_result['checkpoint_size_mb']:.2f} MB"
    )

    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()