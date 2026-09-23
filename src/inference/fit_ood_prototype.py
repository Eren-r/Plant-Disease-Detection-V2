from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from src.utils.config import (
    MODEL_CHECKPOINT,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_ROOT = (
    PROJECT_ROOT
    / "data_split_grouped_6class"
    / "train"
)

VAL_ROOT = (
    PROJECT_ROOT
    / "data_split_grouped_6class"
    / "val"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "results"
    / "efficientnet_b0_ood.json"
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

BATCH_SIZE = 32


def load_model():
    weights = EfficientNet_B0_Weights.DEFAULT

    model = efficientnet_b0(
        weights=None
    )

    # Replace the ImageNet 1000-class head
    # with the project's 6-class head.
    model.classifier[1] = torch.nn.Linear(
        model.classifier[1].in_features,
        6,
    )

    checkpoint = torch.load(
        MODEL_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint[
            "model_state_dict"
        ]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint[
            "state_dict"
        ]
    else:
        state_dict = checkpoint

    cleaned = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned[key] = value

    model.load_state_dict(
        cleaned,
        strict=True,
    )

    model.to(DEVICE)
    model.eval()

    return model, weights.transforms()


@torch.inference_mode()
def extract_embeddings(
    model,
    transform,
    dataset,
):
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    embeddings = []
    labels = []

    for images, targets in loader:
        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        features = model.features(
            images
        )

        pooled = F.adaptive_avg_pool2d(
            features,
            output_size=1,
        )

        batch_embeddings = (
            torch.flatten(
                pooled,
                start_dim=1,
            )
        )

        batch_embeddings = F.normalize(
            batch_embeddings,
            dim=1,
        )

        embeddings.append(
            batch_embeddings.cpu()
        )

        labels.append(
            targets.cpu()
        )

    return (
        torch.cat(embeddings),
        torch.cat(labels),
    )


def main():
    if not TRAIN_ROOT.exists():
        raise FileNotFoundError(
            f"Training split not found: {TRAIN_ROOT}"
        )

    if not VAL_ROOT.exists():
        raise FileNotFoundError(
            f"Validation split not found: {VAL_ROOT}"
        )

    model, transform = load_model()

    train_dataset = ImageFolder(
        TRAIN_ROOT,
        transform=transform,
    )

    val_dataset = ImageFolder(
        VAL_ROOT,
        transform=transform,
    )

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Classes: {train_dataset.classes}"
    )

    train_embeddings, train_labels = (
        extract_embeddings(
            model,
            transform,
            train_dataset,
        )
    )

    val_embeddings, val_labels = (
        extract_embeddings(
            model,
            transform,
            val_dataset,
        )
    )

    class_count = len(
        train_dataset.classes
    )

    centroids = []

    for class_index in range(
        class_count
    ):
        class_embeddings = (
            train_embeddings[
                train_labels
                == class_index
            ]
        )

        centroid = class_embeddings.mean(
            dim=0
        )

        centroid = F.normalize(
            centroid.unsqueeze(0),
            dim=1,
        )[0]

        centroids.append(
            centroid
        )

    centroids = torch.stack(
        centroids
    )

    validation_similarity = (
        val_embeddings
        @ centroids.T
    )

    best_validation_similarity = (
        validation_similarity.max(
            dim=1
        ).values.numpy()
    )

    threshold = float(
        np.percentile(
            best_validation_similarity,
            1.0,
        )
    )

    artifact = {
        "method": (
            "EfficientNet-B0 "
            "nearest class centroid "
            "cosine similarity"
        ),
        "threshold_policy": (
            "1st percentile of "
            "validation nearest-centroid similarity"
        ),
        "similarity_threshold": threshold,
        "classes": train_dataset.classes,
        "embedding_dimension": int(
            train_embeddings.shape[1]
        ),
        "training_samples": int(
            len(train_dataset)
        ),
        "validation_samples": int(
            len(val_dataset)
        ),
        "validation_min_similarity": float(
            best_validation_similarity.min()
        ),
        "validation_mean_similarity": float(
            best_validation_similarity.mean()
        ),
        "validation_max_similarity": float(
            best_validation_similarity.max()
        ),
        "centroids": centroids.tolist(),
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            artifact,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nOOD prototype created:"
    )

    print(
        OUTPUT_PATH
    )

    print(
        f"Threshold: {threshold:.4f}"
    )

    print(
        "Validation similarity:"
    )

    print(
        f"  min  = {best_validation_similarity.min():.4f}"
    )

    print(
        f"  mean = {best_validation_similarity.mean():.4f}"
    )

    print(
        f"  max  = {best_validation_similarity.max():.4f}"
    )


if __name__ == "__main__":
    main()