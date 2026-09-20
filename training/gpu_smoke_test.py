import torch
import torch.nn as nn
from torchvision import models

from src.data.dataset_6class import create_dataloaders


def main():

    print("=" * 70)
    print("GPU TRAINING SMOKE TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # GPU check
    # ---------------------------------------------------------

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device("cuda")

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

    print(
        f"VRAM: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )

    # ---------------------------------------------------------
    # Load existing 6-class data pipeline
    # ---------------------------------------------------------

    (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
        class_weights
    ) = create_dataloaders()

    print(
        f"Classes: {len(train_dataset.classes)}"
    )

    print(
        f"Train samples: {len(train_dataset)}"
    )

    print(
        f"Validation samples: {len(val_dataset)}"
    )

    print(
        f"Test samples: {len(test_dataset)}"
    )

    print(
        f"Batch size: {train_loader.batch_size}"
    )

    # ---------------------------------------------------------
    # Create pretrained ResNet18
    # ---------------------------------------------------------

    weights = models.ResNet18_Weights.DEFAULT

    model = models.resnet18(
        weights=weights
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        len(train_dataset.classes)
    )

    model = model.to(device)

    print("\nModel moved to GPU successfully.")

    # ---------------------------------------------------------
    # Weighted loss
    # ---------------------------------------------------------

    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # ---------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )

    # ---------------------------------------------------------
    # Training smoke test
    # ---------------------------------------------------------

    model.train()

    max_batches = 5

    print(
        f"\nRunning {max_batches} training batches..."
    )

    for batch_idx, (images, labels) in enumerate(
        train_loader
    ):

        if batch_idx >= max_batches:
            break

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        predictions = outputs.argmax(
            dim=1
        )

        accuracy = (
            predictions == labels
        ).float().mean().item()

        print(
            f"Batch {batch_idx + 1}/{max_batches} | "
            f"Loss: {loss.item():.4f} | "
            f"Accuracy: {accuracy:.4f}"
        )

    # ---------------------------------------------------------
    # Synchronize GPU
    # ---------------------------------------------------------

    torch.cuda.synchronize()

    allocated = (
        torch.cuda.memory_allocated()
        / 1024**3
    )

    reserved = (
        torch.cuda.memory_reserved()
        / 1024**3
    )

    print()

    print("=" * 70)
    print("GPU SMOKE TEST PASSED")
    print("=" * 70)

    print(
        f"GPU memory allocated: {allocated:.2f} GB"
    )

    print(
        f"GPU memory reserved:  {reserved:.2f} GB"
    )


if __name__ == "__main__":
    main()