import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torchvision import models

from src.data.dataset_6class import create_dataloaders


MODEL_DIR = Path("models/checkpoints")
RESULTS_DIR = Path("reports/results")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train EfficientNet-B0 on six-class plant disease dataset."
    )

    parser.add_argument(
        "--dataset-root",
        type=str,
        default="data_split_grouped_6class",
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default="efficientnet_b0",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(num_classes):
    weights = models.EfficientNet_B0_Weights.DEFAULT

    model = models.efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        in_features,
        num_classes,
    )

    return model


def evaluate_model(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)

            predictions = torch.argmax(outputs, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    loss = total_loss / len(loader.dataset)

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    return (
        loss,
        accuracy,
        precision,
        recall,
        f1,
        all_labels,
        all_predictions,
    )


def main():
    args = parse_args()

    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    use_amp = device.type == "cuda"

    print("=" * 70)
    print("EFFICIENTNET-B0 SIX-CLASS TRAINING")
    print("=" * 70)

    print(f"Device: {device}")
    print(f"AMP enabled: {use_amp}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Early stopping patience: {args.patience}")
    print(f"Seed: {args.seed}")
    print(f"Dataset root: {args.dataset_root}")
    print(f"Experiment: {args.experiment_name}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

        total_vram = (
            torch.cuda.get_device_properties(0).total_memory
            / (1024 ** 3)
        )

        print(f"VRAM: {total_vram:.2f} GB")

    (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
        class_weights,
    ) = create_dataloaders(args.dataset_root)

    class_names = train_dataset.classes
    num_classes = len(class_names)

    print(f"\nNumber of classes: {num_classes}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Batch size: {train_loader.batch_size}")

    model = build_model(num_classes)
    model = model.to(device)

    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    model_path = (
        MODEL_DIR /
        f"{args.experiment_name}_best.pth"
    )

    history_path = (
        RESULTS_DIR /
        f"{args.experiment_name}_history.json"
    )

    test_results_path = (
        RESULTS_DIR /
        f"{args.experiment_name}_test_results.json"
    )

    confusion_matrix_path = (
        RESULTS_DIR /
        f"{args.experiment_name}_confusion_matrix.json"
    )

    best_val_f1 = -1.0
    best_epoch = 0
    no_improvement = 0

    history = []

    for epoch in range(1, args.epochs + 1):

        print("\n" + "=" * 70)
        print(f"Epoch {epoch}/{args.epochs}")
        print("=" * 70)

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += (
                loss.item() * labels.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1,
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        train_loss = running_loss / total
        train_accuracy = correct / total

        (
            val_loss,
            val_accuracy,
            val_precision,
            val_recall,
            val_f1,
            _,
            _,
        ) = evaluate_model(
            model,
            val_loader,
            criterion,
            device,
        )

        scheduler.step(val_f1)

        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Train Loss:       {train_loss:.4f}")
        print(f"Train Accuracy:   {train_accuracy:.4f}")
        print(f"Val Loss:         {val_loss:.4f}")
        print(f"Val Accuracy:     {val_accuracy:.4f}")
        print(f"Val Precision:    {val_precision:.4f}")
        print(f"Val Recall:       {val_recall:.4f}")
        print(f"Val Macro F1:     {val_f1:.4f}")
        print(f"Learning Rate:    {current_lr:.7f}")

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_macro_f1": val_f1,
            "learning_rate": current_lr,
        }

        history.append(epoch_record)

        if val_f1 > best_val_f1:

            best_val_f1 = val_f1
            best_epoch = epoch
            no_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "best_epoch": best_epoch,
                    "best_val_f1": best_val_f1,
                    "model_name": "efficientnet_b0",
                },
                model_path,
            )

            print(
                f"Best model saved "
                f"(Macro F1: {best_val_f1:.4f})"
            )

        else:
            no_improvement += 1

            print(
                f"No improvement: "
                f"{no_improvement}/{args.patience}"
            )

            if no_improvement >= args.patience:
                print("\nEarly stopping triggered.")
                break

    history_path.write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("LOADING BEST CHECKPOINT")
    print("=" * 70)

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)

    (
        test_loss,
        test_accuracy,
        test_precision,
        test_recall,
        test_macro_f1,
        y_true,
        y_pred,
    ) = evaluate_model(
        model,
        test_loader,
        criterion,
        device,
    )

    (
        weighted_precision,
        weighted_recall,
        weighted_f1,
        _,
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    print(
        f"Test Accuracy:          {test_accuracy:.4f}"
    )

    print(
        f"Test Macro Precision:   {test_precision:.4f}"
    )

    print(
        f"Test Macro Recall:      {test_recall:.4f}"
    )

    print(
        f"Test Macro F1:          {test_macro_f1:.4f}"
    )

    print(
        f"Test Weighted F1:       {weighted_f1:.4f}"
    )

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    print("\nPER-CLASS RESULTS")
    print("-" * 70)

    for class_name in class_names:
        metrics = report[class_name]

        print(
            f"{class_name:<35} "
            f"Precision: {metrics['precision']:.4f} | "
            f"Recall: {metrics['recall']:.4f} | "
            f"F1: {metrics['f1-score']:.4f} | "
            f"Support: {int(metrics['support'])}"
        )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
    )

    test_results = {
        "model": "efficientnet_b0",
        "dataset_root": args.dataset_root,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_val_f1,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_macro_precision": test_precision,
        "test_macro_recall": test_recall,
        "test_macro_f1": test_macro_f1,
        "test_weighted_precision": weighted_precision,
        "test_weighted_recall": weighted_recall,
        "test_weighted_f1": weighted_f1,
        "classification_report": report,
    }

    test_results_path.write_text(
        json.dumps(
            test_results,
            indent=2,
        ),
        encoding="utf-8",
    )

    confusion_matrix_path.write_text(
        json.dumps(
            {
                "class_names": class_names,
                "matrix": cm.tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(f"Best epoch:             {best_epoch}")
    print(
        f"Best validation F1:     "
        f"{best_val_f1:.4f}"
    )

    print(
        f"Final test accuracy:    "
        f"{test_accuracy:.4f}"
    )

    print(
        f"Final test macro F1:    "
        f"{test_macro_f1:.4f}"
    )

    print(f"\nModel:                  {model_path}")
    print(f"History:                {history_path}")
    print(
        f"Test results:           "
        f"{test_results_path}"
    )
    print(
        f"Confusion matrix:       "
        f"{confusion_matrix_path}"
    )


if __name__ == "__main__":
    main()