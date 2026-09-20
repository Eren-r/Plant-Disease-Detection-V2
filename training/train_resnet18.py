from pathlib import Path
import argparse
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import models

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from src.data.dataset_6class import create_dataloaders


# ============================================================
# Configuration
# ============================================================

SEED = 42

EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

PATIENCE = 4

MODEL_DIR = Path("models/checkpoints")
RESULTS_DIR = Path("reports/results")
FIGURES_DIR = Path("reports/figures")


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Enable mixed precision on CUDA.
USE_AMP = DEVICE.type == "cuda"

# ============================================================
# Command-line configuration
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Train ResNet18 on the six-class plant disease dataset."
    )

    parser.add_argument(
        "--dataset-root",
        type=str,
        default="data_split_6class",
        help="Path to the six-class dataset split."
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default="resnet18",
        help="Name used for saved model and result files."
    )

    return parser.parse_args()

# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Model
# ============================================================

def create_model(num_classes):

    weights = models.ResNet18_Weights.DEFAULT

    model = models.resnet18(
        weights=weights
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    return model


# ============================================================
# Training
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler
):

    model.train()

    running_loss = 0.0

    all_predictions = []
    all_labels = []

    for images, labels in loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        # ----------------------------------------------------
        # Mixed precision forward pass
        # ----------------------------------------------------

        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=USE_AMP
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        if USE_AMP:

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

        else:

            loss.backward()

            optimizer.step()

        running_loss += (
            loss.item() * images.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
        )

        all_labels.extend(
            labels.detach()
            .cpu()
            .numpy()
        )

    epoch_loss = (
        running_loss /
        len(loader.dataset)
    )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    return (
        epoch_loss,
        accuracy
    )


# ============================================================
# Validation
# ============================================================

def evaluate(
    model,
    loader,
    criterion
):

    model.eval()

    running_loss = 0.0

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=USE_AMP
            ):

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

            running_loss += (
                loss.item() *
                images.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    epoch_loss = (
        running_loss /
        len(loader.dataset)
    )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0
        )
    )

    return (
        epoch_loss,
        accuracy,
        precision,
        recall,
        f1
    )


# ============================================================
# Final test evaluation
# ============================================================

def evaluate_test_set(
    model,
    loader,
    class_names
):

    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=USE_AMP
            ):

                outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.numpy()
            )

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0
        )
    )

    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average="weighted",
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Per-class metrics
    # --------------------------------------------------------

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            labels=list(range(len(class_names))),
            zero_division=0
        )
    )

    per_class = {}

    for index, class_name in enumerate(
        class_names
    ):

        per_class[class_name] = {
            "precision": float(
                precision[index]
            ),
            "recall": float(
                recall[index]
            ),
            "f1": float(
                f1[index]
            ),
            "support": int(
                support[index]
            ),
        }

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(len(class_names)))
    )

    results = {

        "accuracy": float(
            accuracy
        ),

        "macro_precision": float(
            macro_precision
        ),

        "macro_recall": float(
            macro_recall
        ),

        "macro_f1": float(
            macro_f1
        ),

        "weighted_precision": float(
            weighted_precision
        ),

        "weighted_recall": float(
            weighted_recall
        ),

        "weighted_f1": float(
            weighted_f1
        ),

        "per_class": per_class,

        "class_names": class_names,

        "confusion_matrix": cm.tolist(),
    }

    return results


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    set_seed(SEED)

    model_path = MODEL_DIR / f"{args.experiment_name}_best.pth"
    history_path = RESULTS_DIR / f"{args.experiment_name}_history.json"
    test_results_path = RESULTS_DIR / f"{args.experiment_name}_test_results.json"
    confusion_matrix_path = (
        RESULTS_DIR /
        f"{args.experiment_name}_confusion_matrix.json"
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("RESNET18 SIX-CLASS TRAINING")
    print("=" * 70)

    print(f"Device: {DEVICE}")
    print(f"AMP enabled: {USE_AMP}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Weight decay: {WEIGHT_DECAY}")
    print(f"Early stopping patience: {PATIENCE}")
    print(f"Seed: {SEED}")
    print(f"Dataset root: {args.dataset_root}")
    print(f"Experiment: {args.experiment_name}")

    if DEVICE.type == "cuda":

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"VRAM: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    print()

    # ========================================================
    # Dataset
    # ========================================================

    (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
        class_weights
    ) = create_dataloaders(
        args.dataset_root
    )

    num_classes = len(
        train_dataset.classes
    )

    print(
        f"Number of classes: "
        f"{num_classes}"
    )

    print(
        f"Training samples: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(val_dataset)}"
    )

    print(
        f"Test samples: "
        f"{len(test_dataset)}"
    )

    print(
        f"Batch size: "
        f"{train_loader.batch_size}"
    )

    print()

    # ========================================================
    # Model
    # ========================================================

    model = create_model(
        num_classes
    )

    model = model.to(
        DEVICE
    )

    class_weights = class_weights.to(
        DEVICE
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=USE_AMP
    )

    # ========================================================
    # Training state
    # ========================================================

    best_f1 = -float("inf")

    best_epoch = 0

    epochs_without_improvement = 0

    history = []

    # ========================================================
    # Training loop
    # ========================================================

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        print()
        print(
            "=" * 70
        )

        print(
            f"Epoch {epoch}/{EPOCHS}"
        )

        print(
            "=" * 70
        )

        train_loss, train_acc = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scaler
            )
        )

        (
            val_loss,
            val_acc,
            val_precision,
            val_recall,
            val_f1
        ) = evaluate(
            model,
            val_loader,
            criterion
        )

        scheduler.step(
            val_f1
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        print(
            f"Train Loss:       {train_loss:.4f}"
        )

        print(
            f"Train Accuracy:   {train_acc:.4f}"
        )

        print(
            f"Val Loss:         {val_loss:.4f}"
        )

        print(
            f"Val Accuracy:     {val_acc:.4f}"
        )

        print(
            f"Val Precision:    {val_precision:.4f}"
        )

        print(
            f"Val Recall:       {val_recall:.4f}"
        )

        print(
            f"Val Macro F1:     {val_f1:.4f}"
        )

        print(
            f"Learning Rate:    {current_lr:.7f}"
        )

        epoch_result = {

            "epoch": epoch,

            "train_loss": train_loss,

            "train_accuracy": train_acc,

            "val_loss": val_loss,

            "val_accuracy": val_acc,

            "val_precision_macro":
                val_precision,

            "val_recall_macro":
                val_recall,

            "val_f1_macro":
                val_f1,

            "learning_rate":
                current_lr,
        }

        history.append(
            epoch_result
        )

        # ----------------------------------------------------
        # Save best checkpoint
        # ----------------------------------------------------

        if val_f1 > best_f1:

            best_f1 = val_f1

            best_epoch = epoch

            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "class_names":
                        train_dataset.classes,

                    "best_val_f1":
                        best_f1,

                    "epoch":
                        epoch,

                    "seed":
                        SEED,

                    "model_name":
                        "resnet18",

                    "num_classes":
                        num_classes,
                },
                model_path
            )

            print(
                f"Best model saved "
                f"(Macro F1: {best_f1:.4f})"
            )

        else:

            epochs_without_improvement += 1

            print(
                f"No improvement: "
                f"{epochs_without_improvement}/"
                f"{PATIENCE}"
            )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print(
                "\nEarly stopping triggered."
            )

            break

    # ========================================================
    # Save training history
    # ========================================================

    with open(
        history_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            history,
            file,
            indent=2
        )

    # ========================================================
    # Reload best model
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "LOADING BEST CHECKPOINT"
    )

    print(
        "=" * 70
    )

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # ========================================================
    # Final test evaluation
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "FINAL TEST EVALUATION"
    )

    print(
        "=" * 70
    )

    test_results = evaluate_test_set(
        model,
        test_loader,
        train_dataset.classes
    )

    print(
        f"Test Accuracy:          "
        f"{test_results['accuracy']:.4f}"
    )

    print(
        f"Test Macro Precision:   "
        f"{test_results['macro_precision']:.4f}"
    )

    print(
        f"Test Macro Recall:      "
        f"{test_results['macro_recall']:.4f}"
    )

    print(
        f"Test Macro F1:          "
        f"{test_results['macro_f1']:.4f}"
    )

    print(
        f"Test Weighted F1:       "
        f"{test_results['weighted_f1']:.4f}"
    )

    # --------------------------------------------------------
    # Per-class results
    # --------------------------------------------------------

    print()
    print(
        "PER-CLASS RESULTS"
    )

    print(
        "-" * 70
    )

    for class_name, metrics in (
        test_results["per_class"].items()
    ):

        print(
            f"{class_name:<35} "
            f"Precision: {metrics['precision']:.4f} | "
            f"Recall: {metrics['recall']:.4f} | "
            f"F1: {metrics['f1']:.4f} | "
            f"Support: {metrics['support']}"
        )

    # ========================================================
    # Save test results
    # ========================================================

    with open(
        test_results_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            test_results,
            file,
            indent=2
        )

    # Save confusion matrix separately
    with open(
        confusion_matrix_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            {
                "class_names":
                    train_dataset.classes,

                "matrix":
                    test_results[
                        "confusion_matrix"
                    ],
            },
            file,
            indent=2
        )

    # ========================================================
    # Final summary
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "TRAINING COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Best epoch:             {best_epoch}"
    )

    print(
        f"Best validation F1:     {best_f1:.4f}"
    )

    print(
        f"Final test accuracy:    "
        f"{test_results['accuracy']:.4f}"
    )

    print(
        f"Final test macro F1:    "
        f"{test_results['macro_f1']:.4f}"
    )

    print()
    print(
        f"Model:                  {model_path}"
    )

    print(
        f"History:                {history_path}"
    )

    print(
        f"Test results:           {test_results_path}"
    )

    print(
        f"Confusion matrix:       {confusion_matrix_path}"
    )


if __name__ == "__main__":
    main()