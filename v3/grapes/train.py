"""Train a four-class grape EfficientNet-B0 without modifying the V2 checkpoint."""
from __future__ import annotations
import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from v3.grapes.prepare import CLASSES

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class ManifestImages(Dataset):
    def __init__(self, records, raw_root, split):
        self.rows = [r for r in records if r["split"] == split]
        self.raw_root = Path(raw_root)
        self.mapping = {cls: i for i, cls in enumerate(CLASSES)}
        if not self.rows:
            raise ValueError(f"No images in {split} split")
        if split == "train":
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=.15, contrast=.15, saturation=.15),
                transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256), transforms.CenterCrop(224),
                transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.raw_root / row["relative_path"]) as img:
            image = self.transform(img.convert("RGB"))
        return image, self.mapping[row["label"]]


def build_model(pretrained=True):
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
    return model


def evaluate(model, loader, loss_fn, device):
    model.eval()
    actual, predicted, weighted_loss = [], [], 0.0
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            weighted_loss += float(loss_fn(logits, labels)) * len(labels)
            actual.extend(labels.cpu().tolist())
            predicted.extend(logits.argmax(dim=1).cpu().tolist())
    return {"loss": weighted_loss / len(actual),
            "accuracy": float(accuracy_score(actual, predicted)),
            "macro_f1": float(f1_score(actual, predicted,
                                       labels=list(range(len(CLASSES))),
                                       average="macro", zero_division=0)),
            "actual": actual, "predicted": predicted}


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cpu":
        print("WARNING: CUDA unavailable; training will use CPU.")
    with Path(args.manifest).open(newline="", encoding="utf-8") as file:
        records = list(csv.DictReader(file))
    if not records or set(row["label"] for row in records) != set(CLASSES):
        raise ValueError("Expected exactly the four grape classes")
    group_splits = {}
    for row in records:
        if row["split"] not in {"train", "val", "test"}:
            raise ValueError("Unexpected split")
        if not (args.raw_root / row["relative_path"]).is_file():
            raise FileNotFoundError(row["relative_path"])
        g = row["group_id"]
        if g in group_splits and group_splits[g] != row["split"]:
            raise ValueError("Group leakage: " + g)
        group_splits[g] = row["split"]
    datasets = {s: ManifestImages(records, args.raw_root, s) for s in ("train", "val", "test")}
    loaders = {s: DataLoader(ds, batch_size=args.batch_size, shuffle=(s == "train"),
                             num_workers=args.workers, pin_memory=(device.type == "cuda"))
               for s, ds in datasets.items()}
    counts = Counter(row["label"] for row in datasets["train"].rows)
    if any(counts[cls] == 0 for cls in CLASSES):
        raise ValueError("Missing class in training split")
    n = sum(counts.values())
    weights = torch.tensor([n / (len(CLASSES) * counts[cls]) for cls in CLASSES],
                           dtype=torch.float32, device=device)
    model = build_model(pretrained=True).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=.5, patience=2)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    best, best_epoch, no_improvement, history = -1.0, 0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_total, seen = 0.0, 0
        for images, labels in loaders["train"]:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(images)
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_total += float(loss.detach()) * len(labels)
            seen += len(labels)
        validation = evaluate(model, loaders["val"], loss_fn, device)
        scheduler.step(validation["macro_f1"])
        record = {"epoch": epoch, "train_loss": loss_total / seen,
                  "val_loss": validation["loss"], "val_accuracy": validation["accuracy"],
                  "val_macro_f1": validation["macro_f1"]}
        history.append(record)
        print(record)
        if validation["macro_f1"] > best + 1e-5:
            best, best_epoch, no_improvement = validation["macro_f1"], epoch, 0
            torch.save({"model_state_dict": model.state_dict(), "classes": list(CLASSES),
                        "model_name": "efficientnet_b0_grapes", "best_epoch": epoch,
                        "best_validation_macro_f1": best, "image_size": 224,
                        "mean": MEAN, "std": STD, "dataset": "GVLiD v5",
                        "seed": args.seed}, args.checkpoint)
        else:
            no_improvement += 1
            if no_improvement >= args.patience:
                print("Early stopping")
                break
    # Test set stays unused for training/selection; evaluate best checkpoint once.
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test = evaluate(model, loaders["test"], loss_fn, device)
    results = {
        "dataset": "GVLiD v5", "model": "efficientnet_b0_grapes",
        "classes": list(CLASSES), "best_epoch": best_epoch, "validation_macro_f1": best,
        "test_accuracy": test["accuracy"], "test_macro_f1": test["macro_f1"],
        "confusion_matrix": confusion_matrix(test["actual"], test["predicted"],
                                             labels=list(range(len(CLASSES)))).tolist(),
        "classification_report": classification_report(
            test["actual"], test["predicted"], labels=list(range(len(CLASSES))),
            target_names=list(CLASSES), output_dict=True, zero_division=0),
        "split_sizes": {name: len(ds) for name, ds in datasets.items()},
        "limitation": "Grouped internal test only, not an independent field-validation study."}
    (args.out_dir / "grapes_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (args.out_dir / "grapes_test_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Saved:", args.checkpoint, args.out_dir / "grapes_test_results.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("epochs and batch-size must be positive; workers must be nonnegative")
    run(args)


if __name__ == "__main__":
    main()
