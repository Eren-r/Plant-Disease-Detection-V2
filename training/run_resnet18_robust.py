import runpy
import sys

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import src.data.dataset_6class as dataset_module


# Keep the original dataloader implementation untouched.
_original_create_dataloaders = dataset_module.create_dataloaders


def create_robust_dataloaders(dataset_root="data_split_6class"):
    (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
        class_weights,
    ) = _original_create_dataloaders(dataset_root)

    # Keep the existing augmentation pipeline and add
    # mild robustness transformations for training only.
    original_transform = train_dataset.transform

    train_dataset.transform = transforms.Compose(
        [
            original_transform,
            transforms.RandomApply(
                [
                    transforms.GaussianBlur(
                        kernel_size=3,
                        sigma=(0.1, 2.0),
                    )
                ],
                p=0.20,
            ),
            transforms.RandomErasing(
                p=0.20,
                scale=(0.02, 0.10),
                ratio=(0.3, 3.3),
                value=0,
            ),
        ]
    )

    # Rebuild only the training loader because its dataset transform changed.
    robust_train_loader = DataLoader(
        train_dataset,
        batch_size=train_loader.batch_size,
        shuffle=True,
        num_workers=train_loader.num_workers,
        pin_memory=train_loader.pin_memory,
        drop_last=train_loader.drop_last,
        collate_fn=train_loader.collate_fn,
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        robust_train_loader,
        val_loader,
        test_loader,
        class_weights,
    )


# Monkey-patch the trainer's imported dataloader function.
dataset_module.create_dataloaders = create_robust_dataloaders


# Pass the same grouped benchmark and a new experiment name.
sys.argv = [
    "train_resnet18",
    "--dataset-root",
    "data_split_grouped_6class",
    "--experiment-name",
    "resnet18_robust",
]


# Run the existing, already-validated ResNet18 training engine.
runpy.run_module(
    "training.train_resnet18",
    run_name="__main__",
)