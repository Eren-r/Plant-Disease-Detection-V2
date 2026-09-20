from src.utils.config import MODEL_CHECKPOINT

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


class EfficientNetGradCAM:
    """
    Grad-CAM implementation for EfficientNet-B0.
    """

    def __init__(
        self,
        checkpoint_path=MODEL_CHECKPOINT,
    ):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )

        self.class_names = checkpoint["class_names"]

        self.model = models.efficientnet_b0(
            weights=None
        )

        self.model.classifier[1] = nn.Linear(
            self.model.classifier[1].in_features,
            len(self.class_names),
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[
                        0.485,
                        0.456,
                        0.406,
                    ],
                    std=[
                        0.229,
                        0.224,
                        0.225,
                    ],
                ),
            ]
        )

        # EfficientNet-B0's last convolutional block.
        self.target_layer = self._find_last_conv_layer()

        self.activations = None
        self.gradients = None

        self.forward_handle = self.target_layer.register_forward_hook(
            self._save_activations
        )

        self.backward_handle = (
            self.target_layer.register_full_backward_hook(
                self._save_gradients
            )
        )

    def _find_last_conv_layer(self):
        """
        Find the final Conv2d layer automatically.
        """
        last_conv = None

        for module in self.model.features.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module

        if last_conv is None:
            raise RuntimeError(
                "Could not find a convolutional layer."
            )

        return last_conv

    def _save_activations(
        self,
        module,
        inputs,
        output,
    ):
        self.activations = output

    def _save_gradients(
        self,
        module,
        grad_input,
        grad_output,
    ):
        self.gradients = grad_output[0]

    def generate(
        self,
        image,
        target_class=None,
    ):
        """
        Generate Grad-CAM heatmap.

        Returns:
            heatmap: numpy array in [0, 1]
            prediction_info: dict
        """

        if not isinstance(image, Image.Image):
            raise TypeError(
                "image must be a PIL.Image.Image"
            )

        image = image.convert("RGB")

        input_tensor = self.transform(
            image
        ).unsqueeze(0)

        input_tensor = input_tensor.to(
            self.device
        )

        self.model.zero_grad(
            set_to_none=True
        )

        output = self.model(
            input_tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1,
        )

        predicted_index = (
            probabilities.argmax(
                dim=1
            ).item()
        )

        predicted_confidence = (
            probabilities[0, predicted_index]
            .item()
        )

        if target_class is None:
            target_index = predicted_index

        elif isinstance(target_class, int):
            target_index = target_class

        elif isinstance(target_class, str):

            if target_class not in self.class_names:
                raise ValueError(
                    f"Unknown class: {target_class}"
                )

            target_index = self.class_names.index(
                target_class
            )

        else:
            raise TypeError(
                "target_class must be None, int, or str"
            )

        target_score = output[
            0,
            target_index,
        ]

        target_score.backward()

        if self.activations is None:
            raise RuntimeError(
                "Activations were not captured."
            )

        if self.gradients is None:
            raise RuntimeError(
                "Gradients were not captured."
            )

        # Global average pooling over gradients.
        weights = self.gradients.mean(
            dim=(2, 3),
            keepdim=True,
        )

        # Weighted feature maps.
        cam = (
            weights
            * self.activations
        ).sum(
            dim=1
        )

        # ReLU: only positive contribution.
        cam = torch.relu(cam)

        cam = cam[0].detach().cpu()

        # Normalize.
        cam -= cam.min()

        max_value = cam.max()

        if max_value > 0:
            cam /= max_value

        heatmap = cam.numpy()

        prediction_info = {
            "predicted_class": self.class_names[
                predicted_index
            ],
            "predicted_confidence": predicted_confidence,
            "target_class": self.class_names[
                target_index
            ],
            "target_score": float(
                target_score.detach().cpu().item()
            ),
        }

        return heatmap, prediction_info

    def close(self):
        """
        Remove registered hooks.
        """
        self.forward_handle.remove()
        self.backward_handle.remove()