from src.utils.config import (
    CALIBRATION_FILE,
    MODEL_CHECKPOINT,
    MODEL_NAME,
)

import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


class PlantDiseasePredictor:
    """
    Production inference wrapper for EfficientNet-B0.

    Probabilities are temperature-calibrated using
    a temperature learned from the validation set.

    The predictor intentionally does not make an automatic
    human-review decision because validation showed that
    confidence alone cannot reliably identify every error.
    """

    def __init__(
        self,
        checkpoint_path=MODEL_CHECKPOINT,
        calibration_path=CALIBRATION_FILE,
    ):
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.checkpoint_path = Path(
            checkpoint_path
        )

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: "
                f"{self.checkpoint_path}"
            )

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )

        self.class_names = checkpoint[
            "class_names"
        ]

        self.calibration_path = Path(
            calibration_path
        )

        self.temperature = self._load_temperature()

        if self.temperature <= 0:
            raise ValueError(
                "Calibration temperature must be > 0."
            )

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

    def _load_temperature(self):
        """
        Load temperature learned on the validation set.
        """

        if not self.calibration_path.exists():
            return 1.0

        data = json.loads(
            self.calibration_path.read_text(
                encoding="utf-8"
            )
        )

        return float(
            data.get(
                "temperature",
                1.0,
            )
        )

    def predict(
        self,
        image,
        top_k=3,
    ):
        """
        Predict from a PIL image.

        Returns:
            prediction
            calibrated confidence
            second-best confidence
            prediction margin
            top-k predictions
            calibration temperature
        """

        if not isinstance(
            image,
            Image.Image,
        ):
            raise TypeError(
                "image must be a PIL.Image.Image"
            )

        image = image.convert("RGB")

        tensor = self.transform(
            image
        )

        tensor = tensor.unsqueeze(0)

        tensor = tensor.to(
            self.device,
            non_blocking=True,
        )

        with torch.inference_mode():

            outputs = self.model(
                tensor
            )

            calibrated_logits = (
                outputs / self.temperature
            )

            probabilities = torch.softmax(
                calibrated_logits,
                dim=1,
            )[0]

        top_k = min(
            top_k,
            len(self.class_names),
        )

        top_probabilities, top_indices = (
            torch.topk(
                probabilities,
                k=top_k,
            )
        )

        predictions = []

        for probability, index in zip(
            top_probabilities,
            top_indices,
        ):
            predictions.append(
                {
                    "class": self.class_names[
                        index.item()
                    ],
                    "confidence": float(
                        probability.item()
                    ),
                }
            )

        top_confidence = predictions[
            0
        ]["confidence"]

        if len(predictions) > 1:

            second_confidence = predictions[
                1
            ]["confidence"]

        else:
            second_confidence = 0.0

        margin = (
            top_confidence
            - second_confidence
        )

        return {
            "prediction": predictions[
                0
            ]["class"],

            "confidence": top_confidence,

            "second_best_confidence": (
                second_confidence
            ),

            "margin": margin,

            "top_k": predictions,

            "temperature": self.temperature,

            "model": MODEL_NAME,

            "device": str(
                self.device
            ),

            "review_policy": (
                "not_configured"
            ),
        }

    def predict_file(
        self,
        image_path,
        top_k=3,
    ):
        """
        Predict directly from an image file.
        """

        image_path = Path(
            image_path
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: "
                f"{image_path}"
            )

        with Image.open(
            image_path
        ) as image:

            image = image.convert(
                "RGB"
            )

            return self.predict(
                image,
                top_k=top_k,
            )