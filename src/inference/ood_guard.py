from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import (
    EfficientNet_B0_Weights,
    efficientnet_b0,
)

from src.utils.config import MODEL_CHECKPOINT


class OODGuard:
    """
    Prototype-based novelty detector.

    Compares an image embedding against the six
    known-class EfficientNet-B0 centroids.

    This is an advisory safeguard, not a guarantee
    that an image belongs to the supported classes.
    """

    def __init__(
        self,
        artifact_path: str | Path,
        device: str | None = None,
    ):
        self.artifact_path = Path(artifact_path)

        if not self.artifact_path.exists():
            raise FileNotFoundError(
                f"OOD artifact not found: {self.artifact_path}"
            )

        artifact = json.loads(
            self.artifact_path.read_text(
                encoding="utf-8"
            )
        )

        self.classes = artifact["classes"]
        self.threshold = float(
            artifact["similarity_threshold"]
        )

        self.centroids = torch.tensor(
            artifact["centroids"],
            dtype=torch.float32,
        )

        self.device = torch.device(
            device
            if device is not None
            else (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        weights = EfficientNet_B0_Weights.DEFAULT

        model = efficientnet_b0(
            weights=None
        )

        # The project model has six output classes.
        model.classifier[1] = torch.nn.Linear(
            model.classifier[1].in_features,
            len(self.classes),
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

        cleaned_state_dict = {}

        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[len("module.") :]
            cleaned_state_dict[key] = value

        model.load_state_dict(
            cleaned_state_dict,
            strict=True,
        )

        self.model = model.to(
            self.device
        )
        self.model.eval()

        self.transform = weights.transforms()

        self.centroids = F.normalize(
            self.centroids,
            dim=1,
        ).to(self.device)

    @torch.inference_mode()
    def _embedding(
        self,
        image: Image.Image,
    ) -> torch.Tensor:
        image = image.convert("RGB")

        tensor = self.transform(
            image
        ).unsqueeze(0).to(self.device)

        features = self.model.features(
            tensor
        )

        pooled = F.adaptive_avg_pool2d(
            features,
            output_size=1,
        )

        embedding = torch.flatten(
            pooled,
            start_dim=1,
        )

        return F.normalize(
            embedding,
            dim=1,
        )

    def analyze(
        self,
        image: Image.Image,
    ) -> dict:
        embedding = self._embedding(
            image
        )

        similarities = (
            embedding @ self.centroids.T
        )[0]

        best_index = int(
            torch.argmax(similarities)
        )

        best_similarity = float(
            similarities[best_index]
        )

        is_ood = (
            best_similarity
            < self.threshold
        )

        return {
            "status": (
                "out_of_distribution"
                if is_ood
                else "in_distribution"
            ),
            "is_ood": is_ood,
            "similarity": round(
                best_similarity,
                4,
            ),
            "threshold": round(
                self.threshold,
                4,
            ),
            "nearest_class": self.classes[
                best_index
            ],
        }