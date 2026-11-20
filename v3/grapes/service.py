"""Lazy, local Grapes inference service for the unified FastAPI dashboard.

This experimental model is trained only on four confirmed grape-leaf classes.
It does not use the V2 Potato/Tomato calibration, OOD guard or Grad-CAM model.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from src.utils.config import PROJECT_ROOT

EXPECTED_CLASSES = (
    "Grapes_Black_rot",
    "Grapes_Esca",
    "Grapes_healthy",
    "Grapes_Leaf_blight",
)


def grape_checkpoint_path() -> Path:
    """Local default; GRAPES_MODEL_PATH can override it at API startup."""
    return Path(os.environ.get(
        "GRAPES_MODEL_PATH",
        str(PROJECT_ROOT / "models" / "grapes" / "efficientnet_b0_grapes_best.pth"),
    ))


class GrapesPredictionService:
    def __init__(self, checkpoint_path=None, knowledge_base_path=None):
        self.checkpoint_path = Path(checkpoint_path or grape_checkpoint_path())
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Grapes checkpoint missing: {self.checkpoint_path}. "
                "Train it first or set GRAPES_MODEL_PATH."
            )
        kb_path = Path(knowledge_base_path or PROJECT_ROOT / "knowledge_base" / "grapes.json")
        with kb_path.open(encoding="utf-8") as stream:
            self.knowledge_base = json.load(stream)

        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        self.classes = tuple(checkpoint["classes"])
        if self.classes != EXPECTED_CLASSES:
            raise ValueError(
                "Grapes checkpoint class order differs from the four-class V3 training schema"
            )
        missing = set(self.classes) - set(self.knowledge_base)
        if missing:
            raise ValueError(f"Grapes knowledge base is missing classes: {sorted(missing)}")

        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, len(self.classes)
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=checkpoint["mean"], std=checkpoint["std"]),
        ])

    def predict(self, image: Image.Image, top_k=4) -> dict:
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL image")
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            scores = torch.softmax(self.model(tensor), dim=1)[0]
        values, indices = scores.topk(min(top_k, len(self.classes)))
        rankings = [
            {"class": self.classes[int(index)], "confidence": float(score)}
            for score, index in zip(values.cpu().tolist(), indices.cpu().tolist())
        ]
        winner = rankings[0]
        runner_up = rankings[1]["confidence"] if len(rankings) > 1 else 0.0
        return {
            "model": "EfficientNet-B0 (Grapes V3 experimental)",
            "device": str(self.device),
            "experimental": True,
            "calibration": "none",
            "temperature": None,
            "ood": None,
            "gradcam_available": False,
            "crop_selection": "Grapes (user-selected, not automatically verified)",
            "warning": (
                "Provisional four-class Grapes model. Scores are uncalibrated; "
                "crop identification, unknown-input rejection, vineyard-level "
                "validation and agronomist review are pending."
            ),
            "prediction": {
                "class": winner["class"],
                "confidence": winner["confidence"],
                "second_best_confidence": runner_up,
                "margin": winner["confidence"] - runner_up,
                "top_k": rankings,
            },
            "disease": self.knowledge_base[winner["class"]],
        }


@lru_cache(maxsize=1)
def get_grape_service() -> GrapesPredictionService:
    """Load weights at first Grapes request, then reuse them across requests."""
    return GrapesPredictionService()
