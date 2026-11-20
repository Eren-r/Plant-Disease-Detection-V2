"""Unit tests for the experimental Grapes service without real dataset or GPU.

A tiny compatible model is substituted for EfficientNet to validate checkpoint
loading, class ordering, response shape, and in-memory prediction.
"""
import json

import pytest
import torch
import torch.nn as nn
from PIL import Image

from v3.grapes import service as grapes_service


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(nn.Identity(), nn.Linear(3, 4))

    def forward(self, x):
        return self.classifier(x.mean(dim=(2, 3)))


def make_checkpoint(path, classes=None):
    net = TinyNet()
    torch.save({
        "classes": list(classes or grapes_service.EXPECTED_CLASSES),
        "model_state_dict": net.state_dict(),
        "model_name": "efficientnet_b0_grapes",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }, path)


def make_knowledge_base(path):
    details = {
        name: {"crop": "Grapes", "disease": name.split("_", 1)[-1],
               "advisory": "Experimental only"}
        for name in grapes_service.EXPECTED_CLASSES
    }
    path.write_text(json.dumps(details), encoding="utf-8")


def test_grape_service_loads_once_and_predicts(monkeypatch, tmp_path):
    monkeypatch.setattr(grapes_service.models, "efficientnet_b0",
                        lambda weights=None: TinyNet())
    monkeypatch.setattr(grapes_service.torch.cuda, "is_available", lambda: False)
    ckpt, kb = tmp_path / "grapes.pth", tmp_path / "grapes.json"
    make_checkpoint(ckpt)
    make_knowledge_base(kb)

    svc = grapes_service.GrapesPredictionService(
        checkpoint_path=ckpt, knowledge_base_path=kb)
    result = svc.predict(Image.new("RGB", (224, 224), "green"))
    assert result["disease"]["crop"] == "Grapes"
    assert result["experimental"] is True
    assert result["calibration"] == "none"
    assert result["ood"] is None
    assert result["gradcam_available"] is False
    assert len(result["prediction"]["top_k"]) == 4
    assert sum(item["confidence"] for item in result["prediction"]["top_k"]) == pytest.approx(1.0)


def test_grape_service_rejects_mismatched_checkpoint_classes(monkeypatch, tmp_path):
    monkeypatch.setattr(grapes_service.models, "efficientnet_b0",
                        lambda weights=None: TinyNet())
    ckpt, kb = tmp_path / "grapes.pth", tmp_path / "grapes.json"
    make_checkpoint(ckpt, classes=list(reversed(grapes_service.EXPECTED_CLASSES)))
    make_knowledge_base(kb)
    with pytest.raises(ValueError, match="class order"):
        grapes_service.GrapesPredictionService(
            checkpoint_path=ckpt, knowledge_base_path=kb)


def test_missing_model_is_clear_error(tmp_path):
    kb = tmp_path / "grapes.json"
    make_knowledge_base(kb)
    with pytest.raises(FileNotFoundError, match="checkpoint missing"):
        grapes_service.GrapesPredictionService(
            checkpoint_path=tmp_path / "not_here.pth", knowledge_base_path=kb)
