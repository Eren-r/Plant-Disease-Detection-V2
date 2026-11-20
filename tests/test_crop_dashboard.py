"""Regressions for the unified crop-aware API.

Grapes uses a stub because its trained checkpoint is intentionally kept local.
The existing tests/test_api.py continues to cover the unchanged V2 endpoints.
"""
from io import BytesIO
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.api as api_module


client = TestClient(api_module.app)


@pytest.fixture
def sample_image():
    picture = Image.new("RGB", (224, 224), (60, 140, 70))
    buf = BytesIO()
    picture.save(buf, format="JPEG")
    return buf.getvalue()


def upload(image_bytes, crop):
    return client.post(
        "/predict-crop",
        data={"crop": crop},
        files={"file": ("leaf.jpg", image_bytes, "image/jpeg")},
    )


def test_supported_crops():
    r = client.get("/supported-crops")
    assert r.status_code == 200
    assert r.json()["crops"] == ["Potato", "Tomato", "Grapes"]
    assert r.json()["grapes_experimental"] is True


def test_grapes_routes_to_v3_service_without_v2_ood(monkeypatch, sample_image):
    called = []

    class FakeGrapeService:
        def predict(self, image, top_k):
            called.append((image.size, top_k))
            return {
                "model": "EfficientNet-B0 (Grapes V3 experimental)",
                "experimental": True, "calibration": "none",
                "temperature": None, "ood": None, "gradcam_available": False,
                "prediction": {
                    "class": "Grapes_Black_rot", "confidence": 0.87,
                    "top_k": [{"class": "Grapes_Black_rot", "confidence": 0.87}]
                },
                "disease": {"crop": "Grapes", "disease": "Black rot"},
            }

    monkeypatch.setattr(api_module, "get_grape_service", lambda: FakeGrapeService())

    def should_not_run(_):
        raise AssertionError("V2 OOD guard must not analyze Grapes")

    monkeypatch.setattr(api_module.ood_guard, "analyze", should_not_run)
    response = upload(sample_image, "Grapes")
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["disease"]["crop"] == "Grapes"
    assert result["experimental"] is True
    assert result["ood"] is None
    assert result["image_quality"]["width"] == 224
    assert called == [((224, 224), 4)]


def test_potato_routes_to_existing_v2_model(monkeypatch, sample_image):
    def fake_predict(image, top_k):
        assert top_k == 3
        return {
            "model": "EfficientNet-B0", "temperature": 1.0,
            "prediction": {
                "class": "Potato___healthy", "confidence": 0.8,
                "top_k": [{"class": "Potato___healthy", "confidence": 0.8}]
            },
            "disease": {"crop": "Potato", "disease": "Healthy"},
        }

    monkeypatch.setattr(api_module.service, "predict", fake_predict)
    monkeypatch.setattr(api_module.ood_guard, "analyze", lambda image: {"status": "in_distribution"})
    response = upload(sample_image, "Potato")
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["disease"]["crop"] == "Potato"
    assert result["ood"]["status"] == "in_distribution"
    assert "experimental" not in result


def test_v2_crop_mismatch_rejected(monkeypatch, sample_image):
    monkeypatch.setattr(
        api_module.service, "predict",
        lambda image, top_k: {
            "model": "EfficientNet-B0", "temperature": 1.0,
            "prediction": {"class": "Tomato_healthy", "confidence": .99, "top_k": []},
            "disease": {"crop": "Tomato", "disease": "Healthy"}
        }
    )
    monkeypatch.setattr(api_module.ood_guard, "analyze", lambda image: {})
    response = upload(sample_image, "Potato")
    assert response.status_code == 422
    assert "selected Potato" in response.json()["detail"]


def test_missing_grape_checkpoint_returns_503(monkeypatch, sample_image):
    def missing():
        raise FileNotFoundError("Grapes checkpoint missing; train first")
    monkeypatch.setattr(api_module, "get_grape_service", missing)
    response = upload(sample_image, "Grapes")
    assert response.status_code == 503
    assert "checkpoint" in response.json()["detail"]


def test_unknown_crop_rejected(sample_image):
    response = upload(sample_image, "Mango")
    assert response.status_code == 422


def test_invalid_image_rejected_for_grapes():
    response = upload(b"this-is-not-an-image", "Grapes")
    assert response.status_code == 400


def test_old_v2_route_still_exists():
    routes = {route.path for route in api_module.app.routes}
    assert "/predict" in routes
    assert "/explain" in routes
    assert "/predict-crop" in routes
