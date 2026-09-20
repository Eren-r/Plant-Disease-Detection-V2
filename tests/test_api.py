from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.api import app


client = TestClient(app)


@pytest.fixture(scope="module")
def test_image():
    """Create a small synthetic leaf-like image for API contract tests."""
    image = Image.new("RGB", (224, 224), "white")
    draw = ImageDraw.Draw(image)

    # Simple green leaf-like shape.
    draw.ellipse((45, 30, 180, 195), fill=(70, 150, 70))
    draw.line((112, 45, 112, 185), fill=(35, 100, 35), width=5)
    draw.line((112, 110, 75, 80), fill=(35, 100, 35), width=3)
    draw.line((112, 130, 150, 95), fill=(35, 100, 35), width=3)

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["model"] == "EfficientNet-B0"
    assert data["device"] in {
        "cuda",
        "cpu",
    }
    assert data["temperature"] > 0
    assert data["gradcam"] is True


def test_model_info():
    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert data["model"] == "EfficientNet-B0"
    assert len(data["classes"]) == 6
    assert data["calibration"] == "temperature_scaling"
    assert data["explainability"] == "Grad-CAM"

    assert set(data["classes"]) == set(
        data["knowledge_base_classes"]
    )


def test_predict_valid_image(test_image):
    filename = "test_leaf.jpg"

    response = client.post(
        "/predict",
        files={
            "file": (
                filename,
                test_image,
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["filename"] == filename
    assert "result" in data

    result = data["result"]

    assert result["model"] == "EfficientNet-B0"

    prediction = result["prediction"]

    assert prediction["class"] in {
        "Potato___Early_blight",
        "Potato___Late_blight",
        "Potato___healthy",
        "Tomato_Early_blight",
        "Tomato_Late_blight",
        "Tomato_healthy",
    }

    assert 0.0 <= prediction["confidence"] <= 1.0

    assert (
        0.0
        <= prediction["second_best_confidence"]
        <= 1.0
    )

    assert "top_k" in prediction
    assert len(prediction["top_k"]) == 3

    disease = result["disease"]

    assert disease["crop"]
    assert disease["disease"]
    assert disease["status"]
    assert disease["severity"]
    assert disease["symptoms"]
    assert disease["general_management"]
    assert disease["advisory"]


def test_predict_rejects_unsupported_type():
    response = client.post(
        "/predict",
        files={
            "file": (
                "test.txt",
                b"not an image",
                "text/plain",
            )
        },
    )

    assert response.status_code == 415

    data = response.json()

    assert "Unsupported image type" in data["detail"]


def test_predict_rejects_invalid_image():
    response = client.post(
        "/predict",
        files={
            "file": (
                "fake.jpg",
                b"this is not a real image",
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 400

    data = response.json()

    assert "not a valid image" in data["detail"]


def test_gradcam_valid_image(test_image):
    filename = "test_leaf.jpg"

    response = client.post(
        "/explain",
        files={
            "file": (
                filename,
                test_image,
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 200

    assert (
        response.headers["content-type"]
        == "image/png"
    )

    assert len(response.content) > 1000

    assert response.headers.get(
        "x-predicted-class"
    )

    confidence = float(
        response.headers["x-confidence"]
    )

    assert 0.0 <= confidence <= 1.0