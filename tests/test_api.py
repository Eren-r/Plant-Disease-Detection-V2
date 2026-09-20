from pathlib import Path

from fastapi.testclient import TestClient

from app.api import app


client = TestClient(app)


TEST_IMAGE = Path(
    "data_split_grouped_6class/test/"
    "Tomato_Late_blight/"
    "766771e1-8e8e-4ce4-9af6-b4ed91c9e80a___RS_Late.B 6881.JPG"
)


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


def test_predict_valid_image():
    assert TEST_IMAGE.exists()

    with TEST_IMAGE.open("rb") as image_file:
        response = client.post(
            "/predict",
            files={
                "file": (
                    TEST_IMAGE.name,
                    image_file,
                    "image/jpeg",
                )
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert data["filename"] == TEST_IMAGE.name
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

    assert (
        "not a valid image"
        in data["detail"]
    )


def test_gradcam_valid_image():
    assert TEST_IMAGE.exists()

    with TEST_IMAGE.open("rb") as image_file:
        response = client.post(
            "/explain",
            files={
                "file": (
                    TEST_IMAGE.name,
                    image_file,
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