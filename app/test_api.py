from pathlib import Path

from fastapi.testclient import TestClient

from app.api import app


IMAGE_PATH = Path(
    "data_split_grouped_6class/test/"
    "Tomato_Late_blight/"
    "766771e1-8e8e-4ce4-9af6-b4ed91c9e80a___RS_Late.B 6881.JPG"
)


client = TestClient(app)


def main():
    print("=" * 70)
    print("FASTAPI BACKEND TEST")
    print("=" * 70)

    print("\n1. HEALTH")
    response = client.get("/health")

    print("Status:", response.status_code)
    print(response.json())

    assert response.status_code == 200

    print("\n2. MODEL INFO")
    response = client.get("/model-info")

    print("Status:", response.status_code)
    print(response.json())

    assert response.status_code == 200

    print("\n3. PREDICTION")

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Test image not found: {IMAGE_PATH}"
        )

    with IMAGE_PATH.open("rb") as image_file:

        response = client.post(
            "/predict",
            files={
                "file": (
                    IMAGE_PATH.name,
                    image_file,
                    "image/jpeg",
                )
            },
        )

    print("Status:", response.status_code)

    result = response.json()

    print(result)

    assert response.status_code == 200
    assert "result" in result

    print("\nAPI TEST PASSED")


if __name__ == "__main__":
    main()