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
    print("GRAD-CAM API DIAGNOSTIC TEST")
    print("=" * 70)

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Image not found: {IMAGE_PATH}"
        )

    with IMAGE_PATH.open("rb") as image_file:

        response = client.post(
            "/explain",
            files={
                "file": (
                    IMAGE_PATH.name,
                    image_file,
                    "image/jpeg",
                )
            },
        )

    print("\nStatus code:", response.status_code)
    print("Content type:", response.headers.get("content-type"))

    if response.status_code == 200:
        print(
            "Grad-CAM response received successfully."
        )
        print(
            "Response bytes:",
            len(response.content),
        )

        print(
            "Predicted class:",
            response.headers.get(
                "x-predicted-class"
            ),
        )

        print(
            "Confidence:",
            response.headers.get(
                "x-confidence"
            ),
        )

    else:
        print("\nERROR RESPONSE")
        print("-" * 70)

        try:
            print(response.json())

        except Exception:
            print(response.text)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()