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
    print("PREDICT → EXPLAIN SEQUENCE TEST")
    print("=" * 70)

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Image not found: {IMAGE_PATH}"
        )

    image_bytes = IMAGE_PATH.read_bytes()

    files = {
        "file": (
            IMAGE_PATH.name,
            image_bytes,
            "image/jpeg",
        )
    }

    # ---------------------------------------------------------
    # 1. Predict
    # ---------------------------------------------------------

    print("\n1. /predict")

    predict_response = client.post(
        "/predict",
        files=files,
    )

    print(
        "Status:",
        predict_response.status_code,
    )

    if predict_response.status_code != 200:
        print(predict_response.text)
        raise RuntimeError(
            "Prediction request failed."
        )

    prediction = (
        predict_response.json()
    )

    print(
        "Prediction:",
        prediction["result"]["prediction"]["class"],
    )

    print(
        "Confidence:",
        prediction["result"]["prediction"]["confidence"],
    )

    # ---------------------------------------------------------
    # 2. Explain
    # ---------------------------------------------------------

    print("\n2. /explain")

    # Recreate the exact multipart request as Streamlit does.
    explain_files = {
        "file": (
            IMAGE_PATH.name,
            image_bytes,
            "image/jpeg",
        )
    }

    explain_response = client.post(
        "/explain",
        files=explain_files,
    )

    print(
        "Status:",
        explain_response.status_code,
    )

    print(
        "Content-Type:",
        explain_response.headers.get(
            "content-type"
        ),
    )

    if explain_response.status_code != 200:

        print(
            "ERROR:",
            explain_response.text,
        )

        raise RuntimeError(
            "Grad-CAM request failed."
        )

    print(
        "Response bytes:",
        len(explain_response.content),
    )

    print(
        "Predicted class:",
        explain_response.headers.get(
            "x-predicted-class"
        ),
    )

    print(
        "Confidence:",
        explain_response.headers.get(
            "x-confidence"
        ),
    )

    print("\nSEQUENCE TEST PASSED")


if __name__ == "__main__":
    main()