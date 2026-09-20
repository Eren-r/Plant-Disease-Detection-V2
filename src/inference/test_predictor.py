from pathlib import Path

from PIL import Image

from src.inference.predictor import PlantDiseasePredictor


IMAGE_PATH = Path(
    "data_split_grouped_6class/test/"
    "Tomato_Late_blight/"
    "766771e1-8e8e-4ce4-9af6-b4ed91c9e80a___RS_Late.B 6881.JPG"
)


def main():
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Image not found: {IMAGE_PATH}"
        )

    predictor = PlantDiseasePredictor()

    image = Image.open(IMAGE_PATH)

    result = predictor.predict(
        image,
        top_k=3,
    )

    print("=" * 70)
    print("CONFIDENCE-AWARE INFERENCE TEST")
    print("=" * 70)

    print(f"Device: {result['device']}")
    print(f"Model: {result['model']}")

    print(
        f"\nPrediction: "
        f"{result['prediction']}"
    )

    print(
        f"Confidence: "
        f"{result['confidence']:.4f}"
    )

    print(
        f"Second-best confidence: "
        f"{result['second_best_confidence']:.4f}"
    )

    print(
        f"Prediction margin: "
        f"{result['margin']:.4f}"
    )

    print(
        f"Calibration temperature: "
        f"{result['temperature']:.6f}"
    )


    print("\nTOP-3")
    print("-" * 70)

    for index, prediction in enumerate(
        result["top_k"],
        start=1,
    ):
        print(
            f"{index}. "
            f"{prediction['class']:<35} "
            f"{prediction['confidence']:.4f}"
        )


if __name__ == "__main__":
    main()