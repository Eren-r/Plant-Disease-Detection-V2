from pathlib import Path

from src.inference.disease_service import (
    DiseasePredictionService,
)


IMAGE_PATH = Path(
    "data_split_grouped_6class/test/"
    "Tomato_Late_blight/"
    "766771e1-8e8e-4ce4-9af6-b4ed91c9e80a___RS_Late.B 6881.JPG"
)


def main():

    print("=" * 70)
    print("UNIFIED DISEASE PREDICTION SERVICE TEST")
    print("=" * 70)

    service = DiseasePredictionService()

    result = service.predict_file(
        IMAGE_PATH,
        top_k=3,
    )

    print(
        f"\nModel: "
        f"{result['model']}"
    )

    print(
        f"Device: "
        f"{result['device']}"
    )

    print(
        f"Temperature: "
        f"{result['temperature']:.6f}"
    )

    print("\nPREDICTION")
    print("-" * 70)

    print(
        f"Class: "
        f"{result['prediction']['class']}"
    )

    print(
        f"Confidence: "
        f"{result['prediction']['confidence']:.4f}"
    )

    print(
        f"Second-best: "
        f"{result['prediction']['second_best_confidence']:.4f}"
    )

    print(
        f"Margin: "
        f"{result['prediction']['margin']:.4f}"
    )

    print("\nTOP-3")
    print("-" * 70)

    for index, item in enumerate(
        result["prediction"]["top_k"],
        start=1,
    ):
        print(
            f"{index}. "
            f"{item['class']:<35} "
            f"{item['confidence']:.4f}"
        )

    disease = result["disease"]

    print("\nDISEASE INFORMATION")
    print("-" * 70)

    print(
        f"Crop:     {disease['crop']}"
    )

    print(
        f"Disease:  {disease['disease']}"
    )

    print(
        f"Status:   {disease['status']}"
    )

    print(
        f"Severity: {disease['severity']}"
    )

    print("\nSymptoms:")

    for symptom in disease["symptoms"]:
        print(f"  - {symptom}")

    print("\nGeneral management:")

    for action in disease["general_management"]:
        print(f"  - {action}")

    print("\nAdvisory:")

    print(
        f"  {disease['advisory']}"
    )


if __name__ == "__main__":
    main()