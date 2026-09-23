from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.inference.ood_guard import OODGuard


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ARTIFACT = (
    PROJECT_ROOT
    / "reports"
    / "results"
    / "efficientnet_b0_ood.json"
)

TEST_ROOT = (
    PROJECT_ROOT
    / "data_split_grouped_6class"
    / "test"
)


def make_synthetic_ood_images():
    images = []

    # Solid image
    solid = Image.new(
        "RGB",
        (256, 256),
        (128, 128, 128),
    )
    images.append(
        ("solid_image", solid)
    )

    # Random-noise image
    noise = np.random.default_rng(42).integers(
        0,
        256,
        size=(256, 256, 3),
        dtype=np.uint8,
    )

    noise_image = Image.fromarray(
        noise,
        "RGB",
    )

    images.append(
        ("random_noise", noise_image)
    )

    # Simple geometric shapes
    shapes = Image.new(
        "RGB",
        (256, 256),
        "white",
    )

    draw = ImageDraw.Draw(shapes)

    draw.rectangle(
        (25, 25, 110, 110),
        fill="red",
    )

    draw.ellipse(
        (130, 30, 230, 130),
        fill="blue",
    )

    draw.rectangle(
        (70, 150, 210, 220),
        fill="black",
    )

    images.append(
        ("geometric_shapes", shapes)
    )

    return images


def main():
    guard = OODGuard(
        ARTIFACT
    )

    print(
        f"Threshold: {guard.threshold:.4f}"
    )

    print(
        "\nKnown held-out images:"
    )

    for folder in sorted(
        TEST_ROOT.iterdir()
    ):
        if not folder.is_dir():
            continue

        image_files = [
            *folder.glob("*.JPG"),
            *folder.glob("*.jpg"),
            *folder.glob("*.JPEG"),
            *folder.glob("*.jpeg"),
            *folder.glob("*.PNG"),
            *folder.glob("*.png"),
        ]

        if not image_files:
            continue

        image = Image.open(
            image_files[0]
        ).convert("RGB")

        result = guard.analyze(
            image
        )

        print(
            f"{folder.name}: "
            f"status={result['status']} | "
            f"similarity={result['similarity']:.4f} | "
            f"nearest={result['nearest_class']}"
        )

    print(
        "\nSynthetic OOD images:"
    )

    for name, image in (
        make_synthetic_ood_images()
    ):
        result = guard.analyze(
            image
        )

        print(
            f"{name}: "
            f"status={result['status']} | "
            f"similarity={result['similarity']:.4f} | "
            f"nearest={result['nearest_class']}"
        )


if __name__ == "__main__":
    main()