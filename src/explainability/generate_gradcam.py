from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.explainability.gradcam import (
    EfficientNetGradCAM,
)


IMAGE_PATH = Path(
    "data_split_grouped_6class/test/"
    "Tomato_Late_blight/"
    "766771e1-8e8e-4ce4-9af6-b4ed91c9e80a___RS_Late.B 6881.JPG"
)

OUTPUT_PATH = Path(
    "reports/figures/"
    "efficientnet_b0_gradcam_error.png"
)


def create_overlay(
    image,
    heatmap,
    alpha=0.45,
):
    image_array = np.asarray(
        image.convert("RGB")
    )

    height, width = image_array.shape[:2]

    heatmap_image = Image.fromarray(
        np.uint8(
            heatmap * 255
        )
    )

    heatmap_image = heatmap_image.resize(
        (width, height),
        Image.Resampling.BILINEAR,
    )

    heatmap_array = np.asarray(
        heatmap_image
    ) / 255.0

    cmap = plt.get_cmap("jet")

    colored_heatmap = cmap(
        heatmap_array
    )[:, :, :3]

    overlay = (
        image_array / 255.0
        * (1.0 - alpha)
        +
        colored_heatmap
        * alpha
    )

    overlay = np.clip(
        overlay,
        0.0,
        1.0,
    )

    return overlay


def main():

    print("=" * 70)
    print("EFFICIENTNET-B0 GRAD-CAM")
    print("=" * 70)

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Image not found: {IMAGE_PATH}"
        )

    image = Image.open(
        IMAGE_PATH
    ).convert("RGB")

    gradcam = EfficientNetGradCAM()

    try:

        heatmap, info = gradcam.generate(
            image
        )

    finally:

        gradcam.close()

    overlay = create_overlay(
        image,
        heatmap,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig = plt.figure(
        figsize=(14, 6)
    )

    ax1 = fig.add_subplot(
        1,
        2,
        1,
    )

    ax1.imshow(image)
    ax1.set_title("Original Image")
    ax1.axis("off")

    ax2 = fig.add_subplot(
        1,
        2,
        2,
    )

    ax2.imshow(overlay)
    ax2.set_title(
        "Grad-CAM — "
        f"{info['predicted_class']}"
    )
    ax2.axis("off")

    fig.suptitle(
        "EfficientNet-B0 Explainability",
        fontsize=14,
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_PATH,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"\nPredicted class: "
        f"{info['predicted_class']}"
    )

    print(
        f"Prediction confidence: "
        f"{info['predicted_confidence']:.4f}"
    )

    print(
        f"Target class: "
        f"{info['target_class']}"
    )

    print(
        f"Target score: "
        f"{info['target_score']:.4f}"
    )

    print(
        f"\nSaved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()