from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO

import numpy as np
from PIL import Image, ImageFilter, ImageStat


@dataclass
class ImageQualityResult:
    status: str
    score: float
    width: int
    height: int
    blur_score: float
    brightness: float
    contrast: float
    warnings: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


class ImageQualityAnalyzer:
    """
    Lightweight image-quality checks for uploaded plant images.

    The analyzer is intentionally advisory:
    - 'good'      -> no major quality concern
    - 'warning'   -> one or more quality concerns
    - 'poor'      -> multiple/significant quality concerns

    It does not attempt to determine whether an image contains a leaf.
    """

    MIN_WIDTH = 128
    MIN_HEIGHT = 128

    BLUR_WARNING_THRESHOLD = 40.0
    BRIGHTNESS_LOW = 35.0
    BRIGHTNESS_HIGH = 220.0
    CONTRAST_WARNING_THRESHOLD = 18.0

    def analyze(
        self,
        image_bytes: bytes | Image.Image,
    ) -> ImageQualityResult:

        if isinstance(image_bytes, Image.Image):
            image = image_bytes.convert("RGB")
        else:
            image = Image.open(
                BytesIO(image_bytes)
            ).convert("RGB")

        width, height = image.size

        gray = image.convert("L")

        # Laplacian-style blur proxy using PIL's FIND_EDGES filter.
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edge_array = np.asarray(edges, dtype=np.float32)

        blur_score = float(edge_array.var())

        stat = ImageStat.Stat(gray)
        brightness = float(stat.mean[0])
        contrast = float(stat.stddev[0])

        warnings: list[str] = []

        if width < self.MIN_WIDTH or height < self.MIN_HEIGHT:
            warnings.append(
                f"Low image resolution: {width}x{height}."
            )

        if blur_score < self.BLUR_WARNING_THRESHOLD:
            warnings.append(
                "Image may be blurry or lack visible detail."
            )

        if brightness < self.BRIGHTNESS_LOW:
            warnings.append(
                "Image may be too dark."
            )

        elif brightness > self.BRIGHTNESS_HIGH:
            warnings.append(
                "Image may be overexposed."
            )

        if contrast < self.CONTRAST_WARNING_THRESHOLD:
            warnings.append(
                "Image has low contrast."
            )

        score = 100.0

        if width < self.MIN_WIDTH or height < self.MIN_HEIGHT:
            score -= 30.0

        if blur_score < self.BLUR_WARNING_THRESHOLD:
            score -= 25.0

        if brightness < self.BRIGHTNESS_LOW:
            score -= 20.0
        elif brightness > self.BRIGHTNESS_HIGH:
            score -= 20.0

        if contrast < self.CONTRAST_WARNING_THRESHOLD:
            score -= 15.0

        score = max(0.0, min(100.0, score))

        if score >= 80.0:
            status = "good"
        elif score >= 50.0:
            status = "warning"
        else:
            status = "poor"

        return ImageQualityResult(
            status=status,
            score=round(score, 2),
            width=width,
            height=height,
            blur_score=round(blur_score, 2),
            brightness=round(brightness, 2),
            contrast=round(contrast, 2),
            warnings=warnings,
        )