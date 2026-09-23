from io import BytesIO

from PIL import Image

from src.inference.image_quality import ImageQualityAnalyzer


def create_test_image(
    size=(224, 224),
    brightness=128,
):
    image = Image.new(
        "RGB",
        size,
        (brightness, brightness, brightness),
    )

    buffer = BytesIO()
    image.save(buffer, format="JPEG")

    return buffer.getvalue()


def test_normal_image_is_analyzed():
    analyzer = ImageQualityAnalyzer()

    result = analyzer.analyze(
        create_test_image()
    )

    assert result.width == 224
    assert result.height == 224
    assert 0 <= result.score <= 100
    assert result.status in {
        "good",
        "warning",
        "poor",
    }


def test_small_image_generates_warning():
    analyzer = ImageQualityAnalyzer()

    result = analyzer.analyze(
        create_test_image(size=(64, 64))
    )

    assert result.width == 64
    assert result.height == 64
    assert result.status in {
        "warning",
        "poor",
    }


def test_dark_image_generates_warning():
    analyzer = ImageQualityAnalyzer()

    result = analyzer.analyze(
        create_test_image(brightness=10)
    )

    assert any(
        "dark" in warning.lower()
        for warning in result.warnings
    )