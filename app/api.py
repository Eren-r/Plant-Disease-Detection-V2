from io import BytesIO

import matplotlib

matplotlib.use("Agg")

from matplotlib import colormaps
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError

from src.explainability.gradcam import (
    EfficientNetGradCAM,
)
from src.inference.disease_service import (
    DiseasePredictionService,
)


from src.utils.config import (
    ALLOWED_IMAGE_TYPES,
    APP_NAME,
    APP_VERSION,
    MAX_FILE_SIZE,
    MODEL_NAME,
    OOD_ARTIFACT,
)

from src.inference.image_quality import (
    ImageQualityAnalyzer,
)

from src.inference.ood_guard import OODGuard


app = FastAPI(
    title=APP_NAME,
    description=(
        "AI-powered plant disease screening API "
        "using EfficientNet-B0."
    ),
    version=APP_VERSION,
)


service = DiseasePredictionService()

gradcam_service = EfficientNetGradCAM()

ood_guard = OODGuard(
    OOD_ARTIFACT
)
quality_analyzer = ImageQualityAnalyzer()



def load_uploaded_image(
    file_bytes: bytes,
):
    try:
        image = Image.open(
            BytesIO(file_bytes)
        )

        image.load()

        return image.convert("RGB")

    except (
        UnidentifiedImageError,
        OSError,
    ) as exc:

        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image.",
        ) from exc


async def read_image_upload(
    file: UploadFile,
):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename was provided.",
        )

    content_type = (
        file.content_type or ""
    ).lower()

    if content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                "Unsupported image type. "
                "Use JPEG, PNG, WEBP, or BMP."
            ),
        )

    data = await file.read()

    if not data:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image file is larger than 10 MB.",
        )

    image = load_uploaded_image(data)

    return image


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "device": service.predictor.device.type,
        "temperature": service.temperature,
        "gradcam": True,
    }


@app.get("/model-info")
def model_info():
    return {
        "model": "EfficientNet-B0",
        "classes": service.predictor.class_names,
        "temperature": service.temperature,
        "calibration": "temperature_scaling",
        "knowledge_base_classes": (
            service.knowledge_base.classes()
        ),
        "explainability": "Grad-CAM",
        "ood_protection": {
            "enabled": True,
            "method": (
                "EfficientNet-B0 nearest-class "
                "centroid cosine similarity"
            ),
            "threshold": ood_guard.threshold,
        },
        "image_quality": {
            "enabled": True,
            "mode": "advisory",
        },
        "service_role": "screening_decision_support",
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = await read_image_upload(
        file
    )

    quality = quality_analyzer.analyze(
        image
    )

    ood_result = ood_guard.analyze(
        image
    )

    try:

        result = service.predict(
            image,
            top_k=3,
        )

        result["image_quality"] = (
            quality.to_dict()
        )

        result["ood"] = ood_result

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction failed. "
                f"{type(exc).__name__}"
            ),
        ) from exc

    return {
        "filename": file.filename,
        "result": result,
    }


@app.post("/explain")
async def explain(
    file: UploadFile = File(...)
):
    image = await read_image_upload(
        file
    )

    try:

        heatmap, info = (
            gradcam_service.generate(
                image
            )
        )

        image_array = np.asarray(
            image
        )

        height, width = (
            image_array.shape[:2]
        )

        heatmap_image = Image.fromarray(
            np.uint8(
                np.clip(
                    heatmap,
                    0.0,
                    1.0,
                )
                * 255
            )
        )

        heatmap_image = heatmap_image.resize(
            (width, height),
            Image.Resampling.BILINEAR,
        )

        heatmap_array = (
            np.asarray(
                heatmap_image
            )
            / 255.0
        )

        colormap = colormaps["jet"]

        colored_heatmap = (
            colormap(
                heatmap_array
            )[:, :, :3]
            * 255
        ).astype(
            np.uint8
        )

        heatmap_rgb = Image.fromarray(
            colored_heatmap
        )

        overlay = Image.blend(
            image,
            heatmap_rgb,
            alpha=0.45,
        )

        output = BytesIO()

        overlay.save(
            output,
            format="PNG",
        )

        response = Response(
            content=output.getvalue(),
            media_type="image/png",
        )

        response.headers[
            "X-Predicted-Class"
        ] = info["predicted_class"]

        response.headers[
            "X-Confidence"
        ] = str(
            info["predicted_confidence"]
        )

        return response

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Grad-CAM generation failed. "
                f"{type(exc).__name__}"
            ),
        ) from exc