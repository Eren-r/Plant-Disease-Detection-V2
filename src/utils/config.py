import os
from pathlib import Path


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]


# ------------------------------------------------------------------
# Application
# ------------------------------------------------------------------

APP_NAME = os.getenv(
    "PLANT_APP_NAME",
    "Plant Disease Detection",
)

APP_VERSION = os.getenv(
    "PLANT_APP_VERSION",
    "2.0.0",
)


# ------------------------------------------------------------------
# API
# ------------------------------------------------------------------

API_HOST = os.getenv(
    "PLANT_API_HOST",
    "127.0.0.1",
)

API_PORT = int(
    os.getenv(
        "PLANT_API_PORT",
        "8000",
    )
)

API_URL = os.getenv(
    "PLANT_API_URL",
    f"http://{API_HOST}:{API_PORT}",
)


# ------------------------------------------------------------------
# Uploads
# ------------------------------------------------------------------

MAX_FILE_SIZE = int(
    os.getenv(
        "PLANT_MAX_FILE_SIZE",
        str(10 * 1024 * 1024),
    )
)

ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
}


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------

MODEL_NAME = os.getenv(
    "PLANT_MODEL_NAME",
    "EfficientNet-B0",
)

MODEL_CHECKPOINT = Path(
    os.getenv(
        "PLANT_MODEL_CHECKPOINT",
        str(
            PROJECT_ROOT
            / "models"
            / "checkpoints"
            / "efficientnet_b0_best.pth"
        ),
    )
)

CALIBRATION_FILE = Path(
    os.getenv(
        "PLANT_CALIBRATION_FILE",
        str(
            PROJECT_ROOT
            / "reports"
            / "results"
            / "efficientnet_b0_calibration.json"
        ),
    )
)

KNOWLEDGE_BASE_FILE = Path(
    os.getenv(
        "PLANT_KNOWLEDGE_BASE_FILE",
        str(
            PROJECT_ROOT
            / "knowledge_base"
            / "diseases.json"
        ),
    )
)


# ------------------------------------------------------------------
# Inference
# ------------------------------------------------------------------

TOP_K = int(
    os.getenv(
        "PLANT_TOP_K",
        "3",
    )
)