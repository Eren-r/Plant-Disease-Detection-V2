import json
from pathlib import Path

from PIL import Image

from src.inference.predictor import (
    PlantDiseasePredictor,
)

from src.inference.knowledge_base import (
    DiseaseKnowledgeBase,
)

from src.utils.config import (
    CALIBRATION_FILE,
    KNOWLEDGE_BASE_FILE,
    MODEL_CHECKPOINT,
)


class DiseasePredictionService:
    """
    Unified plant-disease inference service.

    Combines:
        EfficientNet-B0 prediction
        confidence calibration
        top-k predictions
        disease knowledge base
    """

    def __init__(
        self,
        checkpoint_path=MODEL_CHECKPOINT,
        knowledge_base_path=KNOWLEDGE_BASE_FILE,
        calibration_path=CALIBRATION_FILE,
    ):
        self.predictor = PlantDiseasePredictor(
            checkpoint_path=checkpoint_path,
        )

        self.knowledge_base = DiseaseKnowledgeBase(
            knowledge_base_path,
        )

        self.calibration_path = Path(
            calibration_path
        )

        self.temperature = 1.0

        if self.calibration_path.exists():
            calibration = json.loads(
                self.calibration_path.read_text(
                    encoding="utf-8"
                )
            )

            self.temperature = float(
                calibration.get(
                    "temperature",
                    1.0,
                )
            )

        # Ensure model classes and KB classes match.
        model_classes = set(
            self.predictor.class_names
        )

        kb_classes = set(
            self.knowledge_base.classes()
        )

        if model_classes != kb_classes:
            missing_from_kb = (
                model_classes - kb_classes
            )

            extra_in_kb = (
                kb_classes - model_classes
            )

            raise ValueError(
                "Model classes and knowledge-base "
                "classes do not match. "
                f"Missing from KB: {missing_from_kb}; "
                f"Extra in KB: {extra_in_kb}"
            )

    def predict(
        self,
        image,
        top_k=3,
    ):
        """
        Run prediction and attach knowledge-base information.
        """

        if not isinstance(
            image,
            Image.Image,
        ):
            raise TypeError(
                "image must be a PIL.Image.Image"
            )

        prediction = self.predictor.predict(
            image,
            top_k=top_k,
        )

        predicted_class = prediction[
            "prediction"
        ]

        disease_info = self.knowledge_base.get(
            predicted_class
        )

        return {
            "model": prediction["model"],
            "device": prediction["device"],
            "temperature": self.temperature,

            "prediction": {
                "class": predicted_class,
                "confidence": prediction[
                    "confidence"
                ],
                "second_best_confidence": prediction[
                    "second_best_confidence"
                ],
                "margin": prediction[
                    "margin"
                ],
                "top_k": prediction["top_k"],
            },

            "disease": disease_info,
        }

    def predict_file(
        self,
        image_path,
        top_k=3,
    ):
        """
        Predict directly from an image file.
        """

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        with Image.open(image_path) as image:
            image = image.convert("RGB")

            return self.predict(
                image,
                top_k=top_k,
            )