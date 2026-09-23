import requests
import streamlit as st

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import (
    API_URL,
    APP_NAME,
    MODEL_NAME,
)


st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
)


if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "gradcam_image" not in st.session_state:
    st.session_state.gradcam_image = None


st.title(f"{APP_NAME}")

st.caption(
    "AI-powered plant disease screening "
    f"using {MODEL_NAME}"
)


with st.sidebar:

    st.header("System Information")

    st.write(
        f"Model: **{MODEL_NAME}**"
    )

    st.write(
        "Backend: **FastAPI**"
    )

    st.write(
        "Inference: **GPU accelerated**"
    )

    st.divider()

    st.info(
        "This system provides AI-assisted screening "
        "and should not replace professional agricultural diagnosis."
    )


uploaded_file = st.file_uploader(
    "Upload a plant leaf image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp",
        "bmp",
    ],
)


if uploaded_file is not None:

    col1, col2 = st.columns(
        [1, 1]
    )

    with col1:

        st.subheader(
            "Uploaded Image"
        )

        st.image(
            uploaded_file,
            use_container_width=True,
        )

    with col2:

        st.subheader(
            "Analysis"
        )

        analyze = st.button(
            "🔍 Analyze Leaf",
            type="primary",
            use_container_width=True,
        )

        if analyze:

            with st.spinner(
                "Analyzing image..."
            ):

                try:

                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type,
                        )
                    }

                    response = requests.post(
                        f"{API_URL}/predict",
                        files=files,
                        timeout=60,
                    )

                except requests.RequestException as exc:

                    st.error(
                        "Could not connect to the "
                        "FastAPI backend."
                    )

                    st.code(
                        str(exc)
                    )

                    st.stop()

                if response.status_code != 200:

                    st.error(
                        f"API request failed "
                        f"(HTTP {response.status_code})."
                    )

                    try:
                        st.json(
                            response.json()
                        )

                    except Exception:
                        st.write(
                            response.text
                        )

                    st.stop()

                st.session_state.analysis_result = (
                    response.json()["result"]
                )

                st.session_state.gradcam_image = None

            st.success(
                "Analysis completed."
            )


    result = (
        st.session_state.analysis_result
    )


    if result is not None:

        prediction = result[
            "prediction"
        ]

        disease = result[
            "disease"
        ]

        ood = result.get("ood")

        if ood is not None:

            st.subheader(
                "🛡️ Input Novelty Check"
            )

            ood_col1, ood_col2 = st.columns(2)

            with ood_col1:
                if ood["is_ood"]:
                    st.warning(
                        "⚠️ Out-of-distribution"
                    )
                else:
                    st.success(
                        "✅ In-distribution"
                    )

            with ood_col2:
                st.metric(
                    "Similarity",
                    f"{ood['similarity']:.4f}",
                )

            if ood["is_ood"]:
                st.warning(
                    "This image appears different from "
                    "the supported image distribution. "
                    "Treat the prediction with caution."
                )
            else:
                st.success(
                    "The image is consistent with the "
                    "supported image distribution."
                )

            st.caption(
                f"Detection threshold: "
                f"{ood['threshold']:.4f}"
            )

            nearest_class = (
                ood["nearest_class"]
                .replace("___", " - ")
                .replace("_", " ")
            )

            st.caption(
                f"Nearest known class: {nearest_class}"
            )

        st.divider()

        st.subheader(
            "Prediction"
        )

        quality = result.get(
            "image_quality"
        )

        if quality is not None:

            st.subheader(
                "Image Quality"
            )

            quality_col1, quality_col2 = (
                st.columns(2)
            )

            with quality_col1:
                st.metric(
                    "Quality Score",
                    f"{quality['score']:.0f}/100",
                )

            with quality_col2:
                st.metric(
                    "Status",
                    quality["status"].title(),
                )

            if quality["status"] == "good":
                st.success(
                    "Image quality is suitable for analysis."
                )

            elif quality["status"] == "warning":
                st.warning(
                    "Image quality has some limitations. "
                    "Consider uploading a clearer image."
                )

            else:
                st.error(
                    "Image quality is poor. "
                    "A clearer image is recommended."
                )

            for warning in quality["warnings"]:
                st.caption(
                    f"⚠️ {warning}"
                )

            st.divider()

        st.divider()

        st.subheader(
            "Prediction"
        )

        metric1, metric2, metric3 = (
            st.columns(3)
        )

        with metric1:

            st.metric(
                "Crop",
                disease["crop"],
            )

        with metric2:

            st.metric(
                "Disease",
                disease["disease"],
            )

        with metric3:

            st.metric(
                "Confidence",
                f"{prediction['confidence']:.2%}",
            )


        st.divider()

        st.subheader(
            "Top Predictions"
        )

        for index, item in enumerate(
            prediction["top_k"],
            start=1,
        ):

            class_name = item[
                "class"
            ]

            confidence = item[
                "confidence"
            ]

            readable_name = (
                class_name
                .replace(
                    "___",
                    " - ",
                )
                .replace(
                    "_",
                    " ",
                )
            )

            st.write(
                f"**{index}. "
                f"{readable_name}**"
            )

            st.progress(
                min(
                    max(
                        confidence,
                        0.0,
                    ),
                    1.0,
                )
            )

            st.caption(
                f"{confidence:.2%}"
            )


        st.divider()

        st.subheader(
            "🔬 Model Explanation"
        )

        st.write(
            "Grad-CAM highlights image regions "
            "that contributed to the model's prediction."
        )

        explain = st.button(
            "🧠 Generate Grad-CAM",
            use_container_width=True,
        )

        if explain:

            with st.spinner(
                "Generating explanation..."
            ):

                try:

                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type,
                        )
                    }

                    response = requests.post(
                        f"{API_URL}/explain",
                        files=files,
                        timeout=60,
                    )

                    if response.status_code != 200:

                        st.error(
                            f"Grad-CAM generation failed "
                            f"(HTTP {response.status_code})."
                        )

                        try:
                            st.json(
                                response.json()
                            )
                        except Exception:
                            st.code(
                                response.text
                            )

                    else:

                        st.session_state.gradcam_image = (
                            response.content
                        )

                except requests.RequestException as exc:

                    st.error(
                        "Could not connect to "
                        "the FastAPI backend."
                    )

                    st.code(
                        str(exc)
                    )


        if (
            st.session_state.gradcam_image
            is not None
        ):

            st.image(
                st.session_state.gradcam_image,
                caption=(
                    "Grad-CAM attention map"
                ),
                use_container_width=True,
            )


        st.divider()

        st.subheader(
            "Disease Information"
        )

        st.write(
            f"**Status:** "
            f"{disease['status']}"
        )

        st.write(
            f"**Severity:** "
            f"{disease['severity']}"
        )

        st.write(
            "**Symptoms**"
        )

        for symptom in disease[
            "symptoms"
        ]:

            st.write(
                f"• {symptom}"
            )


        st.write(
            "**General Management**"
        )

        for action in disease[
            "general_management"
        ]:

            st.write(
                f"• {action}"
            )


        st.write(
            "**Advisory**"
        )

        st.info(
            disease["advisory"]
        )


        st.caption(
            f"Model: {result['model']} | "
            f"Calibration temperature: "
            f"{result['temperature']:.4f}"
        )

else:

    st.info(
        "Upload a clear plant leaf image to begin."
    )