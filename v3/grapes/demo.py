"""Experimental Streamlit Grapes page. Does not modify the V2 FastAPI service.

Run from repository root:
    streamlit run v3/grapes/demo.py --server.port 8502

Optional environment variable: GRAPES_MODEL_PATH (absolute path to trained checkpoint).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v3.grapes.predict import predict  # noqa: E402

st.set_page_config(page_title="Grapes Disease Research Demo", page_icon="🍇", layout="centered")
st.title("🍇 Grapes Disease Detection — Experimental V3")
st.caption("Standalone four-class research model. Existing V2 Potato/Tomato API is unchanged.")
st.warning(
    "Use only confirmed grape leaf images. The system cannot identify crops, detect arbitrary "
    "new diseases, or reliably recognize unknown inputs. Scores are not yet calibrated. "
    "Do not make treatment decisions from this demonstration."
)

DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "grapes" / "efficientnet_b0_grapes_best.pth"
checkpoint_text = st.text_input(
    "Full path to the trained grape model checkpoint (.pth)",
    value=os.environ.get("GRAPES_MODEL_PATH", str(DEFAULT_CHECKPOINT)),
    help="Defaults to models/grapes/efficientnet_b0_grapes_best.pth inside this project. "
         "You can override it with GRAPES_MODEL_PATH.",
)
uploaded = st.file_uploader("Upload a grape leaf image", type=["jpg", "jpeg", "png", "webp", "bmp"])

if uploaded is not None:
    st.image(uploaded, caption="Selected grape leaf", use_container_width=True)

if st.button("Analyze grape leaf", type="primary", disabled=(uploaded is None)):
    checkpoint = Path(checkpoint_text.strip().strip('"'))
    if not checkpoint.is_file():
        st.error(f"Grape checkpoint not found at: {checkpoint}. Confirm the file exists.")
        st.stop()

    image_bytes = uploaded.getvalue()
    if not image_bytes or len(image_bytes) > 10 * 1024 * 1024:
        st.error("Upload a non-empty image no larger than 10 MB.")
        st.stop()

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError):
        st.error("The uploaded file is not a valid image.")
        st.stop()

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix.lower()) as file:
            file.write(image_bytes)
            tmp_path = Path(file.name)
        with st.spinner("Running experimental prediction..."):
            result = predict(tmp_path, checkpoint)

        best = result["predictions"][0]
        st.subheader("Prediction")
        st.metric("Predicted grape class", best["class_name"].replace("Grapes_", "").replace("_", " "))
        st.metric("Uncalibrated model score", f"{best['score']:.1%}")
        st.subheader("All four classes")
        for record in result["predictions"]:
            st.write(f"**{record['class_name']}** — {record['score']:.1%}")
            st.progress(min(max(record["score"], 0.0), 1.0))

        kb_path = PROJECT_ROOT / "knowledge_base" / "grapes.json"
        if kb_path.exists():
            knowledge = json.loads(kb_path.read_text(encoding="utf-8"))
            details = knowledge.get(best["class_name"], {})
            if details:
                st.subheader("Draft disease information")
                st.write(details.get("advisory", ""))
                st.caption("Expert review status: " + details.get("review_status", "not available"))
        st.info(result["warning"])
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        st.error(f"Could not run grape inference: {type(exc).__name__}: {exc}")
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
