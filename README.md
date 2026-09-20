@'
# 🌿 Plant Disease Detection System — V2

AI-assisted plant disease screening system for Potato and Tomato leaves, built with a reproducible ML pipeline, EfficientNet-B0, FastAPI, Streamlit, Grad-CAM explainability, and a structured disease knowledge base.

> **Important:** This system is a screening and decision-support tool. It is not a substitute for professional agricultural diagnosis.

---

## ✨ Highlights

- Six-class plant disease classification
- Group-aware train/validation/test evaluation
- EfficientNet-B0 transfer-learning model
- **99.90% test accuracy**
- **99.92% test macro F1**
- Held-out test set of **1,001 images**
- No detected cross-split group overlap
- Temperature-scalibrated probabilities
- Top-3 predictions
- Grad-CAM visual explanations
- Structured disease knowledge base
- FastAPI inference backend
- Streamlit user interface
- Automated API regression tests
- GPU inference support with CUDA
- Production-oriented model benchmarking

---

## 🎯 Supported Classes

| Crop | Class |
|---|---|
| Potato | Early Blight |
| Potato | Late Blight |
| Potato | Healthy |
| Tomato | Early Blight |
| Tomato | Late Blight |
| Tomato | Healthy |

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    A[Leaf Image] --> B[Streamlit UI]
    B --> C[FastAPI /predict]
    C --> D[Image Validation]
    D --> E[EfficientNet-B0]
    E --> F[Temperature-Calibrated Probabilities]
    F --> G[Top-3 Prediction]
    G --> H[Disease Knowledge Base]
    H --> I[Prediction + Symptoms + Management]

    B --> J[FastAPI /explain]
    J --> K[Grad-CAM]
    K --> L[Attention Heatmap]
    L --> B