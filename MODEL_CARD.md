Model Card — Plant Disease Detection V2

Model

Model name: EfficientNet-B0
Framework: PyTorch
Task: Six-class plant leaf image classification
Input: RGB leaf image
Input resolution: 224 × 224 after preprocessing
Primary checkpoint: models/checkpoints/efficientnet_b0_best.pth

The model is an ImageNet-pretrained EfficientNet-B0 fine-tuned for classification of Potato and Tomato leaf images.

Intended Use

The model is intended for AI-assisted visual screening and decision support for:

Potato Early Blight

Potato Late Blight

Potato Healthy

Tomato Early Blight

Tomato Late Blight

Tomato Healthy

The application is intended for educational, prototyping, and decision-support use.

It is not a substitute for professional agricultural diagnosis and should not be used as the sole basis for treatment or crop-management decisions.

Out-of-Scope Use

The current model should not be treated as a general plant-disease detector.

It has not been validated as a reliable classifier for:

crops outside Potato and Tomato

diseases outside the six supported classes

arbitrary field conditions

severely degraded images

unseen camera or imaging environments

images substantially different from the training distribution

Dataset

Total validated images: 6,652

Split

Images

Train

4,655

Validation

996

Test

1,001

All 6,652 images passed the project dataset validation process.

A group-aware splitting strategy was used to reduce leakage from duplicated or closely related source images.

Validated cross-split group overlap:

Train <-> Validation: 0 overlapping groups
Train <-> Test:       0 overlapping groups
Validation <-> Test:  0 overlapping groups

Dataset provenance

The current repository documentation does not provide a complete external provenance record for the source dataset. Therefore, no claim about a specific public dataset origin is made here.

Class distribution

The dataset is not perfectly balanced across classes. In particular, the number of healthy and diseased examples differs substantially between some classes.

Inverse-frequency class weighting was used during training to reduce the effect of class imbalance.

Training

The primary model uses transfer learning from an ImageNet-pretrained EfficientNet-B0.

Key training configuration:

Parameter

Value

Model

EfficientNet-B0

Input size

224 × 224

Optimizer

AdamW

Initial learning rate

1e-4

Weight decay

1e-4

Maximum epochs

15

Early stopping patience

4

Loss

Weighted Cross-Entropy

Random seed

42

Training device

CUDA when available

Best checkpoint

Validation macro F1

Training data augmentation includes random resized cropping, horizontal flipping, rotation, color jitter, and ImageNet normalization.

Validation and test preprocessing uses resizing, center cropping, tensor conversion, and ImageNet normalization.

Evaluation

The primary performance claim is based on the held-out group-aware test set of 1,001 images.

Overall performance

Metric

Result

Accuracy

99.90%

Macro Precision

99.89%

Macro Recall

99.94%

Macro F1

99.92%

Weighted F1

99.90%

Per-class performance

Class

Precision

Recall

F1

Support

Potato Early Blight

1.0000

1.0000

1.0000

150

Potato Late Blight

1.0000

1.0000

1.0000

150

Potato Healthy

1.0000

1.0000

1.0000

24

Tomato Early Blight

0.9934

1.0000

0.9967

150

Tomato Late Blight

1.0000

0.9965

0.9983

287

Tomato Healthy

1.0000

1.0000

1.0000

240

The held-out test set contained one classification error:

Actual:      Tomato Late Blight
Predicted:   Tomato Early Blight
Confidence:  0.9253

These measurements describe performance on the evaluated dataset and should not be interpreted as a guarantee of equivalent performance in real agricultural environments.

Model Comparison

A robust ResNet18 model was retained as a reference benchmark.

Model

Parameters

Batched Latency / Image

Throughput

Checkpoint

ResNet18 Robust

11,179,590

0.987 ms

1013.20 img/s

42.72 MB

EfficientNet-B0

4,015,234

1.371 ms

729.40 img/s

15.61 MB

The EfficientNet-B0 checkpoint is substantially smaller while providing stronger grouped held-out classification performance in this project.

Additional benchmark artifacts are available under:

reports/results/

Confidence Calibration

Temperature scaling was evaluated on the validation set.

Metric

Before

After

Accuracy

0.9970

0.9970

NLL

0.0211

0.0211

ECE

0.0038

0.0037

Mean Confidence

0.9944

0.9946

Learned temperature:

0.984268

The small change in calibration metrics indicates that the model was already reasonably calibrated on the validation distribution.

Confidence values should not be interpreted as guaranteed probabilities of real-world diagnostic correctness.

The current system deliberately does not use a hard-coded confidence threshold as an automatic diagnosis/review rule.

Explainability

The application provides Grad-CAM explanations for model predictions.

Grad-CAM targets the final convolutional feature representation of EfficientNet-B0 and produces a heatmap that can be overlaid on the input image.

Relevant project artifacts include:

src/explainability/gradcam.py
src/explainability/generate_gradcam.py
reports/figures/efficientnet_b0_gradcam_error.png

Grad-CAM is intended to provide an interpretable visual indication of the image regions contributing to the model output. It should not be interpreted as proof that the highlighted region represents the true biological cause of the disease.

Misclassification Analysis

The validation analysis identified a small number of errors concentrated in the:

Tomato Late Blight -> Tomato Early Blight

class pair.

This pattern was not converted into a hard-coded correction rule because the observed number of validation errors was too small to justify modifying model predictions based on that pair alone.

Detailed analysis artifacts are stored under:

reports/results/

Knowledge Base

The inference system combines the model prediction with a structured disease knowledge base:

knowledge_base/diseases.json

The knowledge base contains information for all six supported classes, including:

crop

disease/status

severity

symptoms

general management

advisory information

The knowledge base is informational and does not constitute expert agronomic advice.

Inference System

The model is integrated into a unified inference service and exposed through FastAPI.

Main endpoints:

GET  /health
GET  /model-info
POST /predict
POST /explain

The system also includes a Streamlit interface for interactive image upload and prediction.

Hardware and Efficiency

Development and benchmarking were performed with CUDA acceleration on an NVIDIA GeForce RTX 3050 Laptop GPU with 4 GB VRAM.

The production-oriented inference path is designed to support CPU or CUDA environments depending on installed dependencies.

The project includes separate dependency files for:

requirements.txt
requirements-ci.txt
requirements-gpu.txt

Limitations

The following limitations are important:

Evaluation is limited to six Potato and Tomato classes.

The dataset may not represent the diversity of real agricultural environments.

Field conditions may differ from the evaluated images in lighting, camera characteristics, background, scale, and image quality.

Unseen diseases and crops are outside the validated task definition.

The system does not currently contain a dedicated out-of-distribution detector.

Confidence calibration was evaluated on the validation distribution.

High confidence does not guarantee correctness on out-of-distribution inputs.

The knowledge base contains general guidance rather than crop-specific expert recommendations.

External field validation has not yet been performed in this repository.

Risk Considerations

Potential failure modes include:

visually similar disease symptoms

low-quality images

partial or occluded leaves

unusual lighting

background artifacts

multiple leaves or mixed conditions in one image

disease stages not represented in the dataset

crops or diseases outside the supported classes

For these reasons, predictions should be treated as screening outputs rather than definitive diagnoses.

Reproducibility

The repository records important artifacts required to reproduce and audit the project:

models/checkpoints/
reports/results/
reports/figures/
training/
src/
tests/

Recorded experiment artifacts include:

training histories

confusion matrices

test metrics

misclassification analysis

confidence calibration

class-pair risk analysis

model efficiency benchmarks

single-image latency benchmarks

The project uses fixed random seeds for the documented training experiments.

CI and Quality Checks

GitHub Actions validates the software stack using a CPU environment.

The CI workflow performs:

Install dependencies
        ↓
pip check
        ↓
pytest

The current API regression suite contains six tests covering health, model metadata, prediction, input validation, and Grad-CAM behavior.

Version Status

This document describes the current Plant Disease Detection System V2 implementation.

The project includes:

EfficientNet-B0 primary model

confidence calibration

Grad-CAM explainability

FastAPI inference service

Streamlit frontend

structured knowledge base

automated tests

GitHub Actions CI

Docker configuration

Docker configuration is included in the repository, but local Docker image construction has not been validated on the current Windows development machine.

Future Work

Potential future improvements include:

external field-image validation

out-of-distribution detection

image-quality assessment

additional crops and diseases

model quantization

inference optimization

production monitoring

cloud deployment

larger and more diverse datasets

human-in-the-loop review workflows
