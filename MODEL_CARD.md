# Model Card — Plant Disease Detection V2

## Model

**Model name:** EfficientNet-B0  
**Framework:** PyTorch  
**Task:** Six-class plant leaf image classification  
**Input:** RGB leaf image  
**Input resolution:** 224 × 224 after preprocessing  
**Primary checkpoint:** `models/checkpoints/efficientnet_b0_best.pth`

The model is an ImageNet-pretrained EfficientNet-B0 fine-tuned for classification of Potato and Tomato leaf images.

## Intended Use

The model is intended for AI-assisted visual screening and decision support for:

- Potato Early Blight
- Potato Late Blight
- Potato Healthy
- Tomato Early Blight
- Tomato Late Blight
- Tomato Healthy

The application is intended for educational, prototyping, and decision-support use.

It is **not a substitute for professional agricultural diagnosis** and should not be used as the sole basis for treatment or crop-management decisions.

## Out-of-Scope Use

The current model should not be treated as a general plant-disease detector.

It has not been validated as a reliable classifier for:

- crops outside Potato and Tomato
- diseases outside the six supported classes
- arbitrary field conditions
- severely degraded images
- unseen camera or imaging environments
- images substantially different from the training distribution

## Dataset

Total validated images: **6,652**

| Split | Images |
|---|---:|
| Train | 4,655 |
| Validation | 996 |
| Test | 1,001 |

All 6,652 images passed the project dataset validation process.

A group-aware splitting strategy was used to reduce leakage from duplicated or closely related source images.

Validated cross-split group overlap:

```text
Train <-> Validation: 0 overlapping groups
Train <-> Test:       0 overlapping groups
Validation <-> Test:  0 overlapping groups
