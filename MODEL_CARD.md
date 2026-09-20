# Model Card - Plant Disease Detection V2

## Model

**Model name:** EfficientNet-B0  
**Framework:** PyTorch  
**Task:** Six-class plant leaf image classification  
**Input:** RGB leaf image  
**Input resolution:** 224 x 224 after preprocessing  
**Primary checkpoint:** `models/checkpoints/efficientnet_b0_best.pth`

---

## Intended Use

The model provides AI-assisted visual screening for:

- Potato Early Blight
- Potato Late Blight
- Potato Healthy
- Tomato Early Blight
- Tomato Late Blight
- Tomato Healthy

This system is intended for decision support and educational use. It is not a substitute for professional agricultural diagnosis.

---

## Dataset

Total images: **6,652**

| Split | Images |
|---|---:|
| Train | 4,655 |
| Validation | 996 |
| Test | 1,001 |

A group-aware splitting strategy was used to reduce source/group leakage.

Validated overlap:

```text
Train <-> Validation: 0 overlapping groups
Train <-> Test:       0 overlapping groups
Validation <-> Test:  0 overlapping groups