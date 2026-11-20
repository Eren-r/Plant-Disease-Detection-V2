"""Run experimental inference on an image already known to contain a grape leaf."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from v3.grapes.train import build_model


def predict(image_path: Path, checkpoint_path: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    preprocessing = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(checkpoint["mean"], checkpoint["std"]),
    ])
    with Image.open(image_path) as image:
        tensor = preprocessing(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu().tolist()
    predictions = sorted(zip(checkpoint["classes"], probabilities),
                         key=lambda pair: pair[1], reverse=True)
    return {
        "model": checkpoint["model_name"],
        "image": image_path.name,
        "device": str(device),
        "crop_selection": "Grapes (assumed, not detected)",
        "calibrated": False,
        "predictions": [{"class_name": name, "score": round(float(score), 6)}
                        for name, score in predictions],
        "warning": "Experimental four-class grape inference. No grape-specific OOD or calibration has been validated."
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(predict(args.image, args.checkpoint), indent=2))


if __name__ == "__main__":
    main()
