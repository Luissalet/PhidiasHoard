"""Run local Depth Anything inference in a Python with Torch and Transformers installed."""

from pathlib import Path
import sys

from PIL import Image
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


def main():
    model_dir, folder = map(Path, sys.argv[1:])
    torch.set_num_threads(4)
    processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True, use_fast=False)
    model = AutoModelForDepthEstimation.from_pretrained(model_dir, local_files_only=True, use_safetensors=True).eval()
    for path in sorted(folder.glob("*.png")):
        with Image.open(path) as src:
            image = src.convert("RGB")
        with torch.inference_mode():
            raw = model(**processor(images=image, return_tensors="pt")).predicted_depth[:, None]
            depth = torch.nn.functional.interpolate(raw, size=(image.height, image.width), mode="bilinear",
                                                    align_corners=False)[0, 0]
        np.save(path.with_suffix(".npy"), depth.numpy().astype(np.float32))


if __name__ == "__main__":
    main()
