"""

"""

from PIL import Image
import os
import numpy as np
import argparse
from torchvision.transforms import functional as F
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image import PeakSignalNoiseRatio

def preprocess_image(image):
    image = torch.tensor(image).unsqueeze(0)
    image = image.permute(0, 3, 1, 2) / 255.0
    return F.center_crop(image, (256, 256))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_dir", type=str, required=True)
    parser.add_argument("--target_dir", type=str, required=True)
    args = parser.parse_args()

    reference_paths = sorted(p for p in os.listdir(args.reference_dir) if p.endswith(".jpg"))
    target_paths = sorted(p for p in os.listdir(args.target_dir) if p.endswith(".jpg"))

    reference_images = torch.cat([
        preprocess_image(np.array(Image.open(os.path.join(args.reference_dir, p)).convert("RGB")))
        for p in reference_paths
    ])
    target_images = torch.cat([
        preprocess_image(np.array(Image.open(os.path.join(args.target_dir, p)).convert("RGB")))
        for p in target_paths
    ])

    # FID
    fid = FrechetInceptionDistance(normalize=True)
    fid.update(reference_images, real=True)
    fid.update(target_images, real=False)
    fid_score = float(fid.compute())

    # PSNR via torchmetrics
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0)
    psnr_score = psnr_metric(target_images, reference_images).item()

    print(f"reference_dir: {args.reference_dir}")
    print(f"target_dir: {args.target_dir}")
    print(f"FID: {fid_score:.4f}")
    print(f"PSNR: {psnr_score:.4f} dB")
