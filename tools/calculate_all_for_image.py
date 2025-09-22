from PIL import Image
import os
import numpy as np
import argparse
from torchvision.transforms import functional as F
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import lpips


def preprocess_image(image):
    image = torch.tensor(image).unsqueeze(0)
    image = image.permute(0, 3, 1, 2) / 255.0  # Normalize to [0,1]
    return F.center_crop(image, (256, 256))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_dir", type=str, required=True)
    parser.add_argument("--target_dir", type=str, required=True)
    args = parser.parse_args()

    reference_paths = sorted(p for p in os.listdir(args.reference_dir) if p.endswith(".jpg"))
    target_paths = sorted(p for p in os.listdir(args.target_dir) if p.endswith(".jpg"))

    assert len(reference_paths) == len(target_paths), "Image counts must match."

    reference_images = []
    target_images = []

    lpips_scores = []

    loss_fn = lpips.LPIPS(net='alex').cuda()
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)

    for r_name, t_name in zip(reference_paths, target_paths):
        r_img = Image.open(os.path.join(args.reference_dir, r_name)).convert("RGB")
        t_img = Image.open(os.path.join(args.target_dir, t_name)).convert("RGB")

        r_arr = np.array(r_img)
        t_arr = np.array(t_img)

        r_tensor = preprocess_image(r_arr)
        t_tensor = preprocess_image(t_arr)

        # Accumulate for FID / PSNR / SSIM
        reference_images.append(r_tensor)
        target_images.append(t_tensor)

        # LPIPS
        r_lpips = r_tensor.cuda() * 2 - 1  # Scale to [-1, 1]
        t_lpips = t_tensor.cuda() * 2 - 1
        lpips_score = loss_fn.forward(r_lpips, t_lpips).mean().item()
        lpips_scores.append(lpips_score)

    reference_images = torch.cat(reference_images)
    target_images = torch.cat(target_images)

    # FID
    fid = FrechetInceptionDistance(normalize=True)
    fid.update(reference_images, real=True)
    fid.update(target_images, real=False)
    fid_score = float(fid.compute())

    # PSNR
    psnr_score = psnr_metric(target_images, reference_images).item()

    # SSIM using torchmetrics
    ssim_score = ssim_metric(target_images, reference_images).item()

    # Results
    print(f"reference_dir: {args.reference_dir}")
    print(f"target_dir: {args.target_dir}")
    print(f"FID: {fid_score:.4f}")
    print(f"PSNR: {psnr_score:.4f} dB")
    print(f"SSIM: {ssim_score:.4f}")
    print(f"LPIPS: {np.mean(lpips_scores):.4f}")


if __name__ == "__main__":
    main()
