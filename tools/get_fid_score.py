"""
比较两个文件夹下的图片的FID分数
given reference(baseline images) and target(the images of ParaStep or others), get the FID score
# SD3
CUDA_VISIBLE_DEVICES=3 python tools/get_fid_score.py --reference_dir outputs_eval/sd3/coco-mini-10/reference --target_dir outputs_eval/sd3/coco-mini-10/step2
"""

from PIL import Image
import os
import numpy as np
import argparse
from torchvision.transforms import functional as F
import torch
from torchmetrics.image.fid import FrechetInceptionDistance

def preprocess_image(image):
    image = torch.tensor(image).unsqueeze(0)
    image = image.permute(0, 3, 1, 2) / 255.0
    return F.center_crop(image, (256, 256))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_dir", type=str, default='outputs_eval/sd3/coco-mini-10/reference')
    parser.add_argument("--target_dir", type=str, default='outputs_eval/sd3/coco-mini-10/baseline')

    args = parser.parse_args()

    reference_dir = args.reference_dir
    target_dir = args.target_dir

    # dataset_path = "sample-imagenet-images"
    # image_paths = sorted([os.path.join(dataset_path, x) for x in os.listdir(dataset_path)])
    reference_paths = sorted([os.path.join(reference_dir, x) for x in os.listdir(reference_dir)])
    target_paths = sorted([os.path.join(target_dir, x) for x in os.listdir(target_dir)])

    # real_images = [np.array(Image.open(path).convert("RGB")) for path in image_paths]
    reference_images = [np.array(Image.open(path).convert("RGB")) for path in reference_paths if path.endswith(".jpg")]
    target_images = [np.array(Image.open(path).convert("RGB")) for path in target_paths if path.endswith(".jpg")]


    # real_images = torch.cat([preprocess_image(image) for image in real_images])
    reference_images = torch.cat([preprocess_image(image) for image in reference_images])
    target_images = torch.cat([preprocess_image(image) for image in target_images])
    # print(real_images.shape)
    # torch.Size([10, 3, 256, 256])
    print(f"reference_images.shape: {reference_images.shape}")
    print(f"target_images.shape: {target_images.shape}")


    fid = FrechetInceptionDistance(normalize=True)
    # fid.update(real_images, real=True)
    # fid.update(fake_images, real=False)
    fid.update(reference_images, real=True)
    fid.update(target_images, real=False)

    print(f"reference_dir: {reference_dir}")
    print(f"target_dir: {target_dir}")
    print(f"FID: {float(fid.compute())}")
    # FID: 177.7147216796875