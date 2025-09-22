"""
CUDA_VISIBLE_DEVICES=0 python tools/common_metrics_on_video_quality/get_fvd_and_psnr.py \
    --reference_dir outputs_eval/svd/baseline \
    --target_dir outputs_eval/svd/step2

CUDA_VISIBLE_DEVICES=0 python tools/common_metrics_on_video_quality/get_fvd_and_psnr.py \
    --reference_dir outputs_eval/svd/baseline \
    --target_dir outputs_eval/svd/async2
"""

import torch
import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from fvd.styleganv.fvd import get_fvd_feats, frechet_distance, load_i3d_pretrained
from torchmetrics.image import PeakSignalNoiseRatio

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def extract_video_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.array(frames)

def trans(x):
    if x.shape[-3] == 1:
        x = x.repeat(1, 1, 3, 1, 1)
    return x.permute(0, 2, 1, 3, 4)

def calculate_fvd(videos1, videos2, device):
    i3d = load_i3d_pretrained(device=device)
    videos1, videos2 = trans(videos1), trans(videos2)
    fvd_scores = []
    for idx in tqdm(range(videos1.shape[0])):
        feat1 = get_fvd_feats(videos1[idx:idx+1], i3d=i3d, device=device)
        feat2 = get_fvd_feats(videos2[idx:idx+1], i3d=i3d, device=device)
        fvd_scores.append(frechet_distance(feat1, feat2))
    return fvd_scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_dir", type=str, required=True)
    parser.add_argument("--reference_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    input_files = sorted(Path(args.target_dir).glob("*.mp4"))
    ref_files   = sorted(Path(args.reference_dir).glob("*.mp4"))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    input_videos = [extract_video_frames(f) for f in input_files]
    ref_videos   = [extract_video_frames(f) for f in ref_files]

    input_tensor = torch.tensor(np.stack(input_videos)).permute(0,1,4,2,3).float()/255.0
    ref_tensor   = torch.tensor(np.stack(ref_videos)).permute(0,1,4,2,3).float()/255.0

    # FVD
    fvd_scores = calculate_fvd(input_tensor, ref_tensor, device)
    print(f"Mean FVD Score: {np.mean(fvd_scores):.4f}")
    print(f"All FVD Scores: {fvd_scores}")

    # PSNR
    psnr_values = []
    for i in range(input_tensor.shape[0]):
        inp = input_tensor[i]
        ref = ref_tensor[i]
        min_len = min(inp.shape[0], ref.shape[0])
        psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
        score = psnr_metric(inp[:min_len].to(device), ref[:min_len].to(device)).item()
        psnr_values.append(score)

    print(f"Mean PSNR: {np.mean(psnr_values):.4f} dB")
    print(f"All PSNR Scores: {psnr_values}")
