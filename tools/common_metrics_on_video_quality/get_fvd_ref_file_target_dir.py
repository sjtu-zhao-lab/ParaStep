"""
reference为一个视频，target为一个文件夹，计算target中每一个mp4对应的FVD score

python get_fvd_ref_file_target_dir.py --reference_path outputs/type3/cycle_len/cogvideox_baseline.mp4 --input_folder outputs/type3/cycle_len

python get_fvd_ref_file_target_dir.py --reference_path outputs/ltxvideo/frames153/ltxvideo_baseline.mp4 --input_folder outputs/ltxvideo/frames153

python get_fvd_ref_file_target_dir.py --reference_path outputs/cogvideox/cogvideox_baseline.mp4 --input_folder outputs/cogvideox/warmup
"""
import torch
import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from fvd.styleganv.fvd import get_fvd_feats, frechet_distance, load_i3d_pretrained
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def extract_video_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    cap.release()
    return np.array(frames)

def trans(x):
    if x.shape[-3] == 1:
        x = x.repeat(1, 1, 3, 1, 1)
    x = x.permute(0, 2, 1, 3, 4)
    return x

def calculate_fvd(videos1, videos2, device):
    i3d = load_i3d_pretrained(device=device)
    videos1, videos2 = trans(videos1), trans(videos2)
    
    feat1 = get_fvd_feats(videos1, i3d=i3d, device=device)
    feat2 = get_fvd_feats(videos2, i3d=i3d, device=device)
    
    return frechet_distance(feat1, feat2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_path", type=str, required=True, help="Path to the reference video (mp4)")
    parser.add_argument("--input_folder", type=str, required=True, help="Folder containing input mp4 videos")
    parser.add_argument("--device", type=str, default='cuda', help="Computation device")
    args = parser.parse_args()
    
    reference_video_path = Path(args.reference_path)
    input_folder = Path(args.input_folder)
    
    if not reference_video_path.exists() or not reference_video_path.suffix == ".mp4":
        print("Error: Reference path must be an existing mp4 file.")
        exit(1)
    
    video_files = sorted(input_folder.glob("*.mp4"))
    if not video_files:
        print("No video files found in:", args.input_folder)
        exit(1)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    reference_frames = extract_video_frames(reference_video_path)
    reference_tensor = torch.tensor(reference_frames).unsqueeze(0).permute(0, 1, 4, 2, 3).float() / 255.0
    
    for video_file in video_files:
        input_frames = extract_video_frames(video_file)
        if len(input_frames) == 0:
            print(f"Warning: No frames found in {video_file}, skipping...")
            continue
        
        input_tensor = torch.tensor(input_frames).unsqueeze(0).permute(0, 1, 4, 2, 3).float() / 255.0
        
        fvd_score = calculate_fvd(input_tensor, reference_tensor, device)
        print(f"FVD Score for {video_file.name}: {fvd_score:.4f}")
