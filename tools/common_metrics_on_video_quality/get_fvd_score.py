"""
python get_fvd_score.py --input_path outputs_eval/baseline > outputs_eval/fvd_baseline.txt
python get_fvd_score.py --input_path outputs_eval/stepsize2 > outputs_eval/fvd_stepsize2.txt
python get_fvd_score.py --input_path outputs_eval/stepsize3 > outputs_eval/fvd_stepsize3.txt
python get_fvd_score.py --input_path outputs_eval/stepsize4 > outputs_eval/fvd_stepsize4.txt
python get_fvd_score.py --input_path outputs_eval/reuse2 > outputs_eval/fvd_reuse2.txt
python get_fvd_score.py --input_path outputs_eval/reuse3 > outputs_eval/fvd_reuse3.txt
python get_fvd_score.py --input_path outputs_eval/reuse4 > outputs_eval/fvd_reuse4.txt

# rockets
# 0
python get_fvd_score.py --reference_path outputs_eval/rockets/baseline --input_path outputs_eval/rockets/baseline 
# 14
python get_fvd_score.py --reference_path outputs_eval/rockets/baseline --input_path outputs_eval/rockets/stepsize2 
# 26
python get_fvd_score.py --reference_path outputs_eval/rockets/baseline --input_path outputs_eval/rockets/stepsize3 
# 63
python get_fvd_score.py --reference_path outputs_eval/rockets/baseline --input_path outputs_eval/rockets/stepsize4 
# 217
python get_fvd_score.py --reference_path outputs_eval/rockets/baseline --input_path outputs_eval/rockets/reuse2 

# rockets2
python get_fvd_score.py --reference_path outputs_eval/rockets2/baseline --input_path outputs_eval/rockets2/baseline 
# 19
python get_fvd_score.py --reference_path outputs_eval/rockets2/baseline --input_path outputs_eval/rockets2/level1 
# 670
get_fvd_score.py --reference_path outputs_eval/rockets2/baseline --input_path outputs_eval/rockets2/level2 
# 1269
python get_fvd_score.py --reference_path outputs_eval/rockets2/baseline --input_path outputs_eval/rockets2/level3 


# cocoeval_svd
## test
python get_fvd_score.py --reference_path ../../outputs_eval/svd/baseline --input_path ../../outputs_eval/svd/baseline 

## baseline and step2, 147
python get_fvd_score.py --reference_path ../../outputs_eval/svd/baseline --input_path ../../outputs_eval/svd/step2 

## baseline and async2, 293
python get_fvd_score.py --reference_path ../../outputs_eval/svd/baseline --input_path ../../outputs_eval/svd/async2 

## baseline and step3, 282
python get_fvd_score.py --reference_path ../../outputs_eval/svd/baseline --input_path ../../outputs_eval/svd/step3

## baseline and async3, 515
python get_fvd_score.py --reference_path ../../outputs_eval/svd/baseline --input_path ../../outputs_eval/svd/async3 
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
    """
    从视频文件中提取所有帧并转换为 NumPy 数组列表
    """
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV 读取的是 BGR，需要转换为 RGB
        frames.append(frame)

    cap.release()
    cv2.destroyAllWindows()
    
    return np.array(frames)  # 返回所有帧的数组

def trans(x):
    """
    调整视频数据的维度，以匹配 FVD 计算的输入格式
    """
    if x.shape[-3] == 1:
        x = x.repeat(1, 1, 3, 1, 1)
    x = x.permute(0, 2, 1, 3, 4)  # BTCHW -> BCTHW
    return x

def calculate_fvd(videos1, videos2, device, only_final=True):
    """
    计算两个视频集之间的 FVD
    """
    i3d = load_i3d_pretrained(device=device)
    
    videos1 = trans(videos1)
    videos2 = trans(videos2)

    fvd_results = []
    
    assert videos1.shape == videos2.shape, "Input videos must have the same shape!"
    batchsize = videos1.shape[0]
    for b_idx in tqdm(range(0, batchsize)):
        video1 = videos1[b_idx:b_idx+1]
        video2 = videos2[b_idx:b_idx+1]
        feat1 = get_fvd_feats(video1, i3d=i3d, device=device)
        feat2 = get_fvd_feats(video2, i3d=i3d, device=device)
        fvd_result = frechet_distance(feat1, feat2)
        fvd_results.append(fvd_result)
    # feats1 = get_fvd_feats(videos1, i3d=i3d, device=device)
    # feats2 = get_fvd_feats(videos2, i3d=i3d, device=device)
    
    # fvd_result = frechet_distance(feats1, feats2)
    return fvd_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default='datasets/ucf101_mini_mini')
    parser.add_argument("--input_path", type=str, default='outputs_eval/stepsize2')
    parser.add_argument("--reference_path", type=str, default='outputs_eval/baseline')
    parser.add_argument("--device", type=str, default='cuda')
    
    args = parser.parse_args()
    input_path = Path(args.input_path)
    reference_path = Path(args.reference_path)
    
    video_files = sorted(input_path.glob("*.mp4"))
    reference_videos = sorted(reference_path.glob("*.mp4"))
    
    if not video_files or not reference_videos:
        print("No video files found in:", args.input_path, "or", args.reference_path)
        exit(1)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    input_videos = []
    reference_videos_array = []
    
    for video_file in video_files:
        frames = extract_video_frames(video_file)
        if len(frames) == 0:
            print(f"Warning: No frames found in {video_file}, skipping...")
            continue
        input_videos.append(frames)
    
    for ref_video in reference_videos:
        ref_frames = extract_video_frames(ref_video)
        if len(ref_frames) == 0:
            print(f"Warning: No frames found in {ref_video}, skipping...")
            continue
        reference_videos_array.append(ref_frames)
    
    if not input_videos or not reference_videos_array:
        print("No valid videos processed.")
        exit(1)
    
    input_videos = torch.tensor(np.array(input_videos)).permute(0, 1, 4, 2, 3).float() / 255.0
    reference_videos_array = torch.tensor(np.array(reference_videos_array)).permute(0, 1, 4, 2, 3).float() / 255.0
    print(f"input_videos.shape=={input_videos.shape}")
    print(f"reference_videos_array.shape=={reference_videos_array.shape}")
    fvd_values = calculate_fvd(input_videos, reference_videos_array, device)
    mean_fvd_values = sum(fvd_values) / len(fvd_values)
    print(f"Mean FVD Score: {mean_fvd_values:.4f}")
    print(f"All FVD Score: {fvd_values}")
