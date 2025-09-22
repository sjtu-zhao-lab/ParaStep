"""
python tools/get_clip_score.py --input_path <where you push your videos> > <outputs_name>.txt
"""

import torch
import argparse
import cv2
import numpy as np
from pathlib import Path
from torchmetrics.functional.multimodal import clip_score
from functools import partial
from diffusers.utils import load_image

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 定义 CLIP Score 计算函数
clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

def extract_video_frames(video_path):
    """
    从视频文件中提取所有帧并转换为NumPy数组列表
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

def calculate_clip_score(images, prompts):
    """
    计算一组图片的 CLIP Score
    """
    images_int = (images * 255).astype("uint8")  # 归一化到 0-255
    clip_scores = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
    return round(float(clip_scores.mean()), 4)  # 取均值

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default='datasets/ucf101_mini_mini')
    parser.add_argument("--input_path", type=str, default='outputs_eval/stepsize2')

    args = parser.parse_args()

    input_path = Path(args.input_path)

    # 遍历 input_path 目录下的所有 mp4 文件
    video_files = sorted(input_path.glob("*.mp4"))
    if not video_files:
        print("No video files found in:", args.input_path)
        exit(1)

    total_clip_score = 0
    num_videos = 0

    for video_file in video_files:
        print(f"Processing video: {video_file}")

        # 提取所有帧
        frames = extract_video_frames(video_file)

        if len(frames) == 0:
            print(f"Warning: No frames found in {video_file}, skipping...")
            continue

        # 生成相应的文本 prompt（这里假设使用文件名作为 prompt）
        prompt_name = "yoga"

        prompt = [video_file.stem] * len(frames)

        # 计算 CLIP Score
        clip_score_value = calculate_clip_score(frames / 255.0, prompt)
        print(f"CLIP Score for {video_file}: {clip_score_value}, prompt is {prompt_name}")

        total_clip_score += clip_score_value
        num_videos += 1
        # break

    if num_videos > 0:
        avg_clip_score = total_clip_score / num_videos
        print(f"Average CLIP Score across {num_videos} videos: {avg_clip_score:.4f}")
    else:
        print("No valid videos processed.")