"""
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 --run-path cocoeval.py --model_n 2 --stride 1 --dataset_size 2
"""

import json
import argparse
import os
from diffusers.utils import load_image

import torch
import torch.distributed as dist
from asyncdiff.async_sd3 import AsyncDiff
import time
from diffusers import StableDiffusion3Pipeline



import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='stabilityai/stable-diffusion-3-medium-diffusers') 
    parser.add_argument("--prompt", type=str, default='A cat holding a sign that says hello world')
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--dataset_size", type=int, default=10, help="size of dataset to be used for evaluation. Default is 10, max is 5000")
    parser.add_argument("--output", type=str, default='experiments/latency_and_performance/svd/cocoeval/outputs/async')
    parser.add_argument("--height", type=int, default=1440)
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--model_n", type=int, default=2)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--time_shift", type=bool, default=False)
    args = parser.parse_args()

    coco_annotation_path = "../../datasets/coco/captions_val2017.json"

    _world_size = 1
    _rank = int(os.environ["RANK"])
    

    height = args.height
    width = args.width


    output_path = args.output
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    latency_output_path = os.path.join(output_path, "0latency.txt")

    pipeline = StableDiffusion3Pipeline.from_pretrained(args.model, torch_dtype=torch.float16, low_cpu_mem_usage=True, local_files_only=True)

    async_diff = AsyncDiff(pipeline, model_n=args.model_n, stride=args.stride, time_shift=args.time_shift)


    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # warmup
    print("warmup")
    async_diff.reset_state(warm_up=args.warmup)
    image = pipeline(
        args.prompt,
        negative_prompt="",
        num_inference_steps=5,
        guidance_scale=7.0,
        height = height,
        width = width,
        generator = torch.Generator("cuda").manual_seed(args.seed)
    ).images[0]
    dist.barrier()

    times = []

    with open(coco_annotation_path, 'r') as coco_annotation:
        coco_annotation = json.load(coco_annotation)
        for i in tqdm.tqdm(range(args.dataset_size)):
            caption = coco_annotation['annotations'][i]['caption']
            imageid = coco_annotation['annotations'][i]['image_id']
            output_id = str(coco_annotation['annotations'][i]['id'])
            output_id = f"{str(output_id).zfill(12)}.jpg"
            # imagepath, example: 000000000139, 000000481404.jpg
            imageid = f"{str(imageid).zfill(12)}.jpg"
            
            prompt = caption
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)

            torch.cuda.synchronize()
            start = time.time()
            async_diff.reset_state(warm_up=args.warmup)

            image = pipeline(
                prompt=prompt,
                negative_prompt="",
                num_inference_steps=args.num_inference_steps,
                guidance_scale=7.0,
                height = 1440,
                width = 1440,
                generator = torch.Generator("cuda").manual_seed(args.seed)
            ).images[0]

            torch.cuda.synchronize()
            times.append(time.time()-start)

            if _rank == 0:
                image_output_path = os.path.join(output_path, output_id)
                print(f"image_output_path={image_output_path}")
                image.save(image_output_path)
    
    print("Average time taken: ", sum(times)/len(times))
    with open(latency_output_path,"w") as f:
        f.write("Average time taken: {}".format(sum(times)/len(times)))
    # dist.destroy_process_group()
