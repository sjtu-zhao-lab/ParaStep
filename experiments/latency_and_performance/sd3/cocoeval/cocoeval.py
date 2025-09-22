import json
import argparse
import os
from diffusers.utils import load_image

import torch
import torch.distributed as dist
import time


import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, default="True", help="baseline or step_parallelism")
    parser.add_argument("--model", type=str, default='stabilityai/stable-diffusion-3-medium-diffusers') 
    #"A cute penguin is watching TV" "A cute dog is watching TV" "A cool boy is playing computer"
    parser.add_argument("--prompt", type=str, default='A cat holding a sign that says hello world')
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--reuse", type=str, default='False')
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--reduce_memory_level", type=int, default=0, help="0: no memory optimization; 1: offload t5 to cpu, and load t5 to gpu when it is needed; 2: split T5encoder to _world_size stages, each stage belong to single device.")
    parser.add_argument("--dataset_size", type=int, default=10, help="size of dataset to be used for evaluation. Default is 10, max is 5000")
    parser.add_argument("--output", type=str, default='experiments/latency_and_performance/svd/cocoeval/outputs')
    parser.add_argument("--height", type=int, default=1440)
    parser.add_argument("--width", type=int, default=1440)
    args = parser.parse_args()

    coco_annotation_path = "datasets/coco/captions_val2017.json"

    use_baseline = False
    _world_size = 1
    _rank = 0
    reuse = False
    if args.reuse == 'True':
        reuse = True

    height = args.height
    width = args.width

    if args.baseline == 'True':
        use_baseline = True
        from diffusers import StableDiffusion3Pipeline
    else:
        _world_size = int(os.environ["WORLD_SIZE"])
        _rank = int(os.environ["RANK"])
        use_baseline = False
        from sd3.stepParallelism_sd3 import StableDiffusion3Pipeline as StableDiffusion3Pipeline_step_parallelism
        from sd3.stepParallelism_sd3 import StepParallelism
    output_path = ""

    if use_baseline:
        output_path = os.path.join(args.output, f"baseline")
    else:
        if reuse == False:
            output_path = os.path.join(args.output, f"step{_world_size}")
        else:
            output_path = os.path.join(args.output, f"reuse{_world_size}")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    latency_output_path = os.path.join(output_path, "0latency.txt")

    # 加载pipeline
    if use_baseline:
        pipeline = StableDiffusion3Pipeline.from_pretrained(args.model, torch_dtype=torch.float16, low_cpu_mem_usage=True, local_files_only=False)
    else:
        pipeline = StableDiffusion3Pipeline_step_parallelism.from_pretrained(args.model, torch_dtype=torch.float16, low_cpu_mem_usage=True, local_files_only=False)

    if use_baseline == False:
        step_parallelism = StepParallelism(pipeline, step_size=int(os.environ["WORLD_SIZE"]), warmp_up=args.warmup, reuse=reuse, reduce_memory_level=args.reduce_memory_level)
    else:
        pipeline.to('cuda')

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # warmup
    image = pipeline(
        args.prompt,
        negative_prompt="",
        num_inference_steps=args.num_inference_steps,
        guidance_scale=7.0,
        height = height,
        width = width
    ).images[0]
    
    if use_baseline == False:
        dist.barrier()
        step_parallelism.clear_cache()

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

            image = pipeline(
                prompt=prompt,
                negative_prompt="",
                num_inference_steps=args.num_inference_steps,
                guidance_scale=7.0,
                height = 1440,
                width = 1440,
                generator = torch.Generator("cuda").manual_seed(args.seed)
            ).images[0]

            if use_baseline == False:
                step_parallelism.clear_cache()

            torch.cuda.synchronize()
            times.append(time.time()-start)

            if _rank == 0:
                image_output_path = os.path.join(output_path, output_id)
                print(f"image_output_path={image_output_path}")
                image.save(image_output_path)
    
    print("Average time taken: ", sum(times)/len(times))
    with open(latency_output_path,"w") as f:
        f.write("Average time taken: {}".format(sum(times)/len(times)))
    if use_baseline == False:
        dist.destroy_process_group()




