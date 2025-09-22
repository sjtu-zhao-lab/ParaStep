import torch
import torch.distributed as dist
import time
import argparse
import os
from svd.stepParallelism_svd import StableVideoDiffusionPipeline, StepParallelism
from diffusers.utils import load_image, export_to_video, export_to_gif

# from diffusers import StableDiffusionParadigmsPipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # or stabilityai/stable-video-diffusion-img2vid-xt
    parser.add_argument("--model", type=str, default='stabilityai/stable-video-diffusion-img2vid') 
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--decode_chunk_size", type=int, default=8)
    parser.add_argument("--num_frames", type=int, default=14)

    args = parser.parse_args()

    # https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png?download=true
    image = load_image("./datasets/rockets/rocket.png")
    file_name = "rocket"
    image = image.resize((1024, 576))

    pipeline = StableVideoDiffusionPipeline.from_pretrained(
        args.model, torch_dtype=torch.float16, 
        use_safetensors=True, low_cpu_mem_usage=True,
        variant="fp16",
        
    )
    # CfgParallelism中对pipeline做cfg_parallelism的初始化工作
    _world_size = int(os.environ["WORLD_SIZE"])
    _rank = int(os.environ["RANK"])
    step_parallelism = StepParallelism(pipeline, _world_size)

    dist.barrier()
    # warm up
    print("warmup")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    frames = pipeline(
            image, 
            decode_chunk_size=args.decode_chunk_size,
            num_inference_steps=50,
            num_frames = args.num_frames
        ).frames[0]
    
    step_parallelism.clear_cache()

    # inference
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    start = time.time()
    frames = pipeline(
            image, 
            decode_chunk_size=args.decode_chunk_size,
            num_inference_steps=50,
            num_frames = args.num_frames,
            generator = torch.Generator("cuda").manual_seed(args.seed)
        ).frames[0]
    print(f"Rank {_rank} StepParallelism Time taken: {time.time()-start:.2f} seconds.")

    if _rank == 0:
        export_to_video(frames, "svd_worldsize_{}.mp4".format(_world_size), fps=7)
    
    dist.destroy_process_group()