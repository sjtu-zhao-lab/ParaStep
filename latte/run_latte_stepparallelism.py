import torch
import torch.distributed as dist
from diffusers import LattePipeline
from latte.stepParallelism_latte import StepParallelism
from diffusers.utils import export_to_video
import argparse
import os
import time



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='stabilityai/stable-diffusion-3-medium-diffusers') 
    parser.add_argument("--prompt", type=str, default="A handsome cat is crying")
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--warm_up", type=int, default=13)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--video_length", type=int, default=16)
    parser.add_argument("--reduce_memory", type=int, default=0)
    parser.add_argument("--warm_gpu", type=str, default="True")

    args = parser.parse_args()

    _world_size = int(os.environ["WORLD_SIZE"])
    _rank = int(os.environ["RANK"])

    warm_gpu = True if args.warm_gpu == "True" else False


    pipe = LattePipeline.from_pretrained(
        "maxin-cn/Latte-1", 
        torch_dtype=torch.float16,
        
    )

    warm_up = args.warm_up

    step_parallelism = StepParallelism(pipe, _world_size,args.warm_up, reduce_memory = args.reduce_memory)

    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    if warm_gpu == True:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        prompt = args.prompt
        video = pipe(
            prompt=prompt,
            num_inference_steps=5,
            height=args.height,
            width=args.width,
            video_length=args.video_length,
            guidance_scale=6,
            # generator=torch.Generator(device=pipe.device).manual_seed(42),
            output_type="pil" if dist.get_rank() == 0 else "pt",
        ).frames[0]
        dist.barrier()
        step_parallelism.clear_cache()


    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.cuda.synchronize()
    start = time.time()

    prompt = args.prompt
    video = pipe(
        prompt=prompt,
        num_inference_steps=50,
        height=args.height,
        width=args.width,
        video_length=args.video_length,
        guidance_scale=6,
        # generator=torch.Generator(device=pipe.device).manual_seed(42),
        output_type="pil" if dist.get_rank() == 0 else "pt",
        generator = torch.Generator("cuda").manual_seed(args.seed)
    ).frames[0]

    torch.cuda.synchronize()
    end = time.time()
    print(f"device{_rank} Time taken: {end-start:.2f} seconds.")

    if step_parallelism._rank == 0:
        print("Saving video to latte_parallel.mp4")
        export_to_video(video, f"latte_stepparallelism_world{_world_size}_warm{warm_up}.mp4", fps=8)
    dist.destroy_process_group()