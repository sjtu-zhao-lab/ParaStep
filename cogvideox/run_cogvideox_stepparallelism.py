import torch
import torch.distributed as dist
from diffusers import CogVideoXPipeline
from cogvideox.stepParallelism_cogvideox import StepParallelism
from diffusers.utils import export_to_video
import argparse
import os
import time

from cogvideox.cogvideox_stepParallelism_call import CogVideoX__call


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance.")
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--warm_up", type=int, default=13)
    parser.add_argument("--reduce_memory", type=int, default=0)
    parser.add_argument("--warm_gpu", type=str, default="True")

    args = parser.parse_args()

    _world_size = int(os.environ["WORLD_SIZE"])
    _rank = int(os.environ["RANK"])

    warm_gpu = True if args.warm_gpu == "True" else False


    pipe = CogVideoXPipeline.from_pretrained(
        "THUDM/CogVideoX-2b",
        torch_dtype=torch.bfloat16,
        
    )

    num_frames = 45# 45->12, 49->13, 21->6  [(num_frames - 1) // self.vae_scale_factor_temporal + 1], vae_=4
    # num_frames_in_hidden_size = (num_frames-1)//4 + 1

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
            num_videos_per_prompt=1,
            num_inference_steps=10,
            num_frames=num_frames,
            guidance_scale=6,
            # generator=torch.Generator(device=pipe.device).manual_seed(42),
            output_type="pil" if dist.get_rank() == 0 else "pt",
            generator = torch.Generator("cuda").manual_seed(args.seed)
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
        num_videos_per_prompt=1,
        num_inference_steps=50,
        num_frames=num_frames,
        guidance_scale=6,
        # generator=torch.Generator(device=pipe.device).manual_seed(42),
        output_type="pil" if dist.get_rank() == 0 else "pt",
        generator = torch.Generator("cuda").manual_seed(args.seed)
    ).frames[0]

    torch.cuda.synchronize()
    end = time.time()
    print(f"device{_rank} Time taken: {end-start:.2f} seconds.")

    if step_parallelism._rank == 0:
        print("Saving video to cogvideox_parallel.mp4")
        export_to_video(video, f"cogvideox_stepparallelism_world{_world_size}_warm{warm_up}.mp4", fps=8)
    dist.destroy_process_group()
