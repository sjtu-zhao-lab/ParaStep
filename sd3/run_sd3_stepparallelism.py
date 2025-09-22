import torch
import torch.distributed as dist
import time
import argparse
import os

from sd3.stepParallelism_sd3 import StableDiffusion3Pipeline, StepParallelism

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='stabilityai/stable-diffusion-3-medium-diffusers') 
    #"A cute penguin is watching TV" "A cute dog is watching TV" "A cool boy is playing computer"
    parser.add_argument("--prompt", type=str, default='A cat holding a sign that says hello world')
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--reduce_memory_level", type=int, default=0, help="0: no memory optimization; 1: offload t5 to cpu, and load t5 to gpu when it is needed; 2: split T5encoder to _world_size stages, each stage belong to single device.")

    args = parser.parse_args()


    pipeline = StableDiffusion3Pipeline.from_pretrained(args.model, torch_dtype=torch.float16, low_cpu_mem_usage=True, )
    step_parallelism = StepParallelism(pipeline, step_size=int(os.environ["WORLD_SIZE"]), warmp_up=args.warmup, reduce_memory_level=args.reduce_memory_level)

    # warm up
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    image = pipeline(
        args.prompt,
        negative_prompt="",
        num_inference_steps=10,
        guidance_scale=7.0,
        height = 1440,
        width = 1440
    ).images[0]

    step_parallelism.clear_cache()    

    # inference
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # step_parallelism.reset_state(warm_up=args.warm_up)
    torch.cuda.synchronize()
    start = time.time()
    image = pipeline(
        args.prompt,
        negative_prompt="",
        num_inference_steps=args.num_inference_steps,
        guidance_scale=7.0,
        height = 1440,
        width = 1440,
        generator = torch.Generator("cuda").manual_seed(args.seed)
    ).images[0]
    torch.cuda.synchronize()
    print(f"device{step_parallelism._rank} Time taken: {time.time()-start:.2f} seconds.")
    step_parallelism.clear_cache() 

    if step_parallelism._rank == 0:
        image.save("sd3_stepparallelism_device_{}.png".format(int(os.environ["WORLD_SIZE"])))
    
    dist.destroy_process_group()