"""
PYTHONPATH=$PWD:$PYTHONPATH CUDA_VISIBLE_DEVICES=2,3 save_dir="xxx" vbench_list_path="xxx/VBench_test_info.json" torchrun --nproc_per_node=2 baseline/xDiT/eval_xdit.py --model "maxin-cn/Latte-1" --num_inference_steps 50 --seed 20 --prompt "A dog wearing sunglasses floating in space, surreal, nebulae in background." --height 512 --width 512  --ring_degree 2 --warmup_steps 5
"""

import time
import torch
import torch.distributed
from diffusers import AutoencoderKLTemporalDecoder
from xfuser import xFuserLattePipeline, xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_runtime_state,
    is_dp_last_group,
)
import imageio
import os
import json
from tqdm import tqdm
from diffusers.utils import export_to_video
import torch.distributed as dist

def read_prompt_list(prompt_list_path):
    with open(prompt_list_path, "r") as f:
        prompt_list = json.load(f)
    prompt_list = [prompt["prompt_en"] for prompt in prompt_list]
    return prompt_list


def main():
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    local_rank = get_world_group().local_rank

    pipe = xFuserLattePipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        engine_config=engine_config,
        torch_dtype=torch.float16,
    ).to(f"cuda:{local_rank}")
    # pipe.latte_prepare_run(input_config)

    vbench_list_path = os.environ['vbench_list_path']
    save_dir = os.environ['save_dir']
    outputpath = f"ring{engine_args.ring_degree}"
    save_dir = os.path.join(save_dir, outputpath)

    os.makedirs(save_dir, exist_ok=True)
    latency_path = os.path.join(save_dir, "0latency.txt")

    prompt_list = read_prompt_list(vbench_list_path)


    # vae = AutoencoderKLTemporalDecoder.from_pretrained(
    #     engine_config.model_config.model,
    #     subfolder="vae_temporal_decoder",
    #     torch_dtype=torch.float16,
    # ).to(f"cuda:{local_rank}")
    # pipe.vae = vae

    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    times = []
    for prompt in tqdm(prompt_list):
        for l in range(5):
            # prompt = args.prompt
            if l == 0:
                torch.cuda.synchronize()
                start_time = time.time()
                output = pipe(
                    height=input_config.height,
                    width=input_config.width,
                    video_length=16,
                    prompt=prompt,
                    num_inference_steps=input_config.num_inference_steps,
                    output_type="pil",
                    guidance_scale=6,
                    generator = torch.Generator("cuda").manual_seed(args.seed)
                )
                torch.cuda.synchronize()
                end_time = time.time()
                elapsed_time = end_time - start_time
                times.append(elapsed_time)

                # dist.barrier()
            

            # if is_dp_last_group():
            if output != None:
                save_path = os.path.join(save_dir, f"{prompt}-{l}.mp4")
                export_to_video(output.frames[0], save_path, fps=8)
    latency = sum(times)/len(times)

    print(f"latency={latency}")
    # if is_dp_last_group():
    if output != None:
        with open(latency_path,"w") as f:
            latency = sum(times)/len(times)
            f.write(f"mean latency is {latency}")

    get_runtime_state().destory_distributed_env()


if __name__ == "__main__":
    main()
