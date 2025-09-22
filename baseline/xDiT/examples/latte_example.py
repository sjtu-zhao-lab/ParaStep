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
from diffusers.utils import export_to_video


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

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    output = pipe(
        height=input_config.height,
        width=input_config.width,
        video_length=16,
        prompt=input_config.prompt,
        num_inference_steps=input_config.num_inference_steps,
        guidance_scale=6,
        # output_type="pt"
        output_type="pil" if is_dp_last_group() else "pt"
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    parallel_info = (
        f"dp{engine_args.data_parallel_degree}_cfg{engine_config.parallel_config.cfg_degree}_"
        f"ulysses{engine_args.ulysses_degree}_ring{engine_args.ring_degree}_"
        f"pp{engine_args.pipefusion_parallel_degree}_patch{engine_args.num_pipeline_patch}"
    )
    if is_dp_last_group():
        export_to_video(output.frames[0], f"latte.mp4", fps=8)
        # videos = output.frames.cpu()
        # global_rank = get_world_group().rank
        # dp_group_world_size = get_data_parallel_world_size()
        # dp_group_index = global_rank // dp_group_world_size
        # num_dp_groups = engine_config.parallel_config.dp_degree
        # dp_batch_size = (input_config.batch_size + num_dp_groups - 1) // num_dp_groups
        # if input_config.num_frames > 1:
        #     videos = (videos.clamp(0, 1) * 255).to(
        #         dtype=torch.uint8
        #     )  # convert to uint8
        #     imageio.mimwrite(
        #         "./latte_output.mp4", videos[0].permute(0, 2, 3, 1), fps=8, quality=5
        #     )  # highest quality is 10, lowest is 0

    if get_world_group().rank == get_world_group().world_size - 1:
        print(f"epoch time: {elapsed_time:.2f} sec, memory: {peak_memory/1e9} GB")
    get_runtime_state().destory_distributed_env()


if __name__ == "__main__":
    main()
