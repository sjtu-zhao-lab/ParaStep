# from https://github.com/chengzeyi/ParaAttention/blob/main/examples/run_hunyuan_video.py
import functools
from typing import Any, Dict, Union, Optional
import logging
import time

import torch

from diffusers import DiffusionPipeline#, HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig, HunyuanVideoTransformer3DModel, HunyuanVideoPipeline
from transformers.models.llama.modeling_llama import LlamaModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import scale_lora_layers, unscale_lora_layers, USE_PEFT_BACKEND
from diffusers.utils import export_to_video
import logging
import os
import types

from tqdm import tqdm
import json

def read_prompt_list(prompt_list_path):
    with open(prompt_list_path, "r") as f:
        prompt_list = json.load(f)
    prompt_list = [prompt["prompt_en"] for prompt in prompt_list]
    return prompt_list


from xfuser import xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    get_data_parallel_world_size,
    get_data_parallel_rank,
    get_runtime_state,
    get_classifier_free_guidance_world_size,
    get_classifier_free_guidance_rank,
    get_cfg_group,
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    get_sp_group,
    is_dp_last_group,
    initialize_runtime_state,
    get_pipeline_parallel_world_size,
)

from xfuser.model_executor.layers.attention_processor import xFuserHunyuanVideoAttnProcessor2_0

assert xFuserHunyuanVideoAttnProcessor2_0 is not None

import numpy as np
import torch.distributed as dist


def parallelize_transformer(pipe: DiffusionPipeline):
    transformer = pipe.transformer

    @functools.wraps(transformer.__class__.forward)
    def new_forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        pooled_projections: torch.Tensor,
        guidance: torch.Tensor = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logging.warning("Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective.")

        batch_size, num_channels, num_frames, height, width = hidden_states.shape

        assert batch_size % get_classifier_free_guidance_world_size(
        ) == 0, f"Cannot split dim 0 of hidden_states ({batch_size}) into {get_classifier_free_guidance_world_size()} parts."

        p, p_t = self.config.patch_size, self.config.patch_size_t
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p
        post_patch_width = width // p

        # 1. RoPE
        image_rotary_emb = self.rope(hidden_states)

        # 2. Conditional embeddings
        temb = self.time_text_embed(timestep, guidance, pooled_projections)
        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states,
                                                      timestep,
                                                      encoder_attention_mask)

        hidden_states = hidden_states.reshape(batch_size, post_patch_num_frames, post_patch_height, post_patch_width, -1)
        hidden_states = hidden_states.flatten(1, 3)

        hidden_states = torch.chunk(hidden_states,
                                    get_classifier_free_guidance_world_size(),
                                    dim=0)[get_classifier_free_guidance_rank()]
        hidden_states = torch.chunk(hidden_states,
                                    get_sequence_parallel_world_size(),
                                    dim=-2)[get_sequence_parallel_rank()]

        encoder_attention_mask = encoder_attention_mask[0].to(torch.bool)
        encoder_hidden_states_indices = torch.arange(
            encoder_hidden_states.shape[1],
            device=encoder_hidden_states.device)
        encoder_hidden_states_indices = encoder_hidden_states_indices[
            encoder_attention_mask]
        encoder_hidden_states = encoder_hidden_states[
            ..., encoder_hidden_states_indices, :]
        if encoder_hidden_states.shape[-2] % get_sequence_parallel_world_size(
        ) != 0:
            get_runtime_state().split_text_embed_in_sp = False
        else:
            get_runtime_state().split_text_embed_in_sp = True

        encoder_hidden_states = torch.chunk(
            encoder_hidden_states,
            get_classifier_free_guidance_world_size(),
            dim=0)[get_classifier_free_guidance_rank()]
        if get_runtime_state().split_text_embed_in_sp:
            encoder_hidden_states = torch.chunk(
                encoder_hidden_states,
                get_sequence_parallel_world_size(),
                dim=-2)[get_sequence_parallel_rank()]

        freqs_cos, freqs_sin = image_rotary_emb

        def get_rotary_emb_chunk(freqs):
            freqs = torch.chunk(freqs, get_sequence_parallel_world_size(), dim=0)[get_sequence_parallel_rank()]
            return freqs

        freqs_cos = get_rotary_emb_chunk(freqs_cos)
        freqs_sin = get_rotary_emb_chunk(freqs_sin)
        image_rotary_emb = (freqs_cos, freqs_sin)

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:

            def create_custom_forward(module, return_dict=None):

                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False}

            for block in self.transformer_blocks:
                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    None,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )

            for block in self.single_transformer_blocks:
                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    None,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )

        else:
            for block in self.transformer_blocks:
                hidden_states, encoder_hidden_states = block(
                    hidden_states, encoder_hidden_states, temb, None,
                    image_rotary_emb)

            for block in self.single_transformer_blocks:
                hidden_states, encoder_hidden_states = block(
                    hidden_states, encoder_hidden_states, temb, None,
                    image_rotary_emb)

        # 5. Output projection
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = get_sp_group().all_gather(hidden_states, dim=-2)
        hidden_states = get_cfg_group().all_gather(hidden_states, dim=0)

        hidden_states = hidden_states.reshape(batch_size,
                                              post_patch_num_frames,
                                              post_patch_height,
                                              post_patch_width, -1, p_t, p, p)

        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (hidden_states, )

        return Transformer2DModelOutput(sample=hidden_states)

    new_forward = new_forward.__get__(transformer)
    transformer.forward = new_forward

    # for block in transformer.transformer_blocks + transformer.single_transformer_blocks:
    #     block.attn.processor = xFuserHunyuanVideoAttnProcessor2_0()



def main():
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)

    engine_config, input_config = engine_args.create_config()
    local_rank = get_world_group().local_rank

    assert engine_args.pipefusion_parallel_degree == 1, "This script does not support PipeFusion."
    assert engine_args.use_parallel_vae is False, "parallel VAE not implemented for HunyuanVideo"

    vbench_list_path = os.environ['vbench_list_path']
    prompt_list = read_prompt_list(vbench_list_path)
    save_dir = os.environ['save_dir']
    _world_size = int(os.environ["WORLD_SIZE"])
    save_dir = os.path.join(save_dir, f"teacache_xdit{_world_size}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"The generated videos will be saved into {save_dir}")

    # transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    #     pretrained_model_name_or_path=engine_config.model_config.model,
    #     subfolder="transformer",
    #     torch_dtype=torch.bfloat16,
    #     revision="refs/pr/18",
    # )
    # pipe = HunyuanVideoPipeline.from_pretrained(
    #     pretrained_model_name_or_path=engine_config.model_config.model,
    #     transformer=transformer,
    #     torch_dtype=torch.float16,
    #     revision="refs/pr/18",
    # )

    _world_size = int(os.environ["WORLD_SIZE"])
    _rank = int(os.environ["RANK"])

    torch.cuda.set_device(f"cuda:{_rank}")

    quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)
    transformer_8bit = HunyuanVideoTransformer3DModel.from_pretrained(
        "tencent/HunyuanVideo",
        subfolder="transformer",
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        revision="refs/pr/18",
    )

    # transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    #     "tencent/HunyuanVideo",
    #     subfolder="transformer",
    #     torch_dtype=torch.bfloat16,
    # )
    # .to(device)

    text_encoder_8bit = LlamaModel.from_pretrained(
        "tencent/HunyuanVideo",
        subfolder="text_encoder",
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        revision="refs/pr/18",
    )


    pipe = HunyuanVideoPipeline.from_pretrained(
        "tencent/HunyuanVideo",
        transformer=transformer_8bit,
        text_encoder = text_encoder_8bit,
        torch_dtype=torch.float16,
        revision="refs/pr/18",
        # device_map="balanced",
    )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    initialize_runtime_state(pipe, engine_config)
    get_runtime_state().set_video_input_parameters(
        height=input_config.height,
        width=input_config.width,
        num_frames=input_config.num_frames,
        batch_size=1,
        num_inference_steps=input_config.num_inference_steps,
        split_text_embed_in_sp=get_pipeline_parallel_world_size() == 1,
    )
    

    parallelize_transformer(pipe)
    
    if args.enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload(gpu_id=local_rank)
        logging.info(f"rank {local_rank} sequential CPU offload enabled")
    elif args.enable_model_cpu_offload:
        pipe.enable_model_cpu_offload(gpu_id=local_rank)
        logging.info(f"rank {local_rank} model CPU offload enabled")
    else:
        device = torch.device(f"cuda:{local_rank}")
        pipe = pipe.to(device)
    
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    # pipe.enable_model_cpu_offload()



    parameter_peak_memory = torch.cuda.max_memory_allocated(
        device=f"cuda:{local_rank}")


    # torch.cuda.reset_peak_memory_stats()
    # start_time = time.time()
    


    # 替换 time_text_embed 的 forward（如果它是一个 nn.Module）
    # pipe.transformer.time_text_embed.forward = types.MethodType(timestep_forward, pipe.transformer.time_text_embed)


    print(f"-----height={input_config.height}, width={input_config.width}, num_frames={input_config.num_frames}-----")
    
    times=[]
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    for prompt in tqdm(prompt_list):
        print(f"prompt={prompt}")
        input_config.prompt = prompt
        print(f"input_config.prompt is {input_config.prompt}")
        for l in range(5):
            pipe.transformer.__class__.cnt = 0
            # pipe.transformer.__class__.num_steps = input_config.num_inference_steps
            # pipe.transformer.__class__.rel_l1_thresh = 0.1 # 0.1 for 1.6x speedup, 0.15 for 2.1x speedup
            pipe.transformer.__class__.accumulated_rel_l1_distance = 0
            pipe.transformer.__class__.previous_modulated_input = None
            pipe.transformer.__class__.previous_residual = None

            if l == 0:
                torch.cuda.synchronize()
                start_time = time.time()

                output = pipe(
                    height=input_config.height,
                    width=input_config.width,
                    num_frames=input_config.num_frames,
                    prompt=input_config.prompt,
                    num_inference_steps=input_config.num_inference_steps,
                    guidance_scale=6.0,
                    generator = torch.Generator("cuda").manual_seed(args.seed)
                ).frames[0]

                end_time = time.time()
                elapsed_time = end_time - start_time

                torch.cuda.synchronize()
                end_time = time.time()
                elapsed_time = end_time - start_time
                times.append(elapsed_time)
                dist.barrier()

            if is_dp_last_group():
                save_path = os.path.join(save_dir, f"{prompt}-{l}.mp4")
                export_to_video(output, save_path, fps=4)

    latency_path = os.path.join(save_dir, "0latency.txt")
    if is_dp_last_group():
        with open(latency_path,"w") as f:
            latency = sum(times)/len(times)
            f.write(f"mean latency is {latency}")

    get_runtime_state().destory_distributed_env()


# mkdir -p results && torchrun --nproc_per_node=2 examples/hunyuan_video_usp_example.py --model tencent/HunyuanVideo --ulysses_degree 2 --num_inference_steps 30 --warmup_steps 0 --prompt "A cat walks on the grass, realistic" --height 320 --width 512 --num_frames 61 --enable_tiling --enable_model_cpu_offload
# mkdir -p results && torchrun --nproc_per_node=2 examples/hunyuan_video_usp_example.py --model tencent/HunyuanVideo --ulysses_degree 2 --num_inference_steps 30 --warmup_steps 0 --prompt "A cat walks on the grass, realistic" --height 544 --width 960 --num_frames 129 --enable_tiling --enable_model_cpu_offload
if __name__ == "__main__":
    main()
