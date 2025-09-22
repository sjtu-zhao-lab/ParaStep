from typing import Any, Dict, Optional, Tuple, Union, List, Callable
import torch

import torch.distributed as dist

import torch

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.pipelines.latte.pipeline_latte import retrieve_timesteps
from diffusers.utils import BaseOutput

from dataclasses import dataclass



import os

@dataclass
class LattePipelineOutput(BaseOutput):
    frames: torch.Tensor

@torch.no_grad()
def latte__call(
    self,
    prompt: Union[str, List[str]] = None,
    negative_prompt: str = "",
    num_inference_steps: int = 50,
    timesteps: Optional[List[int]] = None,
    guidance_scale: float = 7.5,
    num_images_per_prompt: int = 1,
    video_length: int = 16,
    height: int = 512,
    width: int = 512,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: str = "pil",
    return_dict: bool = True,
    callback_on_step_end: Optional[
        Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
    ] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    clean_caption: bool = True,
    mask_feature: bool = True,
    enable_temporal_attentions: bool = True,
    decode_chunk_size: Optional[int] = None,
) -> Union[LattePipelineOutput, Tuple]:
    if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
        callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

    # 0. Default
    decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else video_length

    # 1. Check inputs. Raise error if not correct
    height = height or self.transformer.config.sample_size * self.vae_scale_factor
    width = width or self.transformer.config.sample_size * self.vae_scale_factor
    self.check_inputs(
        prompt,
        height,
        width,
        negative_prompt,
        callback_on_step_end_tensor_inputs,
        prompt_embeds,
        negative_prompt_embeds,
    )
    self._guidance_scale = guidance_scale
    self._interrupt = False

    # 2. Default height and width to transformer
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0

    # 3. Encode input prompt
    prompt_embeds, negative_prompt_embeds = self.encode_prompt(
        prompt,
        do_classifier_free_guidance,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        clean_caption=clean_caption,
        mask_feature=mask_feature,
    )
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

    # 4. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
    self._num_timesteps = len(timesteps)

    # 5. Prepare latents.
    latent_channels = self.transformer.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        latent_channels,
        video_length,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    # 7. Denoising loop
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            current_timestep = t
            if not torch.is_tensor(current_timestep):
                # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                # This would be a good case for the `match` statement (Python 3.10+)
                is_mps = latent_model_input.device.type == "mps"
                if isinstance(current_timestep, float):
                    dtype = torch.float32 if is_mps else torch.float64
                else:
                    dtype = torch.int32 if is_mps else torch.int64
                current_timestep = torch.tensor([current_timestep], dtype=dtype, device=latent_model_input.device)
            elif len(current_timestep.shape) == 0:
                current_timestep = current_timestep[None].to(latent_model_input.device)
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            current_timestep = current_timestep.expand(latent_model_input.shape[0])

            #===StepParallelism===
            self.inference_step_num += 1
            if self.inference_step_num <= self.warm_up:
                noise_pred = self.transformer(
                    latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=current_timestep,
                    enable_temporal_attentions=enable_temporal_attentions,
                    return_dict=False,
                )[0].contiguous()
                self.noise_pred_cache = noise_pred.clone()
            else:# warmup结束了
                if self.step_rank == 0:
                    if self.round == 0:
                        noise_pred = self.transformer(
                            latent_model_input,
                            encoder_hidden_states=prompt_embeds,
                            timestep=current_timestep,
                            enable_temporal_attentions=enable_temporal_attentions,
                            return_dict=False,
                        )[0].contiguous()
                        self.noise_pred_cache = noise_pred.clone()
                    else:
                        
                        noise_pred = self.noise_pred_cache.clone()
                        dist.recv(noise_pred, self.round, self.step_mesh.get_group())
                        self.noise_pred_cache = noise_pred.clone()
                else:
                    if self.step_rank == self.round:
                        noise_pred = self.transformer(
                            latent_model_input,
                            encoder_hidden_states=prompt_embeds,
                            timestep=current_timestep,
                            enable_temporal_attentions=enable_temporal_attentions,
                            return_dict=False,
                        )[0].contiguous()
                        self.noise_pred_cache = noise_pred.clone()
                        dist.send(noise_pred, 0, self.step_mesh.get_group())
                    else:
                        noise_pred = self.noise_pred_cache.clone()


            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # use learned sigma?
            if not (
                hasattr(self.scheduler.config, "variance_type")
                and self.scheduler.config.variance_type in ["learned", "learned_range"]
            ):
                noise_pred = noise_pred.chunk(2, dim=1)[0]

            # compute previous video: x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

            #===StepParallelism===
            if (self.inference_step_num <= self.warm_up):
                self.latents_cache = latents.clone()

            if self.inference_step_num > self.warm_up: 
                if self.round == self.step_size - 1:
                    dist.broadcast(latents, 0, self.step_mesh.get_group())
                    # dist.broadcast(self.noise_pred_cache, 0, self.step_mesh.get_group())
                    self.latents_cache = latents.clone()

                self.round += 1
                self.round = self.round % self.step_size   
                


            # call the callback, if provided
            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()

    if not output_type == "latents":
        video = self.decode_latents(latents, video_length, decode_chunk_size=14)
        video = self.video_processor.postprocess_video(video=video, output_type=output_type)
    else:
        video = latents

    # Offload all models
    self.maybe_free_model_hooks()

    if not return_dict:
        return (video,)

    return LattePipelineOutput(frames=video)


