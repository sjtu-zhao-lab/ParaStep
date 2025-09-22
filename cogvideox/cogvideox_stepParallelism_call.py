from typing import Any, Dict, Optional, Tuple, Union, List, Callable
import torch

import torch.distributed as dist

import torch

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.schedulers import CogVideoXDPMScheduler
from diffusers.pipelines.cogvideo.pipeline_output import CogVideoXPipelineOutput
import math
from diffusers.pipelines.cogvideo.pipeline_cogvideox import retrieve_timesteps

from transformers.utils import (
    is_torchdynamo_compiling,
)
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions


# CogVideoXpipelinecall
@torch.no_grad()
def CogVideoX__call(
    self,
    prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_frames: Optional[int] = None,
    num_inference_steps: int = 50,
    timesteps: Optional[List[int]] = None,
    guidance_scale: float = 6,
    use_dynamic_cfg: bool = False,
    num_videos_per_prompt: int = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: str = "pil",
    return_dict: bool = True,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end: Optional[
        Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
    ] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 226,
) -> Union[CogVideoXPipelineOutput, Tuple]:
    # print("hello world")
    """
    Function invoked when calling the pipeline for generation.
    """

    if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
        callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs
    
    if self.reduce_memory == 1:
        if self.text_encoder.device != self.transformer.device:
            self.text_encoder = self.text_encoder.to(self.transformer.device)
            dist.barrier()

    height = height or self.transformer.config.sample_height * self.vae_scale_factor_spatial
    width = width or self.transformer.config.sample_width * self.vae_scale_factor_spatial
    num_frames = num_frames or self.transformer.config.sample_frames
    # print("height=={}, width=={}".format(height, width))

    num_videos_per_prompt = 1

    # 1. Check inputs. Raise error if not correct
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
    self._attention_kwargs = attention_kwargs
    self._interrupt = False

    # 2. Default call parameters
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
        negative_prompt,
        do_classifier_free_guidance,
        num_videos_per_prompt=num_videos_per_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        max_sequence_length=max_sequence_length,
        device=device,
    )
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    
    if self.reduce_memory == 1:
        self.text_encoder = self.text_encoder.to('cpu')
        torch.cuda.empty_cache()

    # 4. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
    self._num_timesteps = len(timesteps)

    # 5. Prepare latents
    latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

    # For CogVideoX 1.5, the latent frames should be padded to make it divisible by patch_size_t
    patch_size_t = self.transformer.config.patch_size_t
    additional_frames = 0
    if patch_size_t is not None and latent_frames % patch_size_t != 0:
        additional_frames = patch_size_t - latent_frames % patch_size_t
        num_frames += additional_frames * self.vae_scale_factor_temporal

    latent_channels = self.transformer.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_videos_per_prompt,
        latent_channels,
        num_frames,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    # 7. Create rotary embeds if required
    image_rotary_emb = (
        self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
        if self.transformer.config.use_rotary_positional_embeddings
        else None
    )

    # 8. Denoising loop
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

    with self.progress_bar(total=num_inference_steps) as progress_bar:
        # for DPM-solver++
        old_pred_original_sample = None
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latent_model_input.shape[0])

            self.inference_step_num += 1
            if self.inference_step_num <= self.warm_up:
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    image_rotary_emb=image_rotary_emb,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]
                self.noise_pred_cache = noise_pred.clone()
            else:# warmup结束了
                if self.step_rank == 0:
                    if self.round == 0:
                        noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            encoder_hidden_states=prompt_embeds,
                            timestep=timestep,
                            image_rotary_emb=image_rotary_emb,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]
                        self.noise_pred_cache = noise_pred.clone()
                    else:
                        noise_pred = self.noise_pred_cache.clone()
                        dist.recv(noise_pred, self.round, self.step_mesh.get_group())
                        self.noise_pred_cache = noise_pred.clone()
                else:
                    if self.step_rank == self.round:
                        noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            encoder_hidden_states=prompt_embeds,
                            timestep=timestep,
                            image_rotary_emb=image_rotary_emb,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]
                        self.noise_pred_cache = noise_pred.clone()
                        dist.send(noise_pred, 0, self.step_mesh.get_group())
                    else:
                        noise_pred = self.noise_pred_cache.clone()


            noise_pred = noise_pred.float()

            # perform guidance
            if use_dynamic_cfg:
                self._guidance_scale = 1 + guidance_scale * (
                    (1 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
                )
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
            else:
                latents, old_pred_original_sample = self.scheduler.step(
                    noise_pred,
                    old_pred_original_sample,
                    t,
                    timesteps[i - 1] if i > 0 else None,
                    latents,
                    **extra_step_kwargs,
                    return_dict=False,
                )
            if (self.inference_step_num <= self.warm_up):
                self.latents_cache = latents.clone()

            if self.inference_step_num > self.warm_up:

                if self.round == self.step_size - 1:
                    dist.broadcast(latents, 0, self.step_mesh.get_group())
                    # dist.broadcast(self.noise_pred_cache, 0, self.step_mesh.get_group())
                    self.latents_cache = latents.clone()

                self.round += 1
                self.round = self.round % self.step_size   
                
            


            latents = latents.to(prompt_embeds.dtype)

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

    if not output_type == "latent":
        # Discard any padding frames that were added for CogVideoX 1.5
        latents = latents[:, additional_frames:]
        video = self.decode_latents(latents)
        video = self.video_processor.postprocess_video(video=video, output_type=output_type)
    else:
        video = latents

    # Offload all models
    self.maybe_free_model_hooks()

    if not return_dict:
        return (video,)

    return CogVideoXPipelineOutput(frames=video)

def T5stack_call(
    self,
    input_ids=None,
    attention_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    inputs_embeds=None,
    head_mask=None,
    cross_attn_head_mask=None,
    past_key_values=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    cache_position=None,
):
    # Model parallel
    if self.model_parallel:
        torch.cuda.set_device(self.first_device)
        self.embed_tokens = self.embed_tokens.to(self.first_device)
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is not None and inputs_embeds is not None:
        err_msg_prefix = "decoder_" if self.is_decoder else ""
        raise ValueError(
            f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
        )
    elif input_ids is not None:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
    else:
        err_msg_prefix = "decoder_" if self.is_decoder else ""
        raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    if inputs_embeds is None:
        if self.embed_tokens is None:
            raise ValueError("You have to initialize the model with valid token embeddings")
        inputs_embeds = self.embed_tokens(input_ids)

    batch_size, seq_length = input_shape

    if use_cache is True:
        if not self.is_decoder:
            raise ValueError(f"`use_cache` can only be set to `True` if {self} is used as a decoder")

    # initialize past_key_values
    return_legacy_cache = False
    return_self_attention_cache = False
    if self.is_decoder and (use_cache or past_key_values is not None):
        if isinstance(past_key_values, Cache) and not isinstance(past_key_values, EncoderDecoderCache):
            return_self_attention_cache = True
            past_key_values = EncoderDecoderCache(past_key_values, DynamicCache())
        elif not isinstance(past_key_values, EncoderDecoderCache):
            return_legacy_cache = True
            logger.warning_once(
                "Passing a tuple of `past_key_values` is deprecated and will be removed in Transformers v4.48.0. "
                "You should pass an instance of `EncoderDecoderCache` instead, e.g. "
                "`past_key_values=EncoderDecoderCache.from_legacy_cache(past_key_values)`."
            )
            past_key_values = EncoderDecoderCache.from_legacy_cache(past_key_values)
        elif past_key_values is None:
            past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())
    elif not self.is_decoder:
        # do not pass cache object down the line for encoder stack
        # it messes indexing later in decoder-stack because cache object is modified in-place
        past_key_values = None

    past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0
    if cache_position is None:
        cache_position = torch.arange(
            past_key_values_length, past_key_values_length + seq_length, device=inputs_embeds.device
        )

    if attention_mask is None and not is_torchdynamo_compiling():
        # required mask seq length can be calculated via length of past cache
        mask_seq_length = past_key_values_length + seq_length
        attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)

    if self.config.is_decoder:
        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values.self_attention_cache if past_key_values is not None else None,
            output_attentions,
        )
    elif attention_mask is not None:
        causal_mask = attention_mask[:, None, None, :]
        causal_mask = causal_mask.to(dtype=inputs_embeds.dtype)
        causal_mask = (1.0 - causal_mask) * torch.finfo(inputs_embeds.dtype).min
    else:
        causal_mask = None

    # If a 2D or 3D attention mask is provided for the cross-attention
    # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
    if self.is_decoder and encoder_hidden_states is not None:
        encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(
                encoder_hidden_shape, device=inputs_embeds.device, dtype=torch.long
            )
        encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
    else:
        encoder_extended_attention_mask = None

    # Prepare head mask if needed
    head_mask = self.get_head_mask(head_mask, self.config.num_layers)
    cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
    all_hidden_states = () if output_hidden_states else None
    all_attentions = () if output_attentions else None
    all_cross_attentions = () if (output_attentions and self.is_decoder) else None
    position_bias = None
    encoder_decoder_position_bias = None

    hidden_states = self.dropout(inputs_embeds)
    # print(f"len of block is {len(self.block)}")
    block_num = len(self.block)
    for i, layer_module in enumerate(self.block):
        local = False
        if (i in self.local_block) or (i == 0):
            local = True
        layer_head_mask = head_mask[i]
        cross_attn_layer_head_mask = cross_attn_head_mask[i]
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(hidden_states.device)
            # Ensure that attention_mask is always on the same device as hidden_states
            if causal_mask is not None:
                causal_mask = causal_mask.to(hidden_states.device)
            if position_bias is not None:
                position_bias = position_bias.to(hidden_states.device)
            if encoder_hidden_states is not None:
                encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
            if encoder_extended_attention_mask is not None:
                encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
            if encoder_decoder_position_bias is not None:
                encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
            if layer_head_mask is not None:
                layer_head_mask = layer_head_mask.to(hidden_states.device)
            if cross_attn_layer_head_mask is not None:
                cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                layer_module.forward,
                hidden_states,
                causal_mask,
                position_bias,
                encoder_hidden_states,
                encoder_extended_attention_mask,
                encoder_decoder_position_bias,
                layer_head_mask,
                cross_attn_layer_head_mask,
                None,  # past_key_value is always None with gradient checkpointing
                use_cache,
                output_attentions,
                return_dict,
                cache_position,
            )
        else:
            not_picker = True
            if local == True: 
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    return_dict=return_dict,
                    cache_position=cache_position,
                )
                layer_outputs = (layer_outputs[0].contiguous(), layer_outputs[1].contiguous(),)
                if i == 0:
                    if not_picker == False:
                        pass
                    else:
                        layer_outputs_1_buffer = layer_outputs[0].clone()
                        layer_outputs_2_buffer = layer_outputs[1].clone()
                else:
                    if i == block_num - 1:
                        if not_picker == False:
                            pass
                        else:
                            assert self.step_rank == self.step_size -1
                            dist.broadcast(layer_outputs[0], int(self.step_mesh.mesh[(self.step_rank)]), self.step_mesh.get_group())
                            dist.broadcast(layer_outputs[1], int(self.step_mesh.mesh[(self.step_rank)]), self.step_mesh.get_group())
                    elif i == self.local_block[-1]:
                        if not_picker == False:
                            pass
                        else:
                            dist.send(layer_outputs[0], int(self.step_mesh.mesh[(self.step_rank+1)]), self.step_mesh.get_group())
                            dist.send(layer_outputs[1], int(self.step_mesh.mesh[(self.step_rank+1)]), self.step_mesh.get_group())
            else:
                if i == block_num - 1:
                    if not_picker == False:
                        pass
                    else:
                        dist.broadcast(layer_outputs_1_buffer, int(self.step_mesh.mesh[(self.step_size - 1)]), self.step_mesh.get_group())
                        dist.broadcast(layer_outputs_2_buffer, int(self.step_mesh.mesh[(self.step_size - 1)]), self.step_mesh.get_group())
                        layer_outputs = (layer_outputs_1_buffer.clone(), layer_outputs_2_buffer.clone())
                elif i+1 == self.local_block[0]:
                    if not_picker == False:
                        pass
                    else:
                        dist.recv(layer_outputs_1_buffer, int(self.step_mesh.mesh[(self.step_rank-1)]), self.step_mesh.get_group())
                        dist.recv(layer_outputs_2_buffer, int(self.step_mesh.mesh[(self.step_rank-1)]), self.step_mesh.get_group())
                        layer_outputs = (layer_outputs_1_buffer.clone(), layer_outputs_2_buffer.clone())
                else:
                    continue#skip

        # layer_outputs is a tuple with:
        # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
        if use_cache is False:
            layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

        hidden_states, next_decoder_cache = layer_outputs[:2]

        # We share the position biases between the layers - the first layer store them
        # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
        # (cross-attention position bias), (cross-attention weights)
        position_bias = layer_outputs[2]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]

        if output_attentions:
            all_attentions = all_attentions + (layer_outputs[3],)
            if self.is_decoder:
                all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

        # Model Parallel: If it's the last layer for that device, put things on the next device
        if self.model_parallel:
            for k, v in self.device_map.items():
                if i == v[-1] and "cuda:" + str(k) != self.last_device:
                    hidden_states = hidden_states.to("cuda:" + str(k + 1))


    hidden_states = self.final_layer_norm(hidden_states)
    hidden_states = self.dropout(hidden_states)

    # Add last layer
    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if return_self_attention_cache:
        next_cache = past_key_values.self_attention_cache
    if return_legacy_cache:
        next_cache = past_key_values.to_legacy_cache()

    if not return_dict:
        return tuple(
            v
            for v in [
                hidden_states,
                next_cache,
                all_hidden_states,
                all_attentions,
                all_cross_attentions,
            ]
            if v is not None
        )
    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_attentions,
        cross_attentions=all_cross_attentions,
    )





