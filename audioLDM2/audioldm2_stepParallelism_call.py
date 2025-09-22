import torch
from diffusers.utils import (
    replace_example_docstring,
)
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
from diffusers.pipelines.pipeline_utils import AudioPipelineOutput

import torch.distributed as dist
import os

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import scipy
        >>> import torch
        >>> from diffusers import AudioLDM2Pipeline

        >>> repo_id = "cvssp/audioldm2"
        >>> pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
        >>> pipe = pipe.to("cuda")

        >>> # define the prompts
        >>> prompt = "The sound of a hammer hitting a wooden surface."
        >>> negative_prompt = "Low quality."

        >>> # set the seed for generator
        >>> generator = torch.Generator("cuda").manual_seed(0)

        >>> # run the generation
        >>> audio = pipe(
        ...     prompt,
        ...     negative_prompt=negative_prompt,
        ...     num_inference_steps=200,
        ...     audio_length_in_s=10.0,
        ...     num_waveforms_per_prompt=3,
        ...     generator=generator,
        ... ).audios

        >>> # save the best audio sample (index 0) as a .wav file
        >>> scipy.io.wavfile.write("techno.wav", rate=16000, data=audio[0])
        ```
        ```
        #Using AudioLDM2 for Text To Speech
        >>> import scipy
        >>> import torch
        >>> from diffusers import AudioLDM2Pipeline

        >>> repo_id = "anhnct/audioldm2_gigaspeech"
        >>> pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
        >>> pipe = pipe.to("cuda")

        >>> # define the prompts
        >>> prompt = "A female reporter is speaking"
        >>> transcript = "wish you have a good day"

        >>> # set the seed for generator
        >>> generator = torch.Generator("cuda").manual_seed(0)

        >>> # run the generation
        >>> audio = pipe(
        ...     prompt,
        ...     transcription=transcript,
        ...     num_inference_steps=200,
        ...     audio_length_in_s=10.0,
        ...     num_waveforms_per_prompt=2,
        ...     generator=generator,
        ...     max_new_tokens=512,          #Must set max_new_tokens equa to 512 for TTS
        ... ).audios

        >>> # save the best audio sample (index 0) as a .wav file
        >>> scipy.io.wavfile.write("tts.wav", rate=16000, data=audio[0])
        ```
"""

@torch.no_grad()
@replace_example_docstring(EXAMPLE_DOC_STRING)
def step_parallelism_call(
    self,
    prompt: Union[str, List[str]] = None,
    transcription: Union[str, List[str]] = None,
    audio_length_in_s: Optional[float] = None,
    num_inference_steps: int = 200,
    guidance_scale: float = 3.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_waveforms_per_prompt: Optional[int] = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.Tensor] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    generated_prompt_embeds: Optional[torch.Tensor] = None,
    negative_generated_prompt_embeds: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.LongTensor] = None,
    negative_attention_mask: Optional[torch.LongTensor] = None,
    max_new_tokens: Optional[int] = None,
    return_dict: bool = True,
    callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
    callback_steps: Optional[int] = 1,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    output_type: Optional[str] = "np",
):
    r"""
    The call function to the pipeline for generation.

    Args:
        prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts to guide audio generation. If not defined, you need to pass `prompt_embeds`.
        transcription (`str` or `List[str]`, *optional*):\
            The transcript for text to speech.
        audio_length_in_s (`int`, *optional*, defaults to 10.24):
            The length of the generated audio sample in seconds.
        num_inference_steps (`int`, *optional*, defaults to 200):
            The number of denoising steps. More denoising steps usually lead to a higher quality audio at the
            expense of slower inference.
        guidance_scale (`float`, *optional*, defaults to 3.5):
            A higher guidance scale value encourages the model to generate audio that is closely linked to the text
            `prompt` at the expense of lower sound quality. Guidance scale is enabled when `guidance_scale > 1`.
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts to guide what to not include in audio generation. If not defined, you need to
            pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
        num_waveforms_per_prompt (`int`, *optional*, defaults to 1):
            The number of waveforms to generate per prompt. If `num_waveforms_per_prompt > 1`, then automatic
            scoring is performed between the generated outputs and the text prompt. This scoring ranks the
            generated waveforms based on their cosine similarity with the text input in the joint text-audio
            embedding space.
        eta (`float`, *optional*, defaults to 0.0):
            Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
            to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
        generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
            A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
            generation deterministic.
        latents (`torch.Tensor`, *optional*):
            Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for spectrogram
            generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
            tensor is generated by sampling using the supplied random `generator`.
        prompt_embeds (`torch.Tensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
            provided, text embeddings are generated from the `prompt` input argument.
        negative_prompt_embeds (`torch.Tensor`, *optional*):
            Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
            not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
        generated_prompt_embeds (`torch.Tensor`, *optional*):
            Pre-generated text embeddings from the GPT2 langauge model. Can be used to easily tweak text inputs,
                *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input
                argument.
        negative_generated_prompt_embeds (`torch.Tensor`, *optional*):
            Pre-generated negative text embeddings from the GPT2 language model. Can be used to easily tweak text
            inputs, *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be computed from
            `negative_prompt` input argument.
        attention_mask (`torch.LongTensor`, *optional*):
            Pre-computed attention mask to be applied to the `prompt_embeds`. If not provided, attention mask will
            be computed from `prompt` input argument.
        negative_attention_mask (`torch.LongTensor`, *optional*):
            Pre-computed attention mask to be applied to the `negative_prompt_embeds`. If not provided, attention
            mask will be computed from `negative_prompt` input argument.
        max_new_tokens (`int`, *optional*, defaults to None):
            Number of new tokens to generate with the GPT2 language model. If not provided, number of tokens will
            be taken from the config of the model.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
            plain tuple.
        callback (`Callable`, *optional*):
            A function that calls every `callback_steps` steps during inference. The function is called with the
            following arguments: `callback(step: int, timestep: int, latents: torch.Tensor)`.
        callback_steps (`int`, *optional*, defaults to 1):
            The frequency at which the `callback` function is called. If not specified, the callback is called at
            every step.
        cross_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
            [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        output_type (`str`, *optional*, defaults to `"np"`):
            The output format of the generated audio. Choose between `"np"` to return a NumPy `np.ndarray` or
            `"pt"` to return a PyTorch `torch.Tensor` object. Set to `"latent"` to return the latent diffusion
            model (LDM) output.

    Examples:

    Returns:
        [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
            otherwise a `tuple` is returned where the first element is a list with the generated audio.
    """
    # 0. Convert audio input length from seconds to spectrogram height
    vocoder_upsample_factor = np.prod(self.vocoder.config.upsample_rates) / self.vocoder.config.sampling_rate

    if audio_length_in_s is None:
        audio_length_in_s = self.unet.config.sample_size * self.vae_scale_factor * vocoder_upsample_factor

    height = int(audio_length_in_s / vocoder_upsample_factor)

    original_waveform_length = int(audio_length_in_s * self.vocoder.config.sampling_rate)
    if height % self.vae_scale_factor != 0:
        height = int(np.ceil(height / self.vae_scale_factor)) * self.vae_scale_factor
        logger.info(
            f"Audio length in seconds {audio_length_in_s} is increased to {height * vocoder_upsample_factor} "
            f"so that it can be handled by the model. It will be cut to {audio_length_in_s} after the "
            f"denoising process."
        )

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        audio_length_in_s,
        vocoder_upsample_factor,
        callback_steps,
        transcription,
        negative_prompt,
        prompt_embeds,
        negative_prompt_embeds,
        generated_prompt_embeds,
        negative_generated_prompt_embeds,
        attention_mask,
        negative_attention_mask,
    )

    # 2. Define call parameters
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
    prompt_embeds, attention_mask, generated_prompt_embeds = self.encode_prompt(
        prompt,
        device,
        num_waveforms_per_prompt,
        do_classifier_free_guidance,
        transcription,
        negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        generated_prompt_embeds=generated_prompt_embeds,
        negative_generated_prompt_embeds=negative_generated_prompt_embeds,
        attention_mask=attention_mask,
        negative_attention_mask=negative_attention_mask,
        max_new_tokens=max_new_tokens,
    )

    # 4. Prepare timesteps
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps

    # 5. Prepare latent variables
    num_channels_latents = self.unet.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_waveforms_per_prompt,
        num_channels_latents,
        height,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 6. Prepare extra step kwargs
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    # 7. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    if self.local == False:# perform parastep
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                self.inference_step_num += 1
                if self.inference_step_num <= self.warm_up:#warm_up, use original computation
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=generated_prompt_embeds,
                        encoder_hidden_states_1=prompt_embeds,
                        encoder_attention_mask_1=attention_mask,
                        return_dict=False,
                    )[0]
                    self.noise_pred_cache = noise_pred.clone()
                else:# end warmup, use parastep
                    if self.step_rank == 0:
                        if self.round == 0: # master of this round
                            noise_pred = self.unet(
                                latent_model_input,
                                t,
                                encoder_hidden_states=generated_prompt_embeds,
                                encoder_hidden_states_1=prompt_embeds,
                                encoder_attention_mask_1=attention_mask,
                                return_dict=False,
                            )[0]
                            self.noise_pred_cache = noise_pred.clone()
                        else:
                            # skip
                            noise_pred = self.noise_pred_cache.clone()
                            dist.recv(noise_pred, self.round, self.step_mesh.get_group())
                            self.noise_pred_cache = noise_pred.clone()
                    else:
                        if self.step_rank == self.round:# master of this round, perform `predict` in reuse-then-predict
                            noise_pred = self.unet(
                                latent_model_input,
                                t,
                                encoder_hidden_states=generated_prompt_embeds,
                                encoder_hidden_states_1=prompt_embeds,
                                encoder_attention_mask_1=attention_mask,
                                return_dict=False,
                            )[0]
                            self.noise_pred_cache = noise_pred.clone()
                            dist.send(noise_pred, 0, self.step_mesh.get_group())
                            #
                        else:# `reuse` in reuse-then-predict
                            noise_pred = self.noise_pred_cache.clone()

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                if self.inference_step_num > self.warm_up: #do not need communitation in warmup steps
                    if self.round == self.step_size - 1:
                        dist.broadcast(latents, 0, self.step_mesh.get_group())
                    self.round += 1
                    self.round = self.round % self.step_size

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)
    else:# perform BatchStep
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            local_batch_size = 0
            for i, t in enumerate(timesteps):
                self.inference_step_num += 1
                if self.has_predict_len > 0:# batchstep
                    self.has_predict_len -= 1
                    progress_bar.update()
                    continue

                if self.inference_step_num <= self.warm_up:# warmup
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=generated_prompt_embeds,
                        encoder_hidden_states_1=prompt_embeds,
                        encoder_attention_mask_1=attention_mask,
                        return_dict=False,
                    )[0]
                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    self.noise_pred_cache = noise_pred.clone()
                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                else:# batchstep
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    assert self.has_predict_len == 0, "self.has_predict_len should be zero"
                    latent_model_input_list = []
                    latent_model_input_list.append(latent_model_input)
                    pred_latents = latents.clone()
                    lobal_batchsize = latent_model_input.shape[0]
                    # t = t.expand(lobal_batchsize)
                    t_list = []
                    t_list.append(t.expand(lobal_batchsize))
                    for pred_index in range(i+1, i+self.predict_len):
                        if (pred_index) >= len(timesteps)-1:#最后一步的时候，已经不需要预测
                            break
                        self.has_predict_len += 1
                        t_now = timesteps[pred_index-1]
                        t_nxt = timesteps[pred_index]
                        pred_latents = self.scheduler.step(self.noise_pred_cache, t_now, pred_latents, **extra_step_kwargs).prev_sample
                        pred_latent_model_input = torch.cat([pred_latents] * 2) if do_classifier_free_guidance else pred_latents
                        pred_latent_model_input = self.scheduler.scale_model_input(pred_latent_model_input, t_nxt)
                        latent_model_input_list.append(pred_latent_model_input)
                        t_list.append(t_nxt.expand(lobal_batchsize))

                    
                    batched_latent_model_input = torch.cat(latent_model_input_list, dim=0)
                    batched_t = torch.cat(t_list, dim=0)
                    batched_generated_prompt_embeds = torch.cat([generated_prompt_embeds]*(self.has_predict_len+1), dim=0)
                    batched_prompt_embeds = torch.cat([prompt_embeds]*(self.has_predict_len+1), dim=0)
                    batched_attention_mask = torch.cat([attention_mask]*(self.has_predict_len+1), dim=0)

                    # print(f"latent_model_input.shape=={batched_latent_model_input.shape}, t.shape=={batched_t.shape},generated_prompt_embeds.shape=={batched_generated_prompt_embeds.shape}, prompt_embeds.shape=={batched_prompt_embeds.shape}, attention_mask.shape=={batched_attention_mask.shape} ")
                    batched_noise_pred = self.unet(
                        batched_latent_model_input,
                        batched_t,
                        encoder_hidden_states=batched_generated_prompt_embeds,
                        encoder_hidden_states_1=batched_prompt_embeds,
                        encoder_attention_mask_1=batched_attention_mask,
                        return_dict=False,
                    )[0]

                    chunked_noise_pred = torch.chunk(batched_noise_pred, self.has_predict_len+1, dim=0)
                    
                    for noise_index in range(0, self.has_predict_len+1):
                        noise_pred = chunked_noise_pred[noise_index]
                        t_local = timesteps[i+noise_index]
                        # perform guidance
                        if do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                        # compute the previous noisy sample x_t -> x_t-1
                        latents = self.scheduler.step(noise_pred, t_local, latents, **extra_step_kwargs).prev_sample
                        if noise_index == self.has_predict_len:
                            self.noise_pred_cache = noise_pred.clone()

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

    self.maybe_free_model_hooks()

    # 8. Post-processing
    if not output_type == "latent":
        latents = 1 / self.vae.config.scaling_factor * latents
        mel_spectrogram = self.vae.decode(latents).sample
    else:
        return AudioPipelineOutput(audios=latents)

    audio = self.mel_spectrogram_to_waveform(mel_spectrogram)

    audio = audio[:, :original_waveform_length]

    # 9. Automatic scoring
    if num_waveforms_per_prompt > 1 and prompt is not None:
        audio = self.score_waveforms(
            text=prompt,
            audio=audio,
            num_waveforms_per_prompt=num_waveforms_per_prompt,
            device=device,
            dtype=prompt_embeds.dtype,
        )

    if output_type == "np":
        audio = audio.numpy()

    if not return_dict:
        return (audio,)

    return AudioPipelineOutput(audios=audio)