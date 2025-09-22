import torch

from diffusers.utils import replace_example_docstring
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.image_processor import PipelineImageInput

import torch.distributed as dist

import PIL.Image
from dataclasses import dataclass

from diffusers.utils import BaseOutput
import numpy as np

from diffusers.utils.torch_utils import randn_tensor
import os


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from diffusers import StableVideoDiffusionPipeline
        >>> from diffusers.utils import load_image, export_to_video

        >>> pipe = StableVideoDiffusionPipeline.from_pretrained(
        ...     "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
        ... )
        >>> pipe.to("cuda")

        >>> image = load_image(
        ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd-docstring-example.jpeg"
        ... )
        >>> image = image.resize((1024, 576))

        >>> frames = pipe(image, num_frames=25, decode_chunk_size=8).frames[0]
        >>> export_to_video(frames, "generated.mp4", fps=7)
        ```
"""

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

def _append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]

@dataclass
class StableVideoDiffusionPipelineOutput(BaseOutput):
    r"""
    Output class for Stable Video Diffusion pipeline.

    Args:
        frames (`[List[List[PIL.Image.Image]]`, `np.ndarray`, `torch.Tensor`]):
            List of denoised PIL images of length `batch_size` or numpy array or torch tensor of shape `(batch_size,
            num_frames, height, width, num_channels)`.
    """

    frames: Union[List[List[PIL.Image.Image]], np.ndarray, torch.Tensor]

@torch.no_grad()
@replace_example_docstring(EXAMPLE_DOC_STRING)
def step_parallelism_call(
    self,
    image: Union[PIL.Image.Image, List[PIL.Image.Image], torch.Tensor],
    height: int = 576,
    width: int = 1024,
    num_frames: Optional[int] = None,
    num_inference_steps: int = 25,
    sigmas: Optional[List[float]] = None,
    min_guidance_scale: float = 1.0,
    max_guidance_scale: float = 3.0,
    fps: int = 7,
    motion_bucket_id: int = 127,
    noise_aug_strength: float = 0.02,
    decode_chunk_size: Optional[int] = None,
    num_videos_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.Tensor] = None,
    output_type: Optional[str] = "pil",
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    return_dict: bool = True,
):
    r"""
    The call function to the pipeline for generation.

    Args:
        image (`PIL.Image.Image` or `List[PIL.Image.Image]` or `torch.Tensor`):
            Image(s) to guide image generation. If you provide a tensor, the expected value range is between `[0,
            1]`.
        height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
            The height in pixels of the generated image.
        width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
            The width in pixels of the generated image.
        num_frames (`int`, *optional*):
            The number of video frames to generate. Defaults to `self.unet.config.num_frames` (14 for
            `stable-video-diffusion-img2vid` and to 25 for `stable-video-diffusion-img2vid-xt`).
        num_inference_steps (`int`, *optional*, defaults to 25):
            The number of denoising steps. More denoising steps usually lead to a higher quality video at the
            expense of slower inference. This parameter is modulated by `strength`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
            their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
            will be used.
        min_guidance_scale (`float`, *optional*, defaults to 1.0):
            The minimum guidance scale. Used for the classifier free guidance with first frame.
        max_guidance_scale (`float`, *optional*, defaults to 3.0):
            The maximum guidance scale. Used for the classifier free guidance with last frame.
        fps (`int`, *optional*, defaults to 7):
            Frames per second. The rate at which the generated images shall be exported to a video after
            generation. Note that Stable Diffusion Video's UNet was micro-conditioned on fps-1 during training.
        motion_bucket_id (`int`, *optional*, defaults to 127):
            Used for conditioning the amount of motion for the generation. The higher the number the more motion
            will be in the video.
        noise_aug_strength (`float`, *optional*, defaults to 0.02):
            The amount of noise added to the init image, the higher it is the less the video will look like the
            init image. Increase it for more motion.
        decode_chunk_size (`int`, *optional*):
            The number of frames to decode at a time. Higher chunk size leads to better temporal consistency at the
            expense of more memory usage. By default, the decoder decodes all frames at once for maximal quality.
            For lower memory usage, reduce `decode_chunk_size`.
        num_videos_per_prompt (`int`, *optional*, defaults to 1):
            The number of videos to generate per prompt.
        generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
            A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
            generation deterministic.
        latents (`torch.Tensor`, *optional*):
            Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for video
            generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
            tensor is generated by sampling using the supplied random `generator`.
        output_type (`str`, *optional*, defaults to `"pil"`):
            The output format of the generated image. Choose between `pil`, `np` or `pt`.
        callback_on_step_end (`Callable`, *optional*):
            A function that is called at the end of each denoising step during inference. The function is called
            with the following arguments:
                `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`.
            `callback_kwargs` will include a list of all tensors as specified by
            `callback_on_step_end_tensor_inputs`.
        callback_on_step_end_tensor_inputs (`List`, *optional*):
            The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
            will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
            `._callback_tensor_inputs` attribute of your pipeline class.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
            plain tuple.

    Examples:

    Returns:
        [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] or `tuple`:
            If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] is
            returned, otherwise a `tuple` of (`List[List[PIL.Image.Image]]` or `np.ndarray` or `torch.Tensor`) is
            returned.
    """
    # 0. Default height and width to unet

    # print("hello world")

    height = height or self.unet.config.sample_size * self.vae_scale_factor
    width = width or self.unet.config.sample_size * self.vae_scale_factor

    num_frames = num_frames if num_frames is not None else self.unet.config.num_frames
    decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(image, height, width)

    # 2. Define call parameters
    if isinstance(image, PIL.Image.Image):
        batch_size = 1
    elif isinstance(image, list):
        batch_size = len(image)
    else:
        batch_size = image.shape[0]
    device = self._execution_device
    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    self._guidance_scale = max_guidance_scale

    # 3. Encode input image
    image_embeddings = self._encode_image(image, device, num_videos_per_prompt, self.do_classifier_free_guidance)

    # NOTE: Stable Video Diffusion was conditioned on fps - 1, which is why it is reduced here.
    # See: https://github.com/Stability-AI/generative-models/blob/ed0997173f98eaf8f4edf7ba5fe8f15c6b877fd3/scripts/sampling/simple_video_sample.py#L188
    fps = fps - 1

    # 4. Encode input image using VAE
    image = self.video_processor.preprocess(image, height=height, width=width).to(device)
    noise = randn_tensor(image.shape, generator=generator, device=device, dtype=image.dtype)
    image = image + noise_aug_strength * noise

    needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
    if needs_upcasting:
        self.vae.to(dtype=torch.float32)

    image_latents = self._encode_vae_image(
        image,
        device=device,
        num_videos_per_prompt=num_videos_per_prompt,
        do_classifier_free_guidance=self.do_classifier_free_guidance,
    )
    image_latents = image_latents.to(image_embeddings.dtype)

    # cast back to fp16 if needed
    if needs_upcasting:
        self.vae.to(dtype=torch.float16)

    # Repeat the image latents for each frame so we can concatenate them with the noise
    # image_latents [batch, channels, height, width] ->[batch, num_frames, channels, height, width]
    image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)

    # 5. Get Added Time IDs
    added_time_ids = self._get_add_time_ids(
        fps,
        motion_bucket_id,
        noise_aug_strength,
        image_embeddings.dtype,
        batch_size,
        num_videos_per_prompt,
        self.do_classifier_free_guidance,
    )
    added_time_ids = added_time_ids.to(device)

    # 6. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, None, sigmas)

    # 7. Prepare latent variables
    num_channels_latents = self.unet.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_videos_per_prompt,
        num_frames,
        num_channels_latents,
        height,
        width,
        image_embeddings.dtype,
        device,
        generator,
        latents,
    )

    # 8. Prepare guidance scale
    guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
    guidance_scale = guidance_scale.to(device, latents.dtype)
    guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
    guidance_scale = _append_dims(guidance_scale, latents.ndim)

    self._guidance_scale = guidance_scale

    # 9. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    self._num_timesteps = len(timesteps)
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # Concatenate image_latents over channels dimension
            latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)

            self.inference_step_num += 1
            if self.inference_step_num <= self.warm_up:# warmup
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=image_embeddings,
                    added_time_ids=added_time_ids,
                    return_dict=False,
                )[0]
                self.noise_pred_cache = noise_pred.clone()
            else:# parastep
                if self.step_rank == 0:
                    if self.round == 0:
                        noise_pred = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=image_embeddings,
                            added_time_ids=added_time_ids,
                            return_dict=False,
                        )[0]
                        self.noise_pred_cache = noise_pred.clone()
                    else:
                        noise_pred = self.noise_pred_cache.clone()
                        dist.recv(noise_pred, self.round, self.step_mesh.get_group())
                        self.noise_pred_cache = noise_pred.clone()
                else:
                    if self.step_rank == self.round:# predict in reuse-then-predict
                        noise_pred = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=image_embeddings,
                            added_time_ids=added_time_ids,
                            return_dict=False,
                        )[0]
                        self.noise_pred_cache = noise_pred.clone()
                        dist.send(noise_pred, 0, self.step_mesh.get_group())
                    else:# reuse in reuse-then-predict
                        noise_pred = self.noise_pred_cache.clone()

            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

           
            #....
            if self.inference_step_num > self.warm_up: 
                if self.round == self.step_size - 1:
                    dist.broadcast(latents, 0, self.step_mesh.get_group())
                self.round += 1
                self.round = self.round % self.step_size

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)

            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()

    if not output_type == "latent":
        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)
        frames = self.decode_latents(latents, num_frames, decode_chunk_size)
        frames = self.video_processor.postprocess_video(video=frames, output_type=output_type)
    else:
        frames = latents

    self.maybe_free_model_hooks()

    if not return_dict:
        return frames

    return StableVideoDiffusionPipelineOutput(frames=frames)