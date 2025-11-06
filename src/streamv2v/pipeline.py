import glob
import os
import time
from typing import List, Optional, Union, Any, Dict, Tuple, Literal
from collections import deque

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_small

from diffusers import LCMScheduler, StableDiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    retrieve_latents,
)
from .image_utils import postprocess_image, forward_backward_consistency_check
from .models.utils import get_nn_latent
from .image_filter import SimilarImageFilter


class StreamV2V:
    def __init__(
        self,
        pipe: StableDiffusionPipeline,
        t_index_list: List[int],
        torch_dtype: torch.dtype = torch.float16,
        width: int = 512,
        height: int = 512,
        do_add_noise: bool = True,
        use_denoising_batch: bool = True,
        frame_buffer_size: int = 1,
        cfg_type: Literal["none", "full", "self", "initialize"] = "self",
    ) -> None:
        self.device = pipe.device
        self.dtype = torch_dtype
        self.generator = None

        self.height = height
        self.width = width

        self.latent_height = int(height // pipe.vae_scale_factor)
        self.latent_width = int(width // pipe.vae_scale_factor)

        self.frame_bff_size = frame_buffer_size
        self.denoising_steps_num = len(t_index_list)

        self.cfg_type = cfg_type

        if use_denoising_batch:

            self.batch_size = self.denoising_steps_num * frame_buffer_size

            if self.cfg_type == "initialize":

                self.trt_unet_batch_size = (

                    self.denoising_steps_num + 1

                ) * self.frame_bff_size

            elif self.cfg_type == "full":

                self.trt_unet_batch_size = (

                    2 * self.denoising_steps_num * self.frame_bff_size

                )

            else:

                self.trt_unet_batch_size = self.denoising_steps_num * frame_buffer_size

        else:

            self.trt_unet_batch_size = self.frame_bff_size

            self.batch_size = frame_buffer_size

        self.t_list = t_index_list

        self.do_add_noise = do_add_noise
        self.use_denoising_batch = use_denoising_batch

        self.similar_image_filter = False
        self.similar_filter = SimilarImageFilter()
        self.prev_image_tensor = None
        self.prev_x_t_latent = None
        self.prev_image_result = None

        self.pipe = pipe
        self.image_processor = VaeImageProcessor(pipe.vae_scale_factor)

        self.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.vae = pipe.vae

        self.cached_x_t_latent = deque(maxlen=4)

        self.inference_time_ema = 0

    def load_lcm_lora(
        self,
        pretrained_model_name_or_path_or_dict: Union[
            str, Dict[str, torch.Tensor]
        ] = "latent-consistency/lcm-lora-sdv1-5",
        adapter_name: Optional[Any] = 'lcm',
        **kwargs,
    ) -> None:
        self.pipe.load_lora_weights(
            pretrained_model_name_or_path_or_dict, adapter_name, **kwargs
        )

    def load_lora(
        self,
        pretrained_lora_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        adapter_name: Optional[Any] = None,
        **kwargs,
    ) -> None:
        self.pipe.load_lora_weights(
            pretrained_lora_model_name_or_path_or_dict, adapter_name, **kwargs
        )

    def fuse_lora(
        self,
        fuse_unet: bool = True,
        fuse_text_encoder: bool = True,
        lora_scale: float = 1.0,
        safe_fusing: bool = False,
    ) -> None:
        self.pipe.fuse_lora(
            fuse_unet=fuse_unet,
            fuse_text_encoder=fuse_text_encoder,
            lora_scale=lora_scale,
            safe_fusing=safe_fusing,
        )

    def enable_similar_image_filter(self, threshold: float = 0.98, max_skip_frame: float = 10) -> None:
        self.similar_image_filter = True
        self.similar_filter.set_threshold(threshold)
        self.similar_filter.set_max_skip_frame(max_skip_frame)

    def disable_similar_image_filter(self) -> None:
        self.similar_image_filter = False

    @torch.no_grad()
    def prepare(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 1.2,
        delta: float = 1.0,
        generator: Optional[torch.Generator] = torch.Generator(),
        seed: int = 2,
    ) -> None:
        self.generator = generator
        self.generator.manual_seed(seed)
        # initialize x_t_latent (it can be any random tensor)
        if self.denoising_steps_num > 1:
            self.x_t_latent_buffer = torch.zeros(
                (
                    (self.denoising_steps_num - 1) * self.frame_bff_size,
                    4,
                    self.latent_height,
                    self.latent_width,
                ),
                dtype=self.dtype,
                device=self.device,
            )
        else:
            self.x_t_latent_buffer = None

        if self.cfg_type == "none":

            self.guidance_scale = 1.0

        else:

            self.guidance_scale = guidance_scale

        self.delta = delta



        do_classifier_free_guidance = False

        if self.guidance_scale > 1.0:

            do_classifier_free_guidance = True
        encoder_output = self.pipe._encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
        )

        self.prompt_embeds = encoder_output[0].repeat(self.batch_size, 1, 1)

        self.null_prompt_embeds = encoder_output[1]



        if self.use_denoising_batch and self.cfg_type == "full":

            uncond_prompt_embeds = encoder_output[1].repeat(self.batch_size, 1, 1)

        elif self.cfg_type == "initialize":

            uncond_prompt_embeds = encoder_output[1].repeat(self.frame_bff_size, 1, 1)

        if self.guidance_scale > 1.0 and (

            self.cfg_type == "initialize" or self.cfg_type == "full"

        ):

            self.prompt_embeds = torch.cat(

                [uncond_prompt_embeds, self.prompt_embeds], dim=0

            )


        self.scheduler.set_timesteps(num_inference_steps, self.device)
        self.timesteps = self.scheduler.timesteps.to(self.device)

        # make sub timesteps list based on the indices in the t_list list and the values in the timesteps list
        self.sub_timesteps = []
        for t in self.t_list:
            self.sub_timesteps.append(self.timesteps[t])

        sub_timesteps_tensor = torch.tensor(
            self.sub_timesteps, dtype=torch.long, device=self.device
        )
        self.sub_timesteps_tensor = torch.repeat_interleave(
            sub_timesteps_tensor,
            repeats=self.frame_bff_size if self.use_denoising_batch else 1,
            dim=0,
        )

        self.init_noise = torch.randn(
            (self.batch_size, 4, self.latent_height, self.latent_width),
            generator=generator,
        ).to(device=self.device, dtype=self.dtype)

        self.randn_noise = self.init_noise[:1].clone()
        self.warp_noise = self.init_noise[:1].clone()

        self.stock_noise = torch.zeros_like(self.init_noise)

        c_skip_list = []
        c_out_list = []
        for timestep in self.sub_timesteps:
            c_skip, c_out = self.scheduler.get_scalings_for_boundary_condition_discrete(
                timestep
            )
            c_skip_list.append(c_skip)
            c_out_list.append(c_out)

        self.c_skip = (
            torch.stack(c_skip_list)
            .view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )
        self.c_out = (
            torch.stack(c_out_list)
            .view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )

        alpha_prod_t_sqrt_list = []
        beta_prod_t_sqrt_list = []
        for timestep in self.sub_timesteps:
            alpha_prod_t_sqrt = self.scheduler.alphas_cumprod[timestep].sqrt()
            beta_prod_t_sqrt = (1 - self.scheduler.alphas_cumprod[timestep]).sqrt()
            alpha_prod_t_sqrt_list.append(alpha_prod_t_sqrt)
            beta_prod_t_sqrt_list.append(beta_prod_t_sqrt)
        alpha_prod_t_sqrt = (
            torch.stack(alpha_prod_t_sqrt_list)
            .view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )
        beta_prod_t_sqrt = (
            torch.stack(beta_prod_t_sqrt_list)
            .view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )
        self.alpha_prod_t_sqrt = torch.repeat_interleave(
            alpha_prod_t_sqrt,
            repeats=self.frame_bff_size if self.use_denoising_batch else 1,
            dim=0,
        )
        self.beta_prod_t_sqrt = torch.repeat_interleave(
            beta_prod_t_sqrt,
            repeats=self.frame_bff_size if self.use_denoising_batch else 1,
            dim=0,
        )

    @torch.no_grad()
    def update_prompt(self, prompt: str) -> None:
        encoder_output = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        self.prompt_embeds = encoder_output[0].repeat(self.batch_size, 1, 1)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        t_index: int,
    ) -> torch.Tensor:
        noisy_samples = (
            self.alpha_prod_t_sqrt[t_index] * original_samples
            + self.beta_prod_t_sqrt[t_index] * noise
        )
        return noisy_samples

    def scheduler_step_batch(
        self,
        model_pred_batch: torch.Tensor,
        x_t_latent_batch: torch.Tensor,
        idx: Optional[int] = None,
    ) -> torch.Tensor:
        # TODO: use t_list to select beta_prod_t_sqrt
        if idx is None:
            F_theta = (
                x_t_latent_batch - self.beta_prod_t_sqrt * model_pred_batch
            ) / self.alpha_prod_t_sqrt
            denoised_batch = self.c_out * F_theta + self.c_skip * x_t_latent_batch
        else:
            F_theta = (
                x_t_latent_batch - self.beta_prod_t_sqrt[idx] * model_pred_batch
            ) / self.alpha_prod_t_sqrt[idx]
            denoised_batch = (
                self.c_out[idx] * F_theta + self.c_skip[idx] * x_t_latent_batch
            )

        return denoised_batch

    def unet_step(
    self,
    x_t_latent: torch.Tensor,                 # (B,4,H,W)
    z0_batch: torch.Tensor,                   # (B,4,H,W)
    t_list: Union[torch.Tensor, list[int]],
    idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Build 8-ch input: [scaled z_t, z0]
        if self.guidance_scale > 1.0 and (self.cfg_type == "initialize"):

            x_t_latent_plus_uc = torch.concat([x_t_latent[0:1], x_t_latent], dim=0)

            t_list = torch.concat([t_list[0:1], t_list], dim=0)

        elif self.guidance_scale > 1.0 and (self.cfg_type == "full"):

            x_t_latent_plus_uc = torch.concat([x_t_latent, x_t_latent], dim=0)

            t_list = torch.concat([t_list, t_list], dim=0)

        else:

            x_t_latent_plus_uc = x_t_latent
        scaled = self.scheduler.scale_model_input(x_t_latent, t_list)  # (B,4,H,W)
        model_in = torch.cat([scaled, z0_batch], dim=1)                # (B,8,H,W)

        model_pred = self.unet(
            model_in, t_list, encoder_hidden_states=self.prompt_embeds, return_dict=False
        )[0]

        # --- your CFG code unchanged ---
        if self.guidance_scale > 1.0 and (self.cfg_type == "initialize"):
            noise_pred_text = model_pred[1:]
            self.stock_noise = torch.concat([model_pred[0:1], self.stock_noise[1:]], dim=0)
        elif self.guidance_scale > 1.0 and (self.cfg_type == "full"):
            noise_pred_uncond, noise_pred_text = model_pred.chunk(2)
        else:
            noise_pred_text = model_pred
        if self.guidance_scale > 1.0 and (self.cfg_type == "self" or self.cfg_type == "initialize"):
            noise_pred_uncond = self.stock_noise * self.delta
        if self.guidance_scale > 1.0 and self.cfg_type != "none":
            model_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
        else:
            model_pred = noise_pred_text

        # IMPORTANT: scheduler gets z_t (4-ch), not z0, not [z_t,z0]
        denoised_batch = self.scheduler_step_batch(model_pred, x_t_latent, idx)
        return denoised_batch, model_pred


    def norm_noise(self, noise):
        # Compute mean and std of blended_noise
        mean = noise.mean()
        std = noise.std()

        # Normalize blended_noise to have mean=0 and std=1
        normalized_noise = (noise - mean) / std
        return normalized_noise
        
    def encode_image(self, image_tensors: torch.Tensor) -> torch.Tensor:
        image_tensors = image_tensors.to(device=self.device, dtype=self.vae.dtype)
        z0 = retrieve_latents(self.vae.encode(image_tensors), self.generator)
        z0 = z0 * self.vae.config.scaling_factor  # (1,4,H/8,W/8)
        return z0

    def decode_image(self, x_0_pred_out: torch.Tensor) -> torch.Tensor:
        output_latent = self.vae.decode(
            x_0_pred_out / self.vae.config.scaling_factor, return_dict=False
        )[0]
        return output_latent

    def predict_x0_batch(self, x_t_latent: torch.Tensor) -> torch.Tensor:
        prev_latent_batch = self.x_t_latent_buffer

        if self.use_denoising_batch:
            t_list = self.sub_timesteps_tensor

            # Build z_t batch
            if self.denoising_steps_num > 1:
                z_t_batch = torch.cat((x_t_latent, prev_latent_batch), dim=0)  # (B,4,H,W)
                self.stock_noise = torch.cat((self.init_noise[0:1], self.stock_noise[:-1]), dim=0)
            else:
                z_t_batch = x_t_latent  # (B,4,H,W) with B=self.frame_bff_size (usually 1)

            # Repeat z0 across the batch to match z_t_batch
            z0_batch = self.image_latents_z0.repeat(z_t_batch.shape[0], 1, 1, 1)  # (B,4,H,W)

            # RUN UNET on [scaled z_t, z0]
            x_0_pred_batch, model_pred = self.unet_step(z_t_batch, z0_batch, t_list)

            if self.denoising_steps_num > 1:
                x_0_pred_out = x_0_pred_batch[-1].unsqueeze(0)
                if self.do_add_noise:
                    self.x_t_latent_buffer = (
                        self.alpha_prod_t_sqrt[1:] * x_0_pred_batch[:-1]
                        + self.beta_prod_t_sqrt[1:]  * self.init_noise[1:]
                    )
                else:
                    self.x_t_latent_buffer = (self.alpha_prod_t_sqrt[1:] * x_0_pred_batch[:-1])
            else:
                x_0_pred_out = x_0_pred_batch
                self.x_t_latent_buffer = None

        else:
            # non-batch path
            self.init_noise = x_t_latent
            for idx, t in enumerate(self.sub_timesteps_tensor):
                t = t.view(1,).repeat(self.frame_bff_size,)
                z0_batch = self.image_latents_z0.repeat(self.frame_bff_size, 1, 1, 1)
                x_0_pred, model_pred = self.unet_step(x_t_latent, z0_batch, t, idx)

                if idx < len(self.sub_timesteps_tensor) - 1:
                    if self.do_add_noise:
                        x_t_latent = (
                            self.alpha_prod_t_sqrt[idx + 1] * x_0_pred
                            + self.beta_prod_t_sqrt[idx + 1]  * torch.randn_like(x_0_pred, device=self.device, dtype=self.dtype)
                        )
                    else:
                        x_t_latent = self.alpha_prod_t_sqrt[idx + 1] * x_0_pred
            x_0_pred_out = x_0_pred

        return x_0_pred_out

    @torch.no_grad()
    def __call__(self, x: Union[torch.Tensor, PIL.Image.Image, np.ndarray] = None) -> torch.Tensor:
        x = self.image_processor.preprocess(x, self.height, self.width).to(device=self.device, dtype=self.dtype)

        # --- get clean latents z0 and keep them ---
        if self.similar_image_filter:

                x = self.similar_filter(x)
        z0 = self.encode_image(x)                      # (1,4,H,W)
        self.image_latents_z0 = z0                     # stash for all steps

        # --- initial z_t at the first sub-timestep index 0 ---
        x_t_latent = self.add_noise(z0, self.init_noise[0], 0)  # (1,4,H,W)

        height, width = x_t_latent.shape[-2:]
        height = height * self.pipe.vae_scale_factor
        width  = width  * self.pipe.vae_scale_factor

        _ = self.pipe.prepare_latents(
            self.batch_size, self.pipe.vae.config.latent_channels, height, width,
            self.prompt_embeds.dtype, self.device, self.generator, None,
        )
        # breakpoint()
        x_0_pred_out = self.predict_x0_batch(x_t_latent)
        x_output = self.decode_image(x_0_pred_out).detach().clone()
        self.prev_image_result = x_output
        torch.cuda.synchronize()
        return x_output

    @torch.no_grad()
    def txt2img(self, batch_size: int = 1) -> torch.Tensor:
        x_0_pred_out = self.predict_x0_batch(self.init_noise[0:1])
        x_output = self.decode_image(x_0_pred_out).detach().clone()
        return x_output

    def txt2img_sd_turbo(self, batch_size: int = 1) -> torch.Tensor:
        x_t_latent = self.init_noise[0:1]
        model_pred = self.unet(
            x_t_latent,
            self.sub_timesteps_tensor,
            encoder_hidden_states=self.prompt_embeds,
            return_dict=False,
        )[0]
        x_0_pred_out = (
            x_t_latent - self.beta_prod_t_sqrt * model_pred
        ) / self.alpha_prod_t_sqrt
        return self.decode_image(x_0_pred_out)