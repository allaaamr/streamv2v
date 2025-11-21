# wrapper.py
import gc
import os
from pathlib import Path
import traceback
from typing import List, Literal, Optional, Union, Dict

import numpy as np
import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline, StableDiffusionInstructPix2PixPipeline
from diffusers.models.attention_processor import XFormersAttnProcessor
from PIL import Image

from src.streamv2v.pipeline import StreamV2V
from src.streamv2v.image_utils import postprocess_image
# Cached processors (now also cache CROSS-attention K/V)
from src.streamv2v.models.attention_processor import (
    CachedSTXFormersAttnProcessor,
    CachedSTAttnProcessor2_0,  # in case you later need a non-xformers path
)

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class StreamV2VWrapper:
    def __init__(
        self,
        model_id_or_path: str,
        t_index_list: List[int],
        lora_dict: Optional[Dict[str, float]] = None,
        output_type: Literal["pil", "pt", "np", "latent"] = "pil",
        mode: Literal["img2img", "txt2img"] = "img2img",
        lcm_lora_id: Optional[str] = None,
        vae_id: Optional[str] = None,
        device: Literal["cpu", "cuda"] = "cuda",
        dtype: torch.dtype = torch.float16,
        frame_buffer_size: int = 1,
        width: int = 512,
        height: int = 512,
        warmup: int = 10,
        acceleration: Literal["none", "xformers", "tensorrt"] = "xformers",
        do_add_noise: bool = True,
        device_ids: Optional[List[int]] = None,
        use_lcm_lora: bool = True,
        use_tiny_vae: bool = True,
        enable_similar_image_filter: bool = False,
        similar_image_filter_threshold: float = 0.98,
        similar_image_filter_max_skip_frame: int = 10,
        use_denoising_batch: bool = True,
        cfg_type: Literal["none", "full", "self", "initialize"] = "none",
        use_cached_attn: bool = True,
        use_feature_injection: bool = True,
        feature_injection_strength: float = 0.8,
        feature_similarity_threshold: float = 0.98,
        cache_interval: int = 4,
        cache_maxframes: int = 1,
        use_tome_cache: bool = True,
        tome_metric: str = "keys",
        tome_ratio: float = 0.5,
        use_grid: bool = False,
        seed: int = 2,
        use_safety_checker: bool = False,
        engine_dir: Optional[Union[str, Path]] = "engines",
        # NEW: toggle for cross-attn caching
        cache_cross_attention: bool = True,
    ):
        """
        Wrapper with the same surface API your main.py expects
        (batch_size, __call__, prepare, img2img/txt2img, etc.),
        plus cross-attention caching via custom attention processors.
        """
        self.sd_turbo = "turbo" in (model_id_or_path or "")
        self.sd_xl = "xl" in (model_id_or_path or "")

        if mode == "txt2img":
            if cfg_type != "none":
                raise ValueError("txt2img mode accepts only cfg_type = 'none'")
            if use_denoising_batch and frame_buffer_size > 1 and not self.sd_turbo:
                raise ValueError("txt2img mode cannot use denoising batch with frame_buffer_size > 1.")
        if mode == "img2img" and not use_denoising_batch:
            raise NotImplementedError("vid2vid mode must use denoising batch for now.")
        self.mode = mode

        # Basic config
        self.device = device
        self.dtype = dtype
        self.width = width
        self.height = height
        self.output_type = output_type
        self.frame_buffer_size = frame_buffer_size
        # main.py uses this in warmup
        self.batch_size = (len(t_index_list) * frame_buffer_size) if use_denoising_batch else frame_buffer_size

        self.use_denoising_batch = use_denoising_batch
        self.use_cached_attn = use_cached_attn
        self.use_feature_injection = use_feature_injection
        self.feature_injection_strength = feature_injection_strength
        self.feature_similarity_threshold = feature_similarity_threshold
        self.cache_interval = cache_interval
        self.cache_maxframes = cache_maxframes
        self.use_tome_cache = use_tome_cache
        self.tome_metric = tome_metric
        self.tome_ratio = tome_ratio
        self.use_grid = use_grid
        self.use_safety_checker = use_safety_checker
        self.cache_cross_attention = cache_cross_attention

        # Build/load model and stream
        self.stream: StreamV2V = self._load_model(
            model_id_or_path=model_id_or_path,
            lora_dict=lora_dict,
            lcm_lora_id=lcm_lora_id,
            vae_id=vae_id,
            t_index_list=t_index_list,
            acceleration=acceleration,
            warmup=warmup,
            do_add_noise=do_add_noise,
            use_lcm_lora=use_lcm_lora,
            use_tiny_vae=use_tiny_vae,
            cfg_type=cfg_type,
            seed=seed,
            engine_dir=engine_dir,
        )

        # Optional similar-frame skipping
        if enable_similar_image_filter and hasattr(self.stream, "enable_similar_image_filter"):
            self.stream.enable_similar_image_filter(similar_image_filter_threshold, similar_image_filter_max_skip_frame)

        # Optional safety checker setup (unchanged)
        if self.use_safety_checker:
            from transformers import CLIPFeatureExtractor
            from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

            self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker"
            ).to(self.stream.pipe.device)
            self.feature_extractor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")
            self.nsfw_fallback_img = Image.new("RGB", (512, 512), (0, 0, 0))

    # -------- Public API expected by main.py --------
    def prepare(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        delta: float = 1.0,
    ) -> None:
        self.stream.prepare(
            prompt,
            negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            delta=delta,
        )

    def __call__(
        self,
        image: Optional[Union[str, Image.Image, torch.Tensor]] = None,
        prompt: Optional[str] = None,
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray]:
        if self.mode == "img2img":
            return self.img2img(image, prompt)
        else:
            return self.txt2img(prompt)

    def txt2img(
        self, prompt: Optional[str] = None
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray]:
        if prompt is not None and hasattr(self.stream, "update_prompt"):
            self.stream.update_prompt(prompt)

        if self.sd_turbo and hasattr(self.stream, "txt2img_sd_turbo"):
            image_tensor = self.stream.txt2img(self.batch_size)  # SD Turbo path can vary
        else:
            image_tensor = self.stream.txt2img(self.frame_buffer_size)
        return self._postprocess(image_tensor)

    def img2img(
        self, image: Union[str, Image.Image, torch.Tensor], prompt: Optional[str] = None
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray]:
        if prompt is not None and hasattr(self.stream, "update_prompt"):
            self.stream.update_prompt(prompt)

        if isinstance(image, (str, Image.Image)):
            image = self._preprocess(image)
        image_tensor = self.stream(image)
        return self._postprocess(image_tensor)

    # -------- Helpers --------
    def _preprocess(self, image: Union[str, Image.Image]) -> torch.Tensor:
        if isinstance(image, str):
            image = Image.open(image).convert("RGB").resize((self.width, self.height))
        if isinstance(image, Image.Image):
            image = image.convert("RGB").resize((self.width, self.height))
        return self.stream.image_processor.preprocess(image, self.height, self.width).to(
            device=self.device, dtype=self.dtype
        )

    def _postprocess(
        self, image_tensor: torch.Tensor, output_type: str = None
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray]:
        """
        Make behavior identical to your old wrapper:
        - When frame_buffer_size == 1: return a SINGLE image (CHW for 'pt', PIL for 'pil', etc.)
        - When frame_buffer_size > 1: return the whole batch
        """
        output_type = output_type or self.output_type
        result = postprocess_image(image_tensor.cpu(), output_type=output_type)

        # Old wrapper always returned the first item when frame_buffer_size == 1
        if self.frame_buffer_size == 1:
            # If result is a list, take first; if it's a 4D tensor [1,C,H,W], take [0]
            if isinstance(result, list):
                return result[0]
            if isinstance(result, torch.Tensor) and result.ndim == 4 and result.shape[0] == 1:
                return result[0]
            # If it's already CHW or a PIL image, just return it
            return result
        else:
            # Return the whole batch for multi-frame buffer
            return result

    # -------- Build & attach models --------
    def _load_model(
        self,
        model_id_or_path: str,
        t_index_list: List[int],
        lora_dict: Optional[Dict[str, float]] = None,
        lcm_lora_id: Optional[str] = None,
        vae_id: Optional[str] = None,
        acceleration: Literal["none", "xformers", "tensorrt"] = "xformers",
        warmup: int = 10,
        do_add_noise: bool = True,
        use_lcm_lora: bool = True,
        use_tiny_vae: bool = True,
        cfg_type: Literal["none", "full", "self", "initialize"] = "self",
        seed: int = 2,
        engine_dir: Optional[Union[str, Path]] = "engines",
    ) -> StreamV2V:
        # Use instruct-pix2pix (same as your old code)
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix"
        ).to(device=self.device, dtype=self.dtype)

        stream = StreamV2V(
            pipe=pipe,
            t_index_list=t_index_list,
            torch_dtype=self.dtype,
            width=self.width,
            height=self.height,
            do_add_noise=do_add_noise,
            frame_buffer_size=self.frame_buffer_size,
            use_denoising_batch=self.use_denoising_batch,
            cfg_type=cfg_type,
        )

        # LCM-LoRA & user LoRAs
        if not self.sd_turbo:
            if use_lcm_lora:
                if lcm_lora_id is not None:
                    stream.load_lcm_lora(pretrained_model_name_or_path_or_dict=lcm_lora_id, adapter_name="lcm")
                else:
                    stream.load_lcm_lora(
                        pretrained_model_name_or_path_or_dict="latent-consistency/lcm-lora-sdv1-5",
                        adapter_name="lcm",
                    )
            if lora_dict is not None:
                for lora_name, lora_scale in lora_dict.items():
                    stream.load_lora(lora_name)

        # Tiny VAE
        if use_tiny_vae:
            if vae_id is not None:
                stream.vae = AutoencoderTiny.from_pretrained(vae_id).to(device=pipe.device, dtype=pipe.dtype)
            else:
                stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(device=pipe.device, dtype=pipe.dtype)

        # Acceleration + cached attention (per attention module) + cross-attn K/V cache
        try:
            if acceleration == "xformers":
                stream.pipe.enable_xformers_memory_efficient_attention()
                if self.use_cached_attn:
                    attn_processors = stream.pipe.unet.attn_processors
                    new_attn_processors = {}
                    for key, attn_processor in attn_processors.items():
                        assert isinstance(
                            attn_processor, XFormersAttnProcessor
                        ), "We only replace XFormersAttnProcessor with CachedSTXFormersAttnProcessor"
                        new_attn_processors[key] = CachedSTXFormersAttnProcessor(
                            name=key,
                            use_feature_injection=self.use_feature_injection,
                            feature_injection_strength=self.feature_injection_strength,
                            feature_similarity_threshold=self.feature_similarity_threshold,
                            interval=self.cache_interval,
                            max_frames=self.cache_maxframes,
                            use_tome_cache=self.use_tome_cache,
                            tome_metric=self.tome_metric,
                            tome_ratio=self.tome_ratio,
                            use_grid=self.use_grid,
                            cache_cross_attention=self.cache_cross_attention,  # <- cross-attn K/V caching
                        )
                    stream.pipe.unet.set_attn_processor(new_attn_processors)
        except Exception:
            traceback.print_exc()
            print("Acceleration has failed. Falling back to normal mode.")

        # Seed + initial prepare
        if seed < 0:
            seed = np.random.randint(0, 1_000_000)

        stream.prepare(
            "",
            "",
            num_inference_steps=50,
            guidance_scale=1.2 if stream.cfg_type in ["full", "self", "initialize"] else 1.0,
            generator=torch.manual_seed(seed),
            seed=seed,
        )

        return stream
