# file: stable_diffusion_xl.py
import asyncio
import base64
import io
from enum import Enum
from typing import List, Dict, Any

import torch
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler, UNet2DConditionModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from core.image_generator import ImageGenerator
from models.schemas import EngineRequirement

class StableDiffusionXLGenerator(ImageGenerator):
    class Size(Enum):
        SMALL = (512, 512)
        MEDIUM = (768, 768)
        LARGE = (1024, 1024)

    def __init__(self):
        super().__init__(
            name="StableDiffusionXL",
            description="Image generator using Stable Diffusion XL optimized for GPU and CPU"
        )
        if torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = torch.float16
            variant = "fp16"
        elif torch.backends.mps.is_available():
            self.device = "mps"
            self.dtype = torch.float16
            variant = "fp16"
        else:
            self.device = "cpu"
            self.dtype = torch.float32
            variant = None

        base = "stabilityai/stable-diffusion-xl-base-1.0"
        repo = "ByteDance/SDXL-Lightning"
        ckpt = "sdxl_lightning_4step_unet.safetensors"  # Use the correct ckpt for your step setting!

        unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(self.device, self.dtype)
        ckpt_path = hf_hub_download(repo, ckpt)
        unet.load_state_dict(load_file(ckpt_path, device=self.device))

        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            base,
            unet=unet,
            torch_dtype=self.dtype,
            variant=variant
        ).to(self.device)

        self.pipeline.scheduler = EulerDiscreteScheduler.from_config(
            self.pipeline.scheduler.config,
            timestep_spacing="trailing"
        )

    async def generate(self, params: Dict[str, Any], prompt: str, size: "StableDiffusionXLGenerator.Size", num_images: int) -> List[str]:
        loop = asyncio.get_running_loop()
        width, height = size.value

        def run_pipeline() -> List[str]:
            images = []
            results = self.pipeline(
                prompt,
                height=height,
                width=width,
                num_inference_steps=2,
                guidance_scale=0,
                num_images_per_prompt=num_images,
            )
            for img in results.images:
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                base64_img = base64.b64encode(buffered.getvalue()).decode("utf-8")
                images.append(base64_img)
            return images

        return await loop.run_in_executor(None, run_pipeline)

    def get_required_params(self) -> List[EngineRequirement]:
        return []