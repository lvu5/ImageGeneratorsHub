import gc
from typing import List

import torch
from diffusers import (
    StableDiffusionPipeline,
    AutoPipelineForText2Image,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    EulerDiscreteScheduler
)
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from config import USE_CUDA
from utils import pil_image_to_base64

device = "cuda" if USE_CUDA and torch.cuda.is_available() else "cpu"

# Dictionary to store pipeline loaders
pipeline_loaders = {
    "runwayml": lambda: load_runwayml_pipeline(),
    "sdxl_turbo": lambda: load_sdxl_turbo_pipeline(),
    "sdxl_lightning": lambda: load_sdxl_lightning_pipeline(),
}

engine_kwargs = {
    "sdxl_lightning": {
        "num_inference_steps": 2,
        "guidance_scale": 0.0
    }
}

# Dictionary to store loaded pipelines
local_pipelines = {}


def load_runwayml_pipeline():
    """
    Load the RunwayML pipeline.
    """
    model_id = "runwayml/stable-diffusion-v1-5"
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16 if USE_CUDA else torch.float32
    ).to(device)
    return pipe


def load_sdxl_turbo_pipeline():
    """
    Load the SDXL Turbo pipeline.
    """
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16 if USE_CUDA else torch.float32,
        variant="fp16" if USE_CUDA else None
    ).to(device)
    return pipe


def load_sdxl_lightning_pipeline():
    """
    Load the SDXL Lightning pipeline.
    """
    base = "stabilityai/stable-diffusion-xl-base-1.0"
    repo = "ByteDance/SDXL-Lightning"
    ckpt = "sdxl_lightning_2step_unet.safetensors"

    unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(
        device,
        torch.float16 if USE_CUDA else torch.float32
    )
    unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device))
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base,
        unet=unet,
        torch_dtype=torch.float16 if USE_CUDA else torch.float32,
        variant="fp16" if USE_CUDA else None
    ).to(device)
    # trailing timesteps for Euler
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    return pipe


def get_pipeline(engine_name: str):
    """
    Lazily load and cache the pipeline for the given engine name.
    """
    if engine_name not in pipeline_loaders:
        raise ValueError(f"Unknown engine: {engine_name}")

    # Check if pipeline is already loaded
    if engine_name not in local_pipelines:
        print(f"Loading pipeline: {engine_name}")
        local_pipelines[engine_name] = pipeline_loaders[engine_name]()  # Load the pipeline

    return local_pipelines[engine_name]



def generate_sd_images(engine_name: str, prompt: str, k: int, width: int, height: int) -> List[str]:
    """
    Generate images using a local stable diffusion pipeline.
    Returns a list of base64-encoded PNG images.
    """
    pipe = get_pipeline(engine_name)  # Lazily load the pipeline
    gc.collect()
    if USE_CUDA:
        torch.cuda.empty_cache()

    # Generate images
    result = pipe(prompt, num_images_per_prompt=k, width=width, height=height, **engine_kwargs.get(engine_name, {}))
    images = result.images  # List of PIL images

    # Encode images to base64
    encoded_images = [pil_image_to_base64(img) for img in images]
    return encoded_images
