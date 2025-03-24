from enum import Enum
from typing import List, Dict, Any

import replicate
from fastapi import HTTPException

from core.image_generator import ImageGenerator
from models.schemas import EngineRequirement
from utils import url_to_base64, img_to_base64


class ReplicateGenerator(ImageGenerator):
    class Size(Enum):
        SMALL = (256, 256)
        MEDIUM = (512, 512)
        LARGE = (1024, 1024)

    def __init__(self):
        super().__init__(
            name="Replicate",
            description="Replicate.com hosted models"
        )

    async def generate(self, params: Dict[str, Any], prompt: str, size: "ReplicateGenerator.Size", num_images: int) -> \
            List[str]:
        if "api_token" not in params:
            raise HTTPException(status_code=400, detail="Replicate API token is required")

        # try:
        client = replicate.Client(api_token=params["api_token"])
        model = params.get("model", "stability-ai/sdxl")

        # Configure for multiple images
        input_params = {
            "prompt": prompt,
            "width": size.value[0],
            "height": size.value[1],
            "num_outputs": num_images,
            "output_format": "png"
        }

        response = await client.async_run(
            model,
            input=input_params
        )
        return [url_to_base64(url) for url in response]

        # except Exception as e:
        #     raise HTTPException(status_code=500, detail=f"Replicate generation failed: {str(e)}")

    def get_required_params(self) -> List[EngineRequirement]:
        return [
            EngineRequirement(
                name="api_token",
                description="Replicate API token"
            ),
            EngineRequirement(
                name="model",
                description="Model identifier on Replicate"
            )
        ]


class RealVisXL(ImageGenerator):
    class Size(Enum):
        SMALL = (512, 512)
        MEDIUM = (768, 768)
        LARGE = (1024, 1024)

    def __init__(self):
        super().__init__(name="RealVisXL", description="adirik/realvisxl-v3.0-turbo on Replicate.com")

    async def generate(self, params: Dict[str, Any], prompt: str, size: "RealVisXL.Size", num_images: int) -> \
            List[str]:
        if "api_token" not in params:
            raise HTTPException(status_code=400, detail="Replicate API token is required")

        # try:
        client = replicate.Client(api_token=params["api_token"])
        model = "adirik/realvisxl-v3.0-turbo:3dc73c805b11b4b01a60555e532fd3ab3f0e60d26f6584d9b8ba7e1b95858243"

        # Configure for multiple images
        input_params = {
            "prompt": prompt,
            "width": size.value[0],
            "height": size.value[1],
            "num_outputs": num_images,
            "output_format": "png",
            "refine": "no_refiner",
            "scheduler": "DPM++_SDE_Karras",
            "guidance_scale": 2,
            "apply_watermark": False,
            "high_noise_frac": 0.8,
            "negative_prompt": "(worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch), open mouth",
            "prompt_strength": 0.8,
            "num_inference_steps": 25
        }

        response = await client.async_run(
            model,
            input=input_params
        )
        return [url_to_base64(url) for url in response]

    def get_required_params(self) -> List[EngineRequirement]:
        return [
            EngineRequirement(
                name="api_token",
                description="Replicate API token"
            )
        ]


class Imagen3(ImageGenerator):
    class Size(Enum):
        SMALL = (512, 512)
        MEDIUM = (768, 768)
        LARGE = (1024, 1024)

    def __init__(self):
        super().__init__(name="Imagen3-fast", description="https://replicate.com/google/imagen-3-fast/api")

    async def generate(self, params: Dict[str, Any], prompt: str, size: "RealVisXL.Size", num_images: int) -> \
            List[str]:
        if "api_token" not in params:
            raise HTTPException(status_code=400, detail="Replicate API token is required")

        # try:
        client = replicate.Client(api_token=params["api_token"])
        model = "google/imagen-3-fast"

        # Configure for multiple images
        input_params = {
            "prompt": prompt,
            "aspect_ratio": "1:1",
            "safety_filter_level": "block_only_high",
            "output_format": "png",
        }
        results = []
        for i in range(num_images):
            response = await client.async_run(
                model,
                input=input_params
            )
            results.append(url_to_base64(response))

        return results


    def get_required_params(self) -> List[EngineRequirement]:
        return [
            EngineRequirement(
                name="api_token",
                description="Replicate API token"
            )
        ]
