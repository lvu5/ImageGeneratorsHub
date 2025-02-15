from enum import Enum
from typing import List, Dict, Any

from fastapi import HTTPException
from openai import AsyncOpenAI

from core.image_generator import ImageGenerator
from models.schemas import EngineRequirement


class DallEGenerator(ImageGenerator):
    class Size(Enum):
        SMALL = (256, 256)
        MEDIUM = (512, 512)
        LARGE = (1024, 1024)


    def __init__(self):
        super().__init__(
            name="DALL-E",
            description="OpenAI's DALL-E image generation model"
        )

    async def generate(self, params: Dict[str, Any], prompt: str, size: "DallEGenerator.Size", num_images: int) -> List[str]:
        if "api_key" not in params:
            raise HTTPException(status_code=400, detail="OpenAI API key is required")

        # try:
        client = AsyncOpenAI(api_key=params["api_key"])
        response = await client.images.generate(
            prompt=prompt,
            size=f"{size.value[0]}x{size.value[1]}",
            n=num_images,
            response_format="b64_json"
        )
        return [img.b64_json for img in response.data]

        # except Exception as e:
        #     raise HTTPException(status_code=500, detail=f"DALL-E generation failed: {str(e)}")

    def get_required_params(self) -> List[EngineRequirement]:
        return [
            EngineRequirement(
                name="api_key",
                description="OpenAI API key"
            )
        ]
