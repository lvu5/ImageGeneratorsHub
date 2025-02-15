# app/engines/local.py
from enum import Enum
from typing import List, Dict, Any

import aiohttp
from fastapi import HTTPException

from core.image_generator import ImageGenerator
from models.schemas import EngineRequirement


class LocalGenerator(ImageGenerator):
    class Size(Enum):
        SMALL = (512, 512)
        MEDIUM = (768, 768)
        LARGE = (1024, 1024)

    def __init__(self):
        super().__init__(
            name="Local",
            description="Custom local image generation service"
        )

    def _build_url(self, params: Dict[str, Any]) -> str:
        """Constructs the URL for the local service"""
        host = params.get("host", "").rstrip("/")
        port = params.get("port")
        endpoint = params.get("endpoint", "").lstrip("/")

        if not all([host, port, endpoint]):
            raise HTTPException(
                status_code=400,
                detail="Missing required connection parameters: host, port, or endpoint"
            )

        return f"{host}:{port}/{endpoint}"

    async def generate(self, params: Dict[str, Any], prompt: str, size: "LocalGenerator.Size", num_images: int) -> List[str]:
        url = self._build_url(params)

        request_data = {
            "prompt": prompt,
            "size": f"{size.value[0]}x{size.value[1]}",
            "n": num_images
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=request_data) as response:
                    if response.status != 200:
                        error_detail = await response.text()
                        raise HTTPException(
                            status_code=response.status,
                            detail=f"Local service error: {error_detail}"
                        )

                    data = await response.json()

                    # Assuming the response is a list of base64 encoded images
                    if not isinstance(data, list) or len(data) != num_images:
                        raise HTTPException(
                            status_code=500,
                            detail="Invalid response format from local service"
                        )

                    return data

        except aiohttp.ClientError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to connect to local service: {str(e)}"
            )

    def get_required_params(self) -> List[EngineRequirement]:
        return [
            EngineRequirement(
                name="host",
                description="Host address of the local service (e.g., http://localhost)"
            ),
            EngineRequirement(
                name="port",
                description="Port number of the local service"
            ),
            EngineRequirement(
                name="endpoint",
                description="API endpoint for image generation"
            )
        ]
