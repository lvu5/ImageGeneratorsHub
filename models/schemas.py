# app/models/schemas.py
from enum import Enum
from typing import Dict, Any
from typing import List

from pydantic import BaseModel, Field



class EngineRequirement(BaseModel):
    name: str
    description: str


class EngineInfo(BaseModel):
    name: str
    description: str
    required_params: List[EngineRequirement]


class EngineConfig(BaseModel):
    name: str
    params: Dict[str, Any] = Field(..., description="Required parameters including auth")
    prompt: str


class GenerationRequest(BaseModel):
    engines: List[EngineConfig]
    num_engines_to_use: int = Field(..., description="Number of engines to use")
    num_images: int = Field(1, description="Number of images to generate per engine")
    image_size: int = Field(512, description="Image size in pixels")
    use_fallback: bool = Field(False, description="Whether to use fallback engines on failure")


class GeneratedImage(BaseModel):
    engine_name: str
    base64_image: str


class GenerationResponse(BaseModel):
    images: List[GeneratedImage] = Field(default_factory=list)
    failed_engines: List[str] = Field(default_factory=list)
