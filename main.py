from fastapi import FastAPI
from typing import List

from engines.dalle import DallEGenerator
from engines.local import LocalGenerator
from engines.replicate import ReplicateGenerator, RealVisXL, Imagen3
from models.schemas import GenerationRequest, GenerationResponse, EngineInfo
from services.hub import ImageGeneratorHub

app = FastAPI(title="ImageGeneratorHub")
hub = ImageGeneratorHub()


@app.on_event("startup")
async def startup_event():
    hub.register_engine(DallEGenerator())
    hub.register_engine(ReplicateGenerator())
    hub.register_engine(RealVisXL())
    hub.register_engine(Imagen3())
    hub.register_engine(LocalGenerator())


@app.get("/engines", response_model=List[EngineInfo])
async def list_engines():
    """List all available image generation engines and their requirements"""
    return hub.get_available_engines()


@app.post("/generate", response_model=GenerationResponse)
async def generate_images(request: GenerationRequest):
    """Generate images using specified engines with provided credentials"""
    return await hub.generate_images(request)
