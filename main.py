import gc
from io import BytesIO

import torch
# from diffusers import StableDiffusionPipeline
from diffusers import AutoPipelineForText2Image
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

app = FastAPI()

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to(
    "cuda")


@app.post("/generate")
async def generate_image(prompt: str, size: int = 1024):
    try:
        gc.collect()
        torch.cuda.empty_cache()
        image = pipe(prompt, width=size, height=size).images[0]
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        buffered.seek(0)

        return StreamingResponse(buffered, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

