import asyncio
from typing import Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from config import FALLBACK_ENGINE, OPENAI_API_KEY, REPLICATE_API_TOKEN
from engines import ENGINES
from engines.dalle import generate_dalle_images
from engines.replicate_engine import generate_replicate_images
from engines.stable_diffusion import generate_sd_images

app = FastAPI()


class EngineRequest(BaseModel):
    k: int = Field(..., description="Number of images to generate")
    image_size: str = Field(..., description="Image size, e.g. 512x512")


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Prompt to generate images from.")
    engines: Dict[str, EngineRequest]


@app.get("/engines")
async def list_engines():
    return [
        {
            "name": name,
            "local": info["local"],
            "needs_gpu": info["needs_gpu"],
            "description": info.get("description", "")
        }
        for name, info in ENGINES.items()
    ]


@app.post("/generate")
async def generate_images(body: GenerateRequest):
    requests = body.engines
    prompt = body.prompt

    # Prepare tasks for primary engines
    tasks = []
    engine_names = list(requests.keys())

    for engine_name, params in requests.items():
        if engine_name not in ENGINES:
            raise HTTPException(status_code=400, detail=f"Engine '{engine_name}' not available.")

        # Parse image_size
        try:
            w, h = params.image_size.lower().split('x')
            w = int(w)
            h = int(h)
        except:
            raise HTTPException(status_code=400, detail=f"Invalid image_size for {engine_name}: {params.image_size}")

        k = params.k

        # Create tasks depending on engine type
        if ENGINES[engine_name]["local"]:
            tasks.append(_sd_task(engine_name, prompt, k, w, h))
        elif engine_name == "replicate_flux_schnell":
            if not REPLICATE_API_TOKEN:
                raise HTTPException(status_code=400, detail="Replicate API key not provided, engine unavailable.")
            tasks.append(generate_replicate_images(prompt, k, w, h))
        elif engine_name == "dall-e":
            if not OPENAI_API_KEY:
                raise HTTPException(status_code=400, detail="OpenAI API key not provided, engine unavailable.")
            tasks.append(generate_dalle_images(prompt, k, w, h))
        else:
            raise HTTPException(status_code=400, detail=f"Engine '{engine_name}' not supported.")

    # Run all tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    response = {}
    fallback_needed = False
    failed_engines = []

    # Process results
    for engine_name, result in zip(engine_names, results):
        if isinstance(result, Exception):
            # If an engine fails, mark fallback needed
            fallback_needed = True
            failed_engines.append(engine_name)
        else:
            response[engine_name] = {"images": result}

    # If fallback needed and fallback engine is specified
    if fallback_needed and FALLBACK_ENGINE:
        # Ensure fallback engine is available
        if FALLBACK_ENGINE not in ENGINES:
            raise HTTPException(status_code=500, detail=f"Fallback engine '{FALLBACK_ENGINE}' not available.")

        # Generate fallback images for failed engines
        # Using the same prompt, we can decide a default size if we want.
        # Let's pick a default of 512x512 or we can reuse the failed engine's requested size.
        fallback_tasks = []
        fallback_engine_requests = []
        for fe in failed_engines:
            # Use the same size and k as requested by that engine for consistency
            eng_req = requests[fe]
            w, h = map(int, eng_req.image_size.lower().split('x'))

            if ENGINES[FALLBACK_ENGINE]["local"]:
                fallback_tasks.append(_sd_task(FALLBACK_ENGINE, prompt, eng_req.k, w, h))
            elif FALLBACK_ENGINE == "replicate_flux_schnell":
                fallback_tasks.append(generate_replicate_images(prompt, eng_req.k, w, h))
            elif FALLBACK_ENGINE == "dall-e":
                fallback_tasks.append(generate_dalle_images(prompt, eng_req.k, w, h))
            else:
                raise HTTPException(status_code=500, detail=f"Fallback engine '{FALLBACK_ENGINE}' not supported.")

            fallback_engine_requests.append(fe)

        fallback_results = await asyncio.gather(*fallback_tasks, return_exceptions=True)

        for fe, fres in zip(fallback_engine_requests, fallback_results):
            if isinstance(fres, Exception):
                # Fallback engine also failed, return error
                raise HTTPException(status_code=500,
                                    detail=f"Failed to generate images with fallback engine '{FALLBACK_ENGINE}' for '{fe}'. Original error: {fres}")
            else:
                response[fe] = {"images": fres}
    elif fallback_needed and not FALLBACK_ENGINE:
        # No fallback engine defined, just raise an error
        raise HTTPException(status_code=500,
                            detail=f"Some engines failed: {failed_engines}, no fallback engine provided.")

    return response


def _sd_task(engine_name, prompt, k, w, h):
    # Wrap stable diffusion in run_in_threadpool
    async def run():
        def _run_sync():
            return generate_sd_images(engine_name, prompt, k, w, h)

        return await run_in_threadpool(_run_sync)

    return run()
