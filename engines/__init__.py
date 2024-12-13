from config import OPENAI_API_KEY, REPLICATE_API_TOKEN

ENGINES = {
    # Local engines are always available
    "runwayml": {
        "local": True,
        "needs_gpu": True,
        "description": "Stable Diffusion v1.5 (RunwayML model)"
    },
    "sdxl_turbo": {
        "local": True,
        "needs_gpu": True,
        "description": "SDXL Turbo"
    },
    "sdxl_lightning": {
        "local": True,
        "needs_gpu": True,
        "description": "SDXL Lightning custom UNet"
    },
}

# Only add replicate engine if replicate token is set
if REPLICATE_API_TOKEN:
    ENGINES["replicate_flux_schnell"] = {
        "local": False,
        "needs_gpu": False,
        "description": "Replicate model: black-forest-labs/flux-schnell"
    }

# Only add DALL·E if OpenAI API key is set
if OPENAI_API_KEY:
    ENGINES["dall-e"] = {
        "local": False,
        "needs_gpu": False,
        "description": "DALL·E via OpenAI API"
    }
