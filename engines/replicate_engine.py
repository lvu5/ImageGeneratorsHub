import os
import replicate
from typing import List
from utils import url_to_base64

# If user didn't set REPLICATE_API_TOKEN, replicate might still work
# if the environment is set up, but let's assume we rely on the token
# being available for this engine. The presence of the engine in ENGINES
# is controlled by the config file.
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "")


async def generate_replicate_images(prompt: str, k: int, width: int, height: int) -> List[str]:
    input_params = {
        "prompt": prompt,
        "num_outputs": k,
        "output_format": "png",
        "go_fast": False
    }

    output_urls = replicate.run("black-forest-labs/flux-schnell", input=input_params)

    encoded_images = []
    for url in output_urls:
        encoded = await url_to_base64(url)
        encoded_images.append(encoded)

    return encoded_images
