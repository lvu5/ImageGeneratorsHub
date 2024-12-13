from typing import List

from openai import AsyncOpenAI
from config import OPENAI_API_KEY
from utils import url_to_base64

# Only instantiate client if OPENAI_API_KEY is set
aclient = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


async def generate_dalle_images(prompt: str, k: int, width: int, height: int) -> List[str]:
    if aclient is None:
        raise ValueError("OpenAI API Key not provided, DALL·E engine not available.")

    size_str = f"{width}x{height}"
    response = await aclient.images.generate(prompt=prompt, n=k, size=size_str)
    urls = [item["url"] for item in response["data"]]

    encoded_images = []
    for url in urls:
        encoded = await url_to_base64(url)
        encoded_images.append(encoded)

    return encoded_images
