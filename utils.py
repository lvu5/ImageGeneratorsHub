import base64
from io import BytesIO

import httpx
from PIL import Image


def pil_image_to_base64(img: Image.Image) -> str:
    """
    Convert a PIL image to a base64-encoded PNG data URL.
    """
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    buffered.seek(0)
    img_bytes = buffered.read()
    b64_img = 'data:image/png;base64,' + base64.b64encode(img_bytes).decode('utf-8')
    return b64_img

async def url_to_base64(url: str) -> str:
    """
    Fetch an image from a URL and convert it to base64-encoded PNG data URL.
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        img_bytes = response.content
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        return pil_image_to_base64(img)
