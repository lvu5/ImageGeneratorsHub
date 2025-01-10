import base64

import requests


def img_to_base64(img: bytes) -> str:
    """
    Converts Image binary data to a Base64-encoded string.

    Args:
        img (bytes): Binary Image data.

    Returns:
        str: Base64-encoded image string.
    """
    return base64.b64encode(img).decode('utf-8')


def url_to_base64(url: str) -> str:
    """
    Downloads an image from a URL and converts it to a Base64-encoded string.

    Args:
        url (str): URL of the image.

    Returns:
        str: Base64-encoded image string.
    """
    response = requests.get(url)
    response.raise_for_status()
    return img_to_base64(response.content)
