
import asyncio
import base64
import io
from enum import Enum
from typing import List, Dict, Any

from core.image_generator import ImageGenerator
from models.schemas import EngineRequirement


def get_sd_turbo_model():
    from diffusers import AutoPipelineForText2Image
    import torch

    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=torch.float16,
                                                     variant="fp16")
    # checking if cuda is available before moving to cuda
    if torch.cuda.is_available():
        pipe.to("cuda")
    return pipe


class SDTurboGenerator(ImageGenerator):
    class Size(Enum):
        SMALL = (512, 512)
        MEDIUM = (768, 768)
        LARGE = (1024, 1024)

    def __init__(self):
        super().__init__(
            name="SDTurbo",
            description="Image generator using SD-Turbo"
        )
        self.pipe = get_sd_turbo_model()

    async def generate(self, params: Dict[str, Any], prompt: str, size: "SDTurboGenerator.Size",
                       num_images: int) -> List[str]:
        loop = asyncio.get_running_loop()
        width, height = size.value

        def run_pipeline() -> List[str]:
            images = []
            results = self.pipe(
                prompt,
                height=height,
                width=width,
                num_inference_steps=1,
                guidance_scale=0.0,
                num_images_per_prompt=num_images,
            )
            for img in results.images:
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                base64_img = base64.b64encode(buffered.getvalue()).decode("utf-8")
                images.append(base64_img)
            return images

        return await loop.run_in_executor(None, run_pipeline)

    def get_required_params(self) -> List[EngineRequirement]:
        return []
