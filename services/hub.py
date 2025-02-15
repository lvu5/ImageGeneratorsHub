from typing import List, Dict, Tuple

from fastapi import HTTPException

from core.image_generator import ImageGenerator
from models.schemas import (
    EngineInfo,
    GenerationRequest,
    GenerationResponse,
    GeneratedImage
)


class ImageGeneratorHub:
    def __init__(self):
        self.engines: Dict[str, ImageGenerator] = {}

    def register_engine(self, engine: ImageGenerator):
        self.engines[engine.name] = engine

    def get_available_engines(self) -> List[EngineInfo]:
        return [
            EngineInfo(
                name=engine.name,
                description=engine.description,
                required_params=engine.get_required_params()
            )
            for engine in self.engines.values()
        ]

    async def _try_generate_with_engine(
            self,
            engine: ImageGenerator,
            config: dict,
            size: str,
            num_images: int
    ) -> Tuple[bool, List[GeneratedImage]]:
        """
        Attempts to generate images with a single engine.
        Returns (success, images) tuple.
        """
        try:
            base64_images = await engine.generate(
                params=config["params"],
                prompt=config["prompt"],
                size=engine.convert_size(size),
                num_images=num_images
            )

            return True, [
                GeneratedImage(
                    engine_name=engine.name,
                    base64_image=base64_image
                )
                for base64_image in base64_images
            ]
        except Exception as e:
            return False, []

    async def _generate_with_redistribution(
            self,
            engine_configs: List[dict],
            size: str,
            total_images: int,
            images_per_engine: int,
            num_engines_to_use: int,
            use_fallback: bool
    ) -> Tuple[List[GeneratedImage], List[str]]:
        """
        Generates images with fallback and load redistribution.
        Returns (generated_images, failed_engines).
        """
        successful_images = []
        failed_engines = []
        remaining_images = total_images

        # First attempt: Try with requested number of engines
        for config in engine_configs[:num_engines_to_use]:
            if remaining_images <= 0:
                break

            engine = self.engines.get(config["name"])
            if not engine:
                failed_engines.append(config["name"])
                if not use_fallback:
                    raise HTTPException(status_code=400, detail=f"Engine {config['name']} not found")
                continue

            success, images = await self._try_generate_with_engine(
                engine=engine,
                config=config,
                size=size,
                num_images=images_per_engine
            )

            if success:
                successful_images.extend(images)
                remaining_images -= images_per_engine
            else:
                failed_engines.append(config["name"])
                if not use_fallback:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Engine {config['name']} failed to generate images"
                    )

        # If we need fallback
        if use_fallback and remaining_images > 0:
            # Get available engines excluding the failed ones
            available_configs = [
                config for config in engine_configs
                if config["name"] not in failed_engines
            ]

            # Try each available engine until we get all images
            for config in available_configs:
                if remaining_images <= 0:
                    break

                engine = self.engines.get(config["name"])
                if not engine:
                    failed_engines.append(config["name"])
                    continue

                success, images = await self._try_generate_with_engine(
                    engine=engine,
                    config=config,
                    size=size,
                    num_images=remaining_images  # Try to generate all remaining images
                )

                if success:
                    successful_images.extend(images)
                    remaining_images = 0
                    break
                else:
                    failed_engines.append(config["name"])

        return successful_images, failed_engines

    async def generate_images(self, request: GenerationRequest) -> GenerationResponse:
        if request.num_engines_to_use > len(request.engines):
            raise HTTPException(
                status_code=400,
                detail="num_engines_to_use cannot be greater than number of provided engines"
            )

        total_images = request.num_engines_to_use * request.num_images
        images_per_engine = request.num_images

        generated_images, failed_engines = await self._generate_with_redistribution(
            engine_configs=[config.model_dump() for config in request.engines],
            size=request.image_size.value,
            total_images=total_images,
            images_per_engine=images_per_engine,
            num_engines_to_use=request.num_engines_to_use,
            use_fallback=request.use_fallback
        )

        if not generated_images:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate images with all available engines"
            )

        if len(generated_images) < total_images:
            if not request.use_fallback:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to generate requested number of images"
                )

        return GenerationResponse(
            images=generated_images,
            failed_engines=failed_engines
        )
