from abc import ABC, abstractmethod
from typing import List, Dict, Any

from models.schemas import EngineRequirement


class ImageGenerator(ABC):
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    async def generate(self, params: Dict[str, Any], prompt: str, size: int, num_images: int) -> List[str]:
        """Generate images and return them as base64 strings"""
        pass

    @abstractmethod
    def get_required_params(self) -> List[EngineRequirement]:
        """Return list of required parameters for this engine"""
        pass
