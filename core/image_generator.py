from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Any, Type

from models.schemas import EngineRequirement


class ImageGenerator(ABC):
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Ensure each subclass defines a nested 'Size' enum
        if not hasattr(cls, 'Size'):
            raise TypeError(f"{cls.__name__} must define a nested 'Size' Enum")
        size_enum = getattr(cls, 'Size')
        if not issubclass(size_enum, Enum):
            raise TypeError(f"{cls.__name__}.Size must be an Enum")
        # Enforce that the enum has exactly SMALL, MEDIUM, and LARGE
        required_members = {"SMALL", "MEDIUM", "LARGE"}
        if set(size_enum.__members__.keys()) != required_members:
            raise TypeError(f"{cls.__name__}.Size must have exactly the members: {required_members}")

    def convert_size(self, size_value: str) -> Enum:
        """
        Converts the given size value (expected to be a string: 'small', 'medium', or 'large')
        to the corresponding member of the child class's Size enum.
        """
        # If the size_value is already an enum member, return it as is.
        if isinstance(size_value, self.__class__.Size):
            return size_value
        try:
            # Convert input to uppercase to match enum member names (e.g., 'small' -> 'SMALL')
            return self.__class__.Size[size_value.upper()]
        except KeyError as e:
            raise ValueError(
                f"Invalid size value: {size_value}. Expected one of: small, medium, large"
            ) from e

    @abstractmethod
    async def generate(self, params: Dict[str, Any], prompt: str, size: Enum, num_images: int) -> List[str]:
        """Generate images and return them as base64 strings"""
        pass

    @abstractmethod
    def get_required_params(self) -> List[EngineRequirement]:
        """Return list of required parameters for this engine"""
        pass
