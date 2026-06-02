"""Abstract base pipeline for eyewear detection."""

from abc import ABC, abstractmethod

from PIL import Image as PILImage


class BasePipeline(ABC):
    """Base class for all detection pipelines."""

    @abstractmethod
    def run(self, image: PILImage.Image, image_id: str) -> dict:
        """
        Run the detection pipeline on an image.

        Args:
            image: PIL Image to process
            image_id: UUID of the image in the database

        Returns:
            Detection result dict matching the unified response format
        """
        ...
