"""Abstract adapter interface for vector database."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class VectorDBAdapter(ABC):
    """Base class for vector database adapters. Subclass this to use a custom vector DB."""

    @abstractmethod
    def search(self, embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        """
        Search for similar embeddings.

        Args:
            embedding: Query embedding vector (512-dim for FashionCLIP)
            top_k: Number of results to return

        Returns:
            List of dicts with keys: id, score, metadata
        """
        ...

    @abstractmethod
    def upsert(self, id: str, embedding: np.ndarray, metadata: dict) -> None:
        """Insert or update an embedding with metadata."""
        ...

    @abstractmethod
    def delete(self, id: str) -> None:
        """Delete an embedding by id."""
        ...

    @abstractmethod
    def ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
        ...
