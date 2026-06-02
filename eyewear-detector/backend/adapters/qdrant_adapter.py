"""Qdrant vector database adapter implementation."""

import os
from typing import Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from adapters.vector_db_base import VectorDBAdapter

EMBEDDING_DIM = 512  # FashionCLIP output dimension


class QdrantAdapter(VectorDBAdapter):
    def __init__(
        self,
        url: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        self.url = url or os.getenv("VECTOR_DB_URL", "http://localhost:6333")
        self.collection_name = collection_name or os.getenv("VECTOR_DB_COLLECTION", "eyewear_catalog")
        self.client = QdrantClient(url=self.url)

    def ensure_collection(self) -> None:
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE,
                ),
            )

    def search(self, embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=embedding.tolist(),
            limit=top_k,
        )
        return [
            {
                "id": str(hit.id),
                "score": hit.score,
                "metadata": hit.payload or {},
            }
            for hit in results
        ]

    def upsert(self, id: str, embedding: np.ndarray, metadata: dict) -> None:
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=id,
                    vector=embedding.tolist(),
                    payload=metadata,
                )
            ],
        )

    def delete(self, id: str) -> None:
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=[id],
        )
