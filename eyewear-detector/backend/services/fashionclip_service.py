"""FashionCLIP embedding generation and vector DB matching."""

import os
from typing import Optional

import numpy as np
from PIL import Image as PILImage

from adapters.vector_db_base import VectorDBAdapter
from adapters.qdrant_adapter import QdrantAdapter
from utils.image_utils import crop_normalized

# Lazy-load FashionCLIP
_fclip = None
_adapter: Optional[VectorDBAdapter] = None

# Weights for section-based matching
SECTION_WEIGHTS = {
    "frame_rim": 0.3,
    "lens_left": 0.15,
    "lens_right": 0.15,
    "brand_logo": 0.25,
    "full_image": 0.15,
}


def _get_fclip():
    global _fclip
    if _fclip is None:
        from fashion_clip.fashion_clip import FashionCLIP
        model_name = os.getenv("FASHIONCLIP_MODEL", "fashion-clip")
        _fclip = FashionCLIP(model_name)
    return _fclip


def get_adapter() -> VectorDBAdapter:
    global _adapter
    if _adapter is None:
        adapter_name = os.getenv("VECTOR_DB_ADAPTER", "qdrant")
        if adapter_name == "qdrant":
            _adapter = QdrantAdapter()
            _adapter.ensure_collection()
        else:
            raise ValueError(f"Unknown vector DB adapter: {adapter_name}")
    return _adapter


def get_image_embedding(image: PILImage.Image) -> np.ndarray:
    """Generate a normalized 512-dim embedding for an image."""
    fclip = _get_fclip()
    # FashionCLIP expects a list of PIL images
    embeddings = fclip.encode_images([image], batch_size=1)
    embedding = embeddings[0]
    # Normalize
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding


def get_section_embeddings(
    image: PILImage.Image, detections: list[dict]
) -> dict[str, np.ndarray]:
    """
    Crop each detected section and generate embeddings.

    Returns:
        Dict mapping label -> embedding
    """
    embeddings = {}
    for det in detections:
        label = det["label"]
        bbox = det["bbox"]
        crop = crop_normalized(image, bbox)
        # Skip tiny crops
        if crop.size[0] < 10 or crop.size[1] < 10:
            continue
        embeddings[label] = get_image_embedding(crop)
    return embeddings


def match_to_catalog(
    embedding: np.ndarray, top_k: int = 5
) -> list[dict]:
    """Query vector DB for similar products."""
    adapter = get_adapter()
    return adapter.search(embedding, top_k=top_k)


def weighted_match(
    full_embedding: np.ndarray,
    section_embeddings: dict[str, np.ndarray],
    top_k: int = 5,
) -> list[dict]:
    """
    Perform weighted matching using full image + section embeddings.

    Returns ranked matches with individual section scores.
    """
    adapter = get_adapter()

    # Collect all embeddings to query
    queries = {"full_image": full_embedding}
    queries.update(section_embeddings)

    # Average lens embeddings if both present
    if "lens_left" in queries and "lens_right" in queries:
        queries["lens_avg"] = (queries.pop("lens_left") + queries.pop("lens_right")) / 2

    # Get matches for each component
    all_matches: dict[str, dict] = {}  # product_id -> {scores, metadata}

    for component, emb in queries.items():
        weight = SECTION_WEIGHTS.get(component, 0.1)
        results = adapter.search(emb, top_k=top_k * 2)

        for r in results:
            pid = r["id"]
            if pid not in all_matches:
                all_matches[pid] = {
                    "product_id": pid,
                    "metadata": r["metadata"],
                    "weighted_score": 0.0,
                    "component_scores": {},
                }
            all_matches[pid]["component_scores"][component] = r["score"]
            all_matches[pid]["weighted_score"] += r["score"] * weight

    # Sort by weighted score
    ranked = sorted(all_matches.values(), key=lambda x: x["weighted_score"], reverse=True)
    return ranked[:top_k]
