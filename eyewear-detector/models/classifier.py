"""CLIP-based image type classifier — detects if image is 'shelf' or 'single product'."""

import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

os.environ.setdefault("TRANSFORMERS_CACHE", "./hf_cache")

MODEL_NAME = "openai/clip-vit-base-patch32"

PROMPTS = [
    "multiple sunglasses displayed on a retail shelf or fixture",  # → shelf
    "a single pair of glasses on a plain or white background",     # → single
]

LABELS = ["shelf", "single"]

# Module-level singleton
_model = None
_processor = None
_device = None


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_classifier():
    """Load CLIP model and processor (singleton)."""
    global _model, _processor, _device
    if _model is None:
        _device = _get_device()
        print(f"[INFO] Loading CLIP classifier on: {_device}")
        _model = CLIPModel.from_pretrained(MODEL_NAME).to(_device)
        _processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        _model.eval()
    return _model, _processor, _device


def classify_image(image: Image.Image) -> dict:
    """
    Classify an image as 'shelf' or 'single' using CLIP zero-shot.

    Args:
        image: PIL Image to classify.

    Returns:
        Dict with 'image_type' ('shelf' or 'single') and 'confidence' (float).
    """
    model, processor, device = load_classifier()

    inputs = processor(
        text=PROMPTS,
        images=image,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits_per_image[0]  # shape: [num_prompts]
        probs = logits.softmax(dim=-1).cpu().numpy()

    best_idx = int(probs.argmax())
    return {
        "image_type": LABELS[best_idx],
        "confidence": float(probs[best_idx]),
    }
