"""Classifies images into one of 4 types using Claude Vision API."""

import json
import os
import re

import anthropic
from PIL import Image as PILImage

from utils.image_utils import image_to_base64

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

CLASSIFICATION_PROMPT = """Classify this image into exactly one of these categories:
- "shelf": Multiple eyewear products displayed on a shelf, rack, or fixture
- "single": One eyewear product on a plain/white/neutral background
- "worn": Eyewear being worn by a person
- "closeup": A close-up detail shot of one part of eyewear (lens, temple, logo, bridge)

Respond ONLY with a JSON object: {"type": "shelf"|"single"|"worn"|"closeup", "confidence": 0.0-1.0}"""


def classify_image(image: PILImage.Image) -> dict:
    """
    Classify an image into one of 4 types.

    Returns:
        {"type": str, "confidence": float}
    """
    b64 = image_to_base64(image)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": CLASSIFICATION_PROMPT},
                ],
            }
        ],
    )

    raw = response.content[0].text.strip()
    # Strip markdown fences
    raw = re.sub(r"```(?:json)?\s*", "", raw)
    raw = raw.strip("`").strip()

    result = json.loads(raw)
    return {
        "type": result["type"],
        "confidence": float(result.get("confidence", 0.0)),
    }
