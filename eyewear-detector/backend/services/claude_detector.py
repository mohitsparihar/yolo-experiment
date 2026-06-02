"""Claude Vision API calls for eyewear detection."""

import json
import os
import re
from typing import Optional

import anthropic
from PIL import Image as PILImage

from utils.image_utils import image_to_base64

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SECTION_DETECTION_PROMPT = """You are an eyewear component detector. Analyze this eyewear product image.
Detect as many of these sections as are clearly visible:
frame_rim, lens_left, lens_right, nose_bridge, temple_left, temple_right, brand_logo.

Return ONLY a JSON object — no markdown, no explanation:
{
  "detections": [
    {"label": "frame_rim", "class_id": 0, "bbox": [x1, y1, x2, y2], "confidence": 0.95},
    ...
  ]
}
Coordinates must be normalized floats between 0 and 1.
Omit any section that is not clearly visible. Do not guess."""

PRODUCT_DETECTION_PROMPT = """This is a retail shelf with multiple eyewear products. Detect every individual pair of glasses/sunglasses.
Return ONLY JSON: {"products": [{"bbox": [x1,y1,x2,y2], "confidence": 0.9}, ...]}
Coordinates normalized 0-1. Do not merge adjacent products."""

EYEWEAR_ISOLATION_PROMPT = """This image shows a person wearing eyewear. Detect ONLY the eyewear (glasses or sunglasses) worn on their face.
Return ONLY JSON: {"eyewear_bbox": [x1,y1,x2,y2], "confidence": 0.0-1.0}
Coordinates normalized 0-1."""

CLOSEUP_PART_PROMPT = """This is a close-up image of part of an eyewear product. Identify which part it shows.
Return ONLY JSON: {"part": "lens"|"frame_rim"|"temple"|"nose_bridge"|"brand_logo"|"full_frame"|"unknown", "confidence": 0.0-1.0}"""


def _call_claude_vision(image: PILImage.Image, prompt: str, max_retries: int = 2) -> dict:
    """Call Claude Vision with an image and prompt, with retry logic."""
    b64 = image_to_base64(image)

    for attempt in range(max_retries + 1):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
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
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )

            raw = response.content[0].text.strip()
            # Strip markdown fences
            raw = re.sub(r"```(?:json)?\s*", "", raw)
            raw = raw.strip("`").strip()
            return json.loads(raw)

        except (json.JSONDecodeError, IndexError, KeyError) as e:
            if attempt == max_retries:
                raise ValueError(f"Failed to parse Claude response after {max_retries + 1} attempts: {e}")
            continue


def detect_sections(image: PILImage.Image) -> list[dict]:
    """Detect eyewear sections in a single product image."""
    result = _call_claude_vision(image, SECTION_DETECTION_PROMPT)
    detections = result.get("detections", [])
    # Filter low confidence
    return [d for d in detections if d.get("confidence", 0) >= 0.5]


def detect_products(image: PILImage.Image) -> list[dict]:
    """Detect individual eyewear products on a shelf."""
    result = _call_claude_vision(image, PRODUCT_DETECTION_PROMPT)
    products = result.get("products", [])
    return [p for p in products if p.get("confidence", 0) >= 0.5]


def detect_eyewear_on_person(image: PILImage.Image) -> dict:
    """Detect eyewear region on a person's face."""
    return _call_claude_vision(image, EYEWEAR_ISOLATION_PROMPT)


def classify_closeup_part(image: PILImage.Image) -> dict:
    """Classify which part a close-up image shows."""
    return _call_claude_vision(image, CLOSEUP_PART_PROMPT)
