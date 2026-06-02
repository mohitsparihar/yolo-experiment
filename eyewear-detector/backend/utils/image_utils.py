"""Image utility functions: crop, resize, draw, base64 helpers."""

import base64
import io
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# Color map for section classes
CLASS_COLORS = {
    0: (255, 0, 0),       # frame_rim - red
    1: (0, 255, 0),       # lens_left - green
    2: (0, 0, 255),       # lens_right - blue
    3: (255, 255, 0),     # nose_bridge - yellow
    4: (255, 0, 255),     # temple_left - magenta
    5: (0, 255, 255),     # temple_right - cyan
    6: (255, 128, 0),     # brand_logo - orange
}

SECTION_CLASSES = {
    0: "frame_rim",
    1: "lens_left",
    2: "lens_right",
    3: "nose_bridge",
    4: "temple_left",
    5: "temple_right",
    6: "brand_logo",
}

PRODUCT_CLASSES = {
    0: "eyewear_product",
}


def image_to_base64(img: Image.Image, format: str = "JPEG") -> str:
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def base64_to_image(b64: str) -> Image.Image:
    """Convert base64 string to PIL Image."""
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data))


def crop_normalized(img: Image.Image, bbox: list[float], padding: float = 0.0) -> Image.Image:
    """Crop image using normalized [x1, y1, x2, y2] coordinates with optional padding."""
    w, h = img.size
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    x1 = max(0, x1 - bw * padding) * w
    y1 = max(0, y1 - bh * padding) * h
    x2 = min(1, x2 + bw * padding) * w
    y2 = min(1, y2 + bh * padding) * h
    return img.crop((int(x1), int(y1), int(x2), int(y2)))


_rembg_session = None
_REMBG_MAX_SIZE = 320


def remove_background(image: Image.Image) -> Image.Image:
    """Remove background using rembg (u2netp — smallest/fastest model)."""
    from rembg import remove as rembg_remove, new_session
    global _rembg_session
    if _rembg_session is None:
        _rembg_session = new_session("u2netp")

    w, h = image.size
    max_dim = max(w, h)
    if max_dim > _REMBG_MAX_SIZE:
        scale = _REMBG_MAX_SIZE / max_dim
        small = image.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
        small_result = rembg_remove(small, session=_rembg_session, post_process_mask=True)
        alpha = small_result.split()[-1].resize((w, h), Image.LANCZOS)
        result = image.convert("RGBA")
        result.putalpha(alpha)
        return result
    return rembg_remove(image, session=_rembg_session, post_process_mask=True)


def draw_detections(img: Image.Image, detections: list[dict]) -> Image.Image:
    """Draw bounding boxes and labels on image. Returns annotated copy."""
    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)
    w, h = img.size

    for det in detections:
        class_id = det.get("class_id", 0)
        color = CLASS_COLORS.get(class_id, (255, 255, 255))
        bbox = det["bbox"]
        x1, y1, x2, y2 = bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h

        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = det.get("label", "unknown")
        conf = det.get("confidence", 0.0)
        text = f"{label} {conf:.2f}"
        draw.text((x1, y1 - 12), text, fill=color)

    return annotated


def nms(detections: list[dict], iou_threshold: float = 0.5) -> list[dict]:
    """Non-maximum suppression on detections with normalized bboxes."""
    if not detections:
        return []

    # Sort by confidence descending
    dets = sorted(detections, key=lambda d: d.get("confidence", 0), reverse=True)
    keep = []

    while dets:
        best = dets.pop(0)
        keep.append(best)
        dets = [d for d in dets if _iou(best["bbox"], d["bbox"]) < iou_threshold]

    return keep


def _iou(box1: list[float], box2: list[float]) -> float:
    """Compute IoU between two [x1, y1, x2, y2] normalized boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def bbox_xyxy_to_cxcywh(bbox: list[float]) -> list[float]:
    """Convert [x1, y1, x2, y2] to [cx, cy, w, h] (all normalized)."""
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return [cx, cy, w, h]
