"""Image utility functions: crop, draw boxes, save helpers."""

import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np


# Colors for each section type (RGB)
SECTION_COLORS = {
    "frame_rim":    (78, 205, 196),
    "lens_left":    (69, 183, 209),
    "lens_right":   (150, 206, 180),
    "nose_bridge":  (255, 234, 167),
    "temple_left":  (221, 160, 221),
    "temple_right": (152, 216, 200),
    "brand_logo":   (247, 220, 111),
    "eyewear_product": (255, 107, 107),
}


def crop_normalized(image: Image.Image, bbox: list[float], padding: float = 0.0,
                    padding_x: float = None, padding_y: float = None) -> Image.Image:
    """
    Crop an image using normalized [x1, y1, x2, y2] coordinates.

    Args:
        image: PIL Image to crop.
        bbox: Normalized bounding box [x1, y1, x2, y2] in range [0, 1].
        padding: Uniform padding as fraction of box size (ignored if padding_x/y set).
        padding_x: Horizontal padding override (for temples extending sideways).
        padding_y: Vertical padding override.

    Returns:
        Cropped PIL Image.
    """
    w, h = image.size
    x1, y1, x2, y2 = bbox

    px_pad = padding_x if padding_x is not None else padding
    py_pad = padding_y if padding_y is not None else padding

    bw = x2 - x1
    bh = y2 - y1
    x1 = max(0, x1 - bw * px_pad)
    y1 = max(0, y1 - bh * py_pad)
    x2 = min(1, x2 + bw * px_pad)
    y2 = min(1, y2 + bh * py_pad)

    px1 = int(x1 * w)
    py1 = int(y1 * h)
    px2 = int(x2 * w)
    py2 = int(y2 * h)

    return image.crop((px1, py1, px2, py2))


def draw_boxes(image: Image.Image, detections: list[dict], line_width: int = 3) -> Image.Image:
    """
    Draw bounding boxes with labels on an image.

    Args:
        image: PIL Image to annotate.
        detections: List of dicts with 'label', 'bbox', and optionally 'confidence'.
        line_width: Width of box outline.

    Returns:
        Annotated PIL Image copy.
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for det in detections:
        label = det["label"]
        bbox = det["bbox"]
        conf = det.get("confidence", None)
        color = SECTION_COLORS.get(label, (255, 255, 255))

        # Convert normalized to pixel
        px1 = int(bbox[0] * w)
        py1 = int(bbox[1] * h)
        px2 = int(bbox[2] * w)
        py2 = int(bbox[3] * h)

        draw.rectangle([px1, py1, px2, py2], outline=color, width=line_width)

        # Label text
        text = label
        if conf is not None:
            text += f" {conf:.0%}"

        # Draw text background
        text_bbox = draw.textbbox((px1, py1 - 18), text, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((px1, py1 - 18), text, fill=(0, 0, 0), font=font)

    return img


def remove_background_fast(image: Image.Image) -> Image.Image:
    """Trim background strips from edges by detecting rows/columns that are pure background.
    Instant. Preserves temples and frames since it only removes uniform edge bands."""
    img_array = np.array(image.convert("RGB"))
    h, w = img_array.shape[:2]

    # Sample corner colors to learn background
    cs = max(4, int(min(w, h) * 0.05))
    corners = np.concatenate([
        img_array[:cs, :cs].reshape(-1, 3),
        img_array[:cs, w - cs:].reshape(-1, 3),
        img_array[h - cs:, :cs].reshape(-1, 3),
        img_array[h - cs:, w - cs:].reshape(-1, 3),
    ], axis=0).astype(np.float32)

    bg_mean = corners.mean(axis=0)
    bg_std = corners.std(axis=0).clip(min=10)

    # Per-pixel distance from bg color (normalized)
    diff = np.abs(img_array.astype(np.float32) - bg_mean)
    dist = (diff / bg_std).mean(axis=2)
    is_bg = dist < 2.5  # True where pixel looks like background

    # A row/col is "pure background" only if nearly all pixels are bg-like
    row_bg_ratio = is_bg.mean(axis=1)
    col_bg_ratio = is_bg.mean(axis=0)
    BG_THRESHOLD = 0.99  # stricter — stop as soon as any non-bg content appears

    # Find top trim
    top = 0
    while top < h and row_bg_ratio[top] > BG_THRESHOLD:
        top += 1

    # Find bottom trim
    bottom = h
    while bottom > top and row_bg_ratio[bottom - 1] > BG_THRESHOLD:
        bottom -= 1

    # Find left/right trim
    left = 0
    while left < w and col_bg_ratio[left] > BG_THRESHOLD:
        left += 1

    right = w
    while right > left and col_bg_ratio[right - 1] > BG_THRESHOLD:
        right -= 1

    # Safety: don't trim more than 25% from any side (preserves temples near edges)
    top = min(top, int(h * 0.25))
    bottom = max(bottom, int(h * 0.75))
    left = min(left, int(w * 0.25))
    right = max(right, int(w * 0.75))

    return image.crop((left, top, right, bottom))


_rembg_session = None
_REMBG_MAX_SIZE = 1024


def remove_background_rembg(image: Image.Image) -> Image.Image:
    """Accurate background removal using rembg (BiRefNet-general — best accuracy)."""
    from rembg import remove as rembg_remove, new_session
    global _rembg_session
    if _rembg_session is None:
        print("[INFO] Loading rembg model: birefnet-general")
        _rembg_session = new_session("birefnet-general")

    w, h = image.size
    max_dim = max(w, h)
    if max_dim > _REMBG_MAX_SIZE:
        scale = _REMBG_MAX_SIZE / max_dim
        small = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        small_result = rembg_remove(small, session=_rembg_session, post_process_mask=True)
        alpha = small_result.split()[-1].resize((w, h), Image.LANCZOS)
        result = image.convert("RGBA")
        result.putalpha(alpha)
        return result
    return rembg_remove(image, session=_rembg_session, post_process_mask=True)


def remove_background(image: Image.Image) -> Image.Image:
    """Default: fast GrabCut."""
    return remove_background_fast(image)


def save_crop(image: Image.Image, bbox: list[float], output_path: str, padding: float = 0.02, remove_bg: bool = False) -> str:
    """
    Crop and save a section of an image.

    Args:
        image: Source PIL Image.
        bbox: Normalized [x1, y1, x2, y2] bounding box.
        output_path: Path to save the cropped image.
        padding: Extra padding around the crop.
        remove_bg: Whether to remove the background (saves as PNG).

    Returns:
        The output path where the crop was saved.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    crop = crop_normalized(image, bbox, padding=padding)
    if remove_bg:
        crop = remove_background(crop)
    if crop.mode == "RGBA":
        crop.save(output_path, "PNG")
    else:
        crop.save(output_path, "JPEG", quality=95)
    return output_path
