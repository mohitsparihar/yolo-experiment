"""U2-NetP ONNX inference helper using onnxruntime (CPU)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


MODEL_FILENAME = "u2netp.onnx"


def _normalize(im: Image.Image) -> np.ndarray:
    arr = np.array(im).astype(np.float32)
    arr = arr / max(np.max(arr), 1e-6)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)
    return np.expand_dims(arr, 0).astype(np.float32)


def _resize_with_padding(img: Image.Image, size: int = 320) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    w, h = img.size
    scale = min(size / w, size / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    pad_left = (size - new_w) // 2
    pad_top = (size - new_h) // 2
    pad_right = size - new_w - pad_left
    pad_bottom = size - new_h - pad_top

    padded = Image.new("RGB", (size, size), (0, 0, 0))
    padded.paste(resized, (pad_left, pad_top))

    return padded, (pad_left, pad_top, pad_right, pad_bottom)


def _remove_padding(mask: Image.Image, padding: Tuple[int, int, int, int]) -> Image.Image:
    pad_left, pad_top, pad_right, pad_bottom = padding
    w, h = mask.size
    return mask.crop((pad_left, pad_top, w - pad_right, h - pad_bottom))


@lru_cache(maxsize=2)
def _load_session(model_path: str):
    import onnxruntime as ort

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 1
    sess_opts.inter_op_num_threads = 1

    return ort.InferenceSession(model_path, sess_options=sess_opts, providers=["CPUExecutionProvider"])


def ensure_model_path(model_path: str | None) -> Path:
    if model_path:
        return Path(model_path)
    return Path(__file__).resolve().parent.parent / "models" / MODEL_FILENAME


def u2netp_mask(
    image: Image.Image,
    model_path: str | None = None,
    max_side: int = 640,
) -> np.ndarray:
    """Return a 0-255 mask using U2-NetP ONNX."""
    model_path = ensure_model_path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"U2-NetP model not found at {model_path}. Run scripts/setup_models.py first."
        )

    img_rgb = image.convert("RGB")
    orig_w, orig_h = img_rgb.size

    scale = max_side / max(orig_w, orig_h)
    if scale < 1.0:
        new_w = max(1, int(orig_w * scale))
        new_h = max(1, int(orig_h * scale))
        img_rgb = img_rgb.resize((new_w, new_h), Image.Resampling.LANCZOS)

    padded, padding = _resize_with_padding(img_rgb, size=320)
    input_tensor = _normalize(padded)

    sess = _load_session(str(model_path))
    input_name = sess.get_inputs()[0].name
    pred = sess.run(None, {input_name: input_tensor})[0]

    pred = pred[:, 0, :, :]
    pred = (pred - pred.min()) / max(pred.max() - pred.min(), 1e-6)
    mask = (np.squeeze(pred) * 255.0).astype(np.uint8)

    mask_img = Image.fromarray(mask, mode="L")
    mask_img = _remove_padding(mask_img, padding)
    mask_img = mask_img.resize(img_rgb.size, Image.Resampling.LANCZOS)

    if img_rgb.size != (orig_w, orig_h):
        mask_img = mask_img.resize((orig_w, orig_h), Image.Resampling.LANCZOS)

    return np.array(mask_img)


def u2netp_remove_background(
    image: Image.Image,
    model_path: str | None = None,
    max_side: int = 640,
) -> Image.Image:
    mask = u2netp_mask(image, model_path=model_path, max_side=max_side)
    result = image.convert("RGBA")
    result.putalpha(Image.fromarray(mask))
    return result
