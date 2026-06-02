"""Fast OpenCV-based background removal tuned for product shots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from PIL import Image


@dataclass
class OpenCVConfig:
    border_frac: float = 0.03
    grabcut_iters: int = 5
    color_threshold: float = 18.0
    min_border: int = 5


def _border_mask(h: int, w: int, border: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[:border, :] = True
    mask[h - border :, :] = True
    mask[:, :border] = True
    mask[:, w - border :] = True
    return mask


def _estimate_bg_color_lab(img_bgr: np.ndarray, border: int) -> Tuple[np.ndarray, float]:
    import cv2

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    h, w = lab.shape[:2]
    bmask = _border_mask(h, w, border)
    samples = lab[bmask]
    if samples.size == 0:
        samples = lab.reshape(-1, 3)
    median = np.median(samples, axis=0)
    mad = np.median(np.abs(samples - median), axis=0)
    spread = float(np.mean(mad))
    return median, max(spread, 1.0)


def _initial_fg_mask(img_bgr: np.ndarray, cfg: OpenCVConfig) -> np.ndarray:
    import cv2

    h, w = img_bgr.shape[:2]
    border = max(cfg.min_border, int(min(h, w) * cfg.border_frac))
    bg_lab, spread = _estimate_bg_color_lab(img_bgr, border)

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    dist = np.linalg.norm(lab - bg_lab.reshape(1, 1, 3), axis=2)

    thresh = max(cfg.color_threshold, spread * 2.5)
    fg = dist > thresh

    # Ensure border stays background
    fg[_border_mask(h, w, border)] = False
    return (fg * 255).astype(np.uint8)


def _refine_mask(mask: np.ndarray) -> np.ndarray:
    import cv2

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    refined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel, iterations=1)
    refined = cv2.GaussianBlur(refined, (5, 5), 1.2)
    return refined


def opencv_remove_background(image: Image.Image, cfg: OpenCVConfig | None = None) -> Image.Image:
    """Remove background using a fast OpenCV pipeline + GrabCut refinement."""
    import cv2

    if cfg is None:
        cfg = OpenCVConfig()

    img_rgb = np.array(image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    init_mask = _initial_fg_mask(img_bgr, cfg)

    # Build GrabCut mask
    gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    gc_mask[init_mask > 0] = cv2.GC_PR_FGD

    border = max(cfg.min_border, int(min(h, w) * cfg.border_frac))
    gc_mask[_border_mask(h, w, border)] = cv2.GC_BGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(img_bgr, gc_mask, None, bgd_model, fgd_model, cfg.grabcut_iters, cv2.GC_INIT_WITH_MASK)
        fg_mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except cv2.error:
        fg_mask = init_mask

    fg_mask = _refine_mask(fg_mask)

    result = image.convert("RGBA")
    result.putalpha(Image.fromarray(fg_mask))
    return result


def opencv_mask(image: Image.Image, cfg: OpenCVConfig | None = None) -> np.ndarray:
    """Return the binary foreground mask (0-255) from the OpenCV pipeline."""
    if cfg is None:
        cfg = OpenCVConfig()

    import cv2

    img_rgb = np.array(image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    init_mask = _initial_fg_mask(img_bgr, cfg)

    gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    gc_mask[init_mask > 0] = cv2.GC_PR_FGD

    border = max(cfg.min_border, int(min(h, w) * cfg.border_frac))
    gc_mask[_border_mask(h, w, border)] = cv2.GC_BGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(img_bgr, gc_mask, None, bgd_model, fgd_model, cfg.grabcut_iters, cv2.GC_INIT_WITH_MASK)
        fg_mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except cv2.error:
        fg_mask = init_mask

    return _refine_mask(fg_mask)
