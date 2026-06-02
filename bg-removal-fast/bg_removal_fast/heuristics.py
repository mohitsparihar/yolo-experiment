"""Heuristics for deciding when to fall back to U2-NetP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from PIL import Image


@dataclass
class MaskQuality:
    foreground_ratio: float
    border_leakage: float
    edge_overlap: float
    is_good: bool


@dataclass
class HeuristicConfig:
    min_fg_ratio: float = 0.03
    max_fg_ratio: float = 0.95
    max_border_leakage: float = 0.20
    min_edge_overlap: float = 0.05
    check_edge_overlap: bool = True
    downscale_max: int = 512


def _downscale(img: np.ndarray, max_side: int) -> np.ndarray:
    import cv2

    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return img
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def evaluate_mask_quality(
    image: Image.Image,
    mask: np.ndarray,
    cfg: HeuristicConfig | None = None,
) -> MaskQuality:
    import cv2

    if cfg is None:
        cfg = HeuristicConfig()

    img_rgb = np.array(image.convert("RGB"))
    mask_u8 = mask.astype(np.uint8)

    img_rgb = _downscale(img_rgb, cfg.downscale_max)
    mask_u8 = _downscale(mask_u8, cfg.downscale_max)

    if mask_u8.ndim == 3:
        mask_u8 = mask_u8[:, :, 0]

    fg = mask_u8 > 127
    fg_ratio = float(np.mean(fg))

    h, w = mask_u8.shape[:2]
    border = max(2, int(min(h, w) * 0.03))
    border_mask = np.zeros_like(fg, dtype=bool)
    border_mask[:border, :] = True
    border_mask[h - border :, :] = True
    border_mask[:, :border] = True
    border_mask[:, w - border :] = True

    border_leakage = float(np.mean(fg[border_mask])) if border_mask.any() else 0.0

    edge_overlap = 1.0
    if cfg.check_edge_overlap:
        edges_img = cv2.Canny(img_rgb, 80, 160)
        edges_mask = cv2.Canny(mask_u8, 80, 160)
        overlap = np.logical_and(edges_img > 0, edges_mask > 0)
        edge_overlap = float(np.sum(overlap)) / max(float(np.sum(edges_img > 0)), 1.0)

    is_good = (
        cfg.min_fg_ratio <= fg_ratio <= cfg.max_fg_ratio
        and border_leakage <= cfg.max_border_leakage
        and (edge_overlap >= cfg.min_edge_overlap if cfg.check_edge_overlap else True)
    )

    return MaskQuality(
        foreground_ratio=fg_ratio,
        border_leakage=border_leakage,
        edge_overlap=edge_overlap,
        is_good=is_good,
    )
