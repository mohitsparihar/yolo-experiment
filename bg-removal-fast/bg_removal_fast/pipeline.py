"""Hybrid pipeline: OpenCV fast path with U2-NetP fallback."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Tuple

from PIL import Image

from .heuristics import HeuristicConfig, MaskQuality, evaluate_mask_quality
from .opencv_fast import OpenCVConfig, opencv_mask, opencv_remove_background
from .u2netp import u2netp_remove_background


def remove_bg_opencv(image: Image.Image, cfg: OpenCVConfig | None = None) -> Image.Image:
    return opencv_remove_background(image, cfg=cfg)


def remove_bg_u2netp(
    image: Image.Image,
    model_path: str | None = None,
    max_side: int = 640,
) -> Image.Image:
    return u2netp_remove_background(image, model_path=model_path, max_side=max_side)


def remove_bg_hybrid(
    image: Image.Image,
    model_path: str | None = None,
    max_side: int = 640,
    opencv_cfg: OpenCVConfig | None = None,
    heuristic_cfg: HeuristicConfig | None = None,
) -> Tuple[Image.Image, MaskQuality, bool]:
    """Run OpenCV first, evaluate mask quality, fallback to U2-NetP when needed."""
    mask = opencv_mask(image, cfg=opencv_cfg)
    quality = evaluate_mask_quality(image, mask, cfg=heuristic_cfg)

    if quality.is_good:
        result = image.convert("RGBA")
        result.putalpha(Image.fromarray(mask))
        return result, quality, False

    result = u2netp_remove_background(image, model_path=model_path, max_side=max_side)
    return result, quality, True


def _iter_images(input_dir: Path, recursive: bool = False) -> Iterable[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    if recursive:
        for path in input_dir.rglob("*"):
            if path.suffix.lower() in exts and path.is_file():
                yield path
    else:
        for path in input_dir.iterdir():
            if path.suffix.lower() in exts and path.is_file():
                yield path


def _output_path(input_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{input_path.stem}.png"


def _process_one(
    input_path: str,
    output_dir: str,
    mode: str,
    model_path: str | None,
    max_side: int,
    opencv_cfg: OpenCVConfig | None,
    heuristic_cfg: HeuristicConfig | None,
) -> Tuple[str, bool]:
    image = Image.open(input_path)
    output_path = _output_path(Path(input_path), Path(output_dir))

    used_fallback = False
    if mode == "opencv":
        result = remove_bg_opencv(image, cfg=opencv_cfg)
    elif mode == "u2netp":
        result = remove_bg_u2netp(image, model_path=model_path, max_side=max_side)
    else:
        result, _quality, used_fallback = remove_bg_hybrid(
            image,
            model_path=model_path,
            max_side=max_side,
            opencv_cfg=opencv_cfg,
            heuristic_cfg=heuristic_cfg,
        )

    result.save(output_path, "PNG")
    return str(output_path), used_fallback


def process_single(
    input_path: str | Path,
    output_dir: str | Path,
    mode: str = "hybrid",
    model_path: str | None = None,
    max_side: int = 640,
    opencv_cfg: OpenCVConfig | None = None,
    heuristic_cfg: HeuristicConfig | None = None,
) -> Path:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path, _used = _process_one(
        str(input_path),
        str(output_dir),
        mode,
        model_path,
        max_side,
        opencv_cfg,
        heuristic_cfg,
    )
    return Path(output_path)


def process_folder(
    input_dir: str | Path,
    output_dir: str | Path,
    mode: str = "hybrid",
    model_path: str | None = None,
    max_side: int = 640,
    workers: int | None = None,
    recursive: bool = False,
    opencv_cfg: OpenCVConfig | None = None,
    heuristic_cfg: HeuristicConfig | None = None,
) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    images = list(_iter_images(input_dir, recursive=recursive))
    if not images:
        raise RuntimeError(f"No images found in {input_dir}")

    if workers is None:
        workers = max(1, (os.cpu_count() or 2) // 2)

    try:
        from tqdm import tqdm  # type: ignore
    except Exception:  # pragma: no cover - optional
        tqdm = None

    if workers <= 1:
        iterator = images
        if tqdm is not None:
            iterator = tqdm(iterator, desc="Removing BG", unit="img")
        for img_path in iterator:
            _process_one(
                str(img_path),
                str(output_dir),
                mode,
                model_path,
                max_side,
                opencv_cfg,
                heuristic_cfg,
            )
        return

    from concurrent.futures import ProcessPoolExecutor, as_completed

    futures = []
    with ProcessPoolExecutor(max_workers=workers) as exe:
        for img_path in images:
            futures.append(
                exe.submit(
                    _process_one,
                    str(img_path),
                    str(output_dir),
                    mode,
                    model_path,
                    max_side,
                    opencv_cfg,
                    heuristic_cfg,
                )
            )

        iterator = as_completed(futures)
        if tqdm is not None:
            iterator = tqdm(iterator, total=len(futures), desc="Removing BG", unit="img")

        for _ in iterator:
            pass
