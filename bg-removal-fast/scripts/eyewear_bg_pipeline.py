#!/usr/bin/env python3
"""Run eyewear crops first, then background removal per crop.

This script does not modify the eyewear-detector codebase; it imports and uses
its EyewearDetector class directly.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow importing eyewear-detector without changing that project
ROOT = Path(__file__).resolve().parents[2]
EYEWEAR_DIR = ROOT / "eyewear-detector"
if str(EYEWEAR_DIR) not in sys.path:
    sys.path.insert(0, str(EYEWEAR_DIR))

from detector import EyewearDetector  # type: ignore

from bg_removal_fast.pipeline import process_folder


def _collect_crop_dirs(output_root: Path) -> list[Path]:
    crop_dirs: list[Path] = []
    if not output_root.exists():
        return crop_dirs
    for image_dir in output_root.iterdir():
        if not image_dir.is_dir():
            continue
        for product_dir in image_dir.iterdir():
            if product_dir.is_dir() and product_dir.name.startswith("product_"):
                crop_dirs.append(product_dir)
    return crop_dirs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run eyewear crops then background removal per crop."
    )
    parser.add_argument("--image", help="Path to a single image")
    parser.add_argument("--input-dir", help="Folder of images")
    parser.add_argument(
        "--output-dir",
        default="./eyewear_bg_output",
        help="Output directory for crops and background-removed crops",
    )
    parser.add_argument(
        "--detector-backend",
        default="auto",
        choices=["auto", "dino", "yoloworld", "yolo"],
        help="Eyewear detector backend (default: auto).",
    )
    parser.add_argument(
        "--mode",
        default="hybrid",
        choices=["hybrid", "opencv", "u2netp"],
        help="Background removal mode",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=640,
        help="Max side for U2-NetP inference",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional path to u2netp.onnx",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.08,
        help="Padding around crops (as used by eyewear detector)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Workers for background removal",
    )
    args = parser.parse_args()

    if not args.image and not args.input_dir:
        parser.error("Provide either --image or --input-dir")

    output_root = Path(args.output_dir).resolve()
    crops_dir = output_root / "crops"
    bg_dir = output_root / "bg_removed"

    image_paths: list[str] = []
    if args.image:
        image_paths.append(str(Path(args.image).resolve()))
    if args.input_dir:
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
            image_paths.extend(str(p.resolve()) for p in sorted(Path(args.input_dir).glob(ext)))

    # Ensure eyewear-detector reads its trained model paths relative to its own root
    prev_cwd = Path.cwd()
    os.chdir(EYEWEAR_DIR)
    try:
        detector = EyewearDetector(backend=args.detector_backend)

        for img_path in image_paths:
            detector.run(
                img_path,
                str(crops_dir),
                visualize=False,
                padding=args.padding,
                remove_bg=False,
            )
    finally:
        os.chdir(prev_cwd)

    # Background remove each crop image
    crop_dirs = _collect_crop_dirs(crops_dir)
    for product_dir in crop_dirs:
        # Process all images in each product dir (typically full_product.jpg)
        process_folder(
            input_dir=product_dir,
            output_dir=bg_dir / product_dir.relative_to(crops_dir),
            mode=args.mode,
            model_path=args.model_path,
            max_side=args.max_side,
            workers=args.workers,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
