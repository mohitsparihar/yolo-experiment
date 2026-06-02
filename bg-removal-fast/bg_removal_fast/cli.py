"""Command-line interface for fast background removal."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import process_folder, process_single


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fast background removal (OpenCV + U2-NetP hybrid)."
    )
    parser.add_argument(
        "--input",
        help="Image file or folder to process.",
    )
    parser.add_argument(
        "--input-dir",
        help="Folder containing images to process (deprecated; use --input).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Folder to write PNG outputs.",
    )
    parser.add_argument(
        "--mode",
        default="hybrid",
        choices=["hybrid", "opencv", "u2netp"],
        help="Processing mode (default: hybrid).",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to u2netp.onnx (optional).",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=640,
        help="Max side for U2-NetP inference (default: 640).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: cpu_count/2).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan subfolders for images.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    input_path = args.input or args.input_dir
    if not input_path:
        parser.error("Provide --input (file or folder) or --input-dir")

    path = Path(input_path)
    if path.is_file():
        process_single(
            input_path=path,
            output_dir=Path(args.output_dir),
            mode=args.mode,
            model_path=args.model_path,
            max_side=args.max_side,
        )
        return 0

    process_folder(
        input_dir=path,
        output_dir=Path(args.output_dir),
        mode=args.mode,
        model_path=args.model_path,
        max_side=args.max_side,
        workers=args.workers,
        recursive=args.recursive,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
