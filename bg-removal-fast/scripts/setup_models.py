#!/usr/bin/env python3
"""Download U2-NetP ONNX model to local models directory."""

from __future__ import annotations

import argparse
from pathlib import Path

import requests

URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx"
FILENAME = "u2netp.onnx"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 // total
                    print(f"\rDownloading... {pct}%", end="", flush=True)
    if total:
        print("\rDownloading... 100%")
    print(f"Saved model to: {dest}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download U2-NetP ONNX model.")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[1] / "models" / FILENAME),
        help="Destination path for the model file.",
    )
    args = parser.parse_args()

    dest = Path(args.output)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Model already exists at {dest}")
        return 0

    download(URL, dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
