"""Background removal using rembg (BiRefNet-general — best accuracy).

Usage:
    # Single image
    python remove_bg_rembg.py --image crops/IMG20260329140724/product_0/full_product.jpg

    # Entire crops folder
    python remove_bg_rembg.py --input-dir crops/IMG20260329140724

    # Custom output directory
    python remove_bg_rembg.py --input-dir crops/IMG20260329140724 --output crops_nobg

    # Use lite model for faster processing
    python remove_bg_rembg.py --input-dir crops/IMG20260329140724 --model birefnet-general-lite
"""

import argparse
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
from rembg import remove, new_session

# Best accuracy model — no downscaling, feed full resolution to the model
DEFAULT_MODEL = "birefnet-general"
MAX_SIZE = 1024
SESSION = None


def get_session(model_name: str | None = None):
    global SESSION
    model_name = model_name or DEFAULT_MODEL
    if SESSION is None:
        print(f"[INFO] Loading rembg model: {model_name}")
        SESSION = new_session(model_name)
    return SESSION


def remove_bg(image: Image.Image, session=None) -> Image.Image:
    """Remove background at full resolution for best accuracy."""
    session = session or get_session()
    w, h = image.size
    max_dim = max(w, h)

    # Only downscale if image is very large (>1024px) to avoid OOM,
    # but apply the mask at original resolution for sharp edges
    if max_dim > MAX_SIZE:
        scale = MAX_SIZE / max_dim
        small = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        result = remove(small, session=session)
        alpha = result.split()[-1].resize((w, h), Image.LANCZOS)
        out = image.convert("RGBA")
        out.putalpha(alpha)
        return out

    return remove(image, session=session)


def process_file(input_path: str, output_path: str, session=None) -> str:
    """Process a single image file."""
    session = session or get_session()
    img = Image.open(input_path).convert("RGB")
    result = remove_bg(img, session=session)
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    result.save(output_path, "PNG")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Background removal using rembg")
    parser.add_argument("--image", type=str, help="Single image path")
    parser.add_argument("--input-dir", type=str, help="Folder of images (recursive)")
    parser.add_argument("--output", type=str, help="Output directory (default: replaces in place as .png)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers (default: 4)")
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help="rembg model name (default: birefnet-general). Options: birefnet-general, birefnet-general-lite, isnet-general-use, u2net",
    )
    args = parser.parse_args()

    # Initialize session with chosen model
    get_session(args.model)

    if not args.image and not args.input_dir:
        parser.error("Provide --image or --input-dir")

    # Collect files
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    jobs = []

    if args.image:
        p = Path(args.image)
        out = Path(args.output) / p.stem if args.output else p.with_suffix(".png")
        jobs.append((str(p), str(out)))

    if args.input_dir:
        for f in sorted(Path(args.input_dir).rglob("*")):
            if f.suffix.lower() in exts:
                if args.output:
                    rel = f.relative_to(args.input_dir)
                    out = Path(args.output) / rel.with_suffix(".png")
                else:
                    out = f.with_suffix(".png")
                jobs.append((str(f), str(out)))

    print(f"[INFO] Processing {len(jobs)} image(s) with {args.workers} workers...")
    t0 = time.time()

    def _process(job):
        inp, out = job
        process_file(inp, out)
        return inp

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for path in pool.map(_process, jobs):
            done += 1
            print(f"[{done}/{len(jobs)}] {path}")

    print(f"[INFO] Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
