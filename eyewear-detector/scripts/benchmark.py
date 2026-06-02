"""Benchmark detection backends: dino vs yoloworld vs yolo (trained)."""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image, ImageDraw, ImageFont
from utils.image_utils import draw_boxes

MAX_INFERENCE_SIZE = 1024
OUTPUT_DIR = Path("benchmark_results")


def resize_for_inference(image: Image.Image) -> Image.Image:
    w, h = image.size
    max_dim = max(w, h)
    if max_dim <= MAX_INFERENCE_SIZE:
        return image
    scale = MAX_INFERENCE_SIZE / max_dim
    return image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def benchmark_backend(backend: str, images: list[tuple[str, Image.Image]]) -> dict:
    """Benchmark a single backend. Returns timing, detection stats, and raw detections."""
    import importlib
    import models.product_finder as pf
    importlib.reload(pf)

    # Time model loading
    t0 = time.time()
    try:
        pf.load_product_finder(backend=backend)
    except (FileNotFoundError, Exception) as e:
        return {"error": str(e)}
    load_time = time.time() - t0

    # Time inference on each image
    inference_times = []
    detection_counts = []
    all_products = []  # store raw detections for annotation

    for name, img in images:
        t0 = time.time()
        products = pf.find_products(img)
        elapsed = time.time() - t0
        inference_times.append(elapsed)
        detection_counts.append(len(products))
        all_products.append(products)

    return {
        "load_time": load_time,
        "inference_times": inference_times,
        "avg_inference": sum(inference_times) / len(inference_times),
        "total_time": load_time + sum(inference_times),
        "detection_counts": detection_counts,
        "all_products": all_products,
    }


def create_annotated(image: Image.Image, products: list[dict], backend: str,
                     inference_time: float) -> Image.Image:
    """Draw detections + a label banner on the image."""
    detections = [
        {"label": "eyewear_product", "bbox": p["bbox"], "confidence": p["confidence"]}
        for p in products
    ]
    annotated = draw_boxes(image, detections, line_width=2)

    # Add banner at top with backend name, count, and time
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except (OSError, IOError):
        font = ImageFont.load_default()

    banner_text = f"{backend.upper()}  |  {len(products)} products  |  {inference_time:.3f}s"
    text_bbox = draw.textbbox((0, 0), banner_text, font=font)
    tw = text_bbox[2] - text_bbox[0]
    th = text_bbox[3] - text_bbox[1]
    banner_h = th + 16

    # Draw banner background
    draw.rectangle([0, 0, annotated.width, banner_h], fill=(0, 0, 0))
    draw.text(((annotated.width - tw) // 2, 8), banner_text, fill=(255, 255, 255), font=font)

    return annotated


def create_comparison_grid(image_name: str, backend_images: dict[str, Image.Image]) -> Image.Image:
    """Create a side-by-side comparison image from multiple backends."""
    imgs = list(backend_images.values())
    if not imgs:
        return None

    # Make all images same height
    max_h = max(img.height for img in imgs)
    resized = []
    for img in imgs:
        if img.height != max_h:
            scale = max_h / img.height
            img = img.resize((int(img.width * scale), max_h), Image.LANCZOS)
        resized.append(img)

    total_w = sum(img.width for img in resized) + (len(resized) - 1) * 4  # 4px gap
    grid = Image.new("RGB", (total_w, max_h), (40, 40, 40))

    x = 0
    for img in resized:
        grid.paste(img, (x, 0))
        x += img.width + 4

    return grid


def main():
    parser = argparse.ArgumentParser(description="Benchmark detection backends")
    parser.add_argument("--images", type=str, help="Path to images directory")
    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--n", type=int, default=5, help="Number of images to test (default: 5)")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR), help="Output directory for comparison images")
    args = parser.parse_args()

    if not args.images and not args.image:
        parser.error("Provide either --image or --images")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    if args.image:
        # Single image mode
        sample_files = [Path(args.image)]
    else:
        images_dir = Path(args.images)
        image_files = sorted([f for f in images_dir.iterdir() if f.suffix.lower() in exts])
        if not image_files:
            print("[ERROR] No images found.")
            sys.exit(1)
        step = max(1, len(image_files) // args.n)
        sample_files = image_files[::step][:args.n]

    print(f"[INFO] Benchmarking with {len(sample_files)} image(s)\n")

    # Load and resize images once (keep originals for annotation)
    images = []
    original_images = []
    for f in sample_files:
        img = Image.open(f).convert("RGB")
        inference_img = resize_for_inference(img)
        images.append((f.stem, inference_img))
        original_images.append((f.stem, inference_img.copy()))  # use inference size for comparison
        print(f"  {f.name} ({inference_img.size[0]}x{inference_img.size[1]})")

    # Check which backends are available
    backends = ["dino", "yoloworld"]
    active_path = Path("trained_models/product_finder/active.txt")
    if active_path.exists() and Path(active_path.read_text().strip()).exists():
        backends.append("yolo")

    print(f"\n[INFO] Testing backends: {', '.join(backends)}")
    print(f"[INFO] Saving comparison images to: {output_dir}\n")
    print("=" * 70)

    results = {}
    for backend in backends:
        print(f"\n--- {backend.upper()} ---")
        r = benchmark_backend(backend, images)

        if "error" in r:
            print(f"  SKIPPED: {r['error']}")
            continue

        results[backend] = r
        print(f"  Model load:     {r['load_time']:.2f}s")
        print(f"  Avg inference:  {r['avg_inference']:.3f}s per image")
        print(f"  Total time:     {r['total_time']:.2f}s ({len(images)} images)")
        print(f"  Detections:     {r['detection_counts']}")

    # Generate annotated images and comparison grids
    print(f"\n[INFO] Generating comparison images...")

    for img_idx, (img_name, img) in enumerate(original_images):
        backend_annotated = {}

        for backend, r in results.items():
            products = r["all_products"][img_idx]
            inf_time = r["inference_times"][img_idx]
            annotated = create_annotated(img.copy(), products, backend, inf_time)
            backend_annotated[backend] = annotated

            # Save individual annotated image
            individual_path = output_dir / f"{img_name}_{backend}.jpg"
            annotated.save(individual_path, "JPEG", quality=95)

        # Save side-by-side comparison
        if len(backend_annotated) > 1:
            grid = create_comparison_grid(img_name, backend_annotated)
            if grid:
                grid_path = output_dir / f"{img_name}_comparison.jpg"
                grid.save(grid_path, "JPEG", quality=95)
                print(f"  Saved: {grid_path}")

    # Summary table
    if results:
        print("\n" + "=" * 70)
        print(f"\n{'Backend':<12} {'Load (s)':<10} {'Avg/img (s)':<14} {'Total (s)':<12} {'Avg detections'}")
        print("-" * 60)
        for name, r in results.items():
            avg_det = sum(r["detection_counts"]) / len(r["detection_counts"])
            print(f"{name:<12} {r['load_time']:<10.2f} {r['avg_inference']:<14.3f} {r['total_time']:<12.2f} {avg_det:.1f}")

        # Speedup comparison
        if "dino" in results and len(results) > 1:
            dino_avg = results["dino"]["avg_inference"]
            print(f"\nSpeedup vs DINO:")
            for name, r in results.items():
                if name == "dino":
                    continue
                speedup = dino_avg / r["avg_inference"] if r["avg_inference"] > 0 else float("inf")
                print(f"  {name}: {speedup:.1f}x faster")

        print(f"\n[INFO] All comparison images saved to: {output_dir}/")
        print(f"[INFO] Open *_comparison.jpg files to see side-by-side results")


if __name__ == "__main__":
    main()
