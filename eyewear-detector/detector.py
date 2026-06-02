"""Eyewear Section Detector — Main CLI entry point."""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

# from models.classifier import classify_image, load_classifier
from models.product_finder import load_product_finder, find_products, get_backend
# from models.section_finder import load_section_finder, find_sections, SECTIONS
from utils.image_utils import crop_normalized, draw_boxes, save_crop, remove_background_fast, remove_background_rembg

# Max dimension for inference — images are resized to this before detection,
# then bboxes are mapped back to original resolution for cropping.
MAX_INFERENCE_SIZE = 1024


def _resize_for_inference(image: Image.Image) -> tuple[Image.Image, float]:
    """Resize image for faster inference, return (resized_image, scale_factor)."""
    w, h = image.size
    max_dim = max(w, h)
    if max_dim <= MAX_INFERENCE_SIZE:
        return image, 1.0
    scale = MAX_INFERENCE_SIZE / max_dim
    new_w = int(w * scale)
    new_h = int(h * scale)
    return image.resize((new_w, new_h), Image.LANCZOS), scale


class EyewearDetector:
    """Eyewear detection pipeline — usable as both CLI and importable class."""

    def __init__(self, backend: str = "auto"):
        print("[INFO] Loading models...")
        # load_classifier()  # Skipped — product detection handles both shelf and single
        load_product_finder(backend=backend)
        # load_section_finder()
        print(f"[INFO] All models loaded. Backend: {get_backend()}")

    def run(self, image_path: str, output_dir: str, visualize: bool = False,
            padding: float = 0.08, remove_bg: bool = False) -> dict:
        """
        Run detection on a single image.

        Args:
            image_path: Path to input image.
            output_dir: Directory to save crops.
            visualize: Whether to save annotated image.
            padding: Extra padding around crops as fraction of box size.
            remove_bg: Whether to remove background from crops.

        Returns:
            Results dict with image_type, products, and section details.
        """
        image = Image.open(image_path).convert("RGB")
        image_name = Path(image_path).stem

        # Resize for faster inference (crops are still taken from original resolution)
        inference_image, scale = _resize_for_inference(image)
        if scale < 1.0:
            print(f"[INFO] Resized {image.size[0]}x{image.size[1]} → {inference_image.size[0]}x{inference_image.size[1]} for inference")

        # Step 1: Run product detection (skip CLIP — infer type from result count)
        t0 = time.time()
        raw_products = find_products(inference_image)
        elapsed = time.time() - t0
        print(f"[INFO] Product detection: {elapsed:.2f}s ({get_backend()}) — {len(raw_products)} products")

        # Determine image type from detection count
        if len(raw_products) <= 1:
            image_type = "single"
            if len(raw_products) == 0:
                raw_products = [{"bbox": [0.0, 0.0, 1.0, 1.0], "confidence": 1.0}]
            print(f"[INFO] Image type: single ({len(raw_products)} product)")
        else:
            image_type = "shelf"
            print(f"[INFO] Image type: shelf — found {len(raw_products)} products")

        # Step 3: Save product crops
        products_results = []
        all_detections = []  # for visualization
        crop_ext = "jpg"  # always JPEG — edge trimming produces RGB, not RGBA

        # First pass: crop all products (fast)
        crop_jobs = []  # (crop_image, output_path) for bg removal
        for idx, product in enumerate(raw_products):
            product_dir = os.path.join(output_dir, image_name, f"product_{idx}")
            os.makedirs(product_dir, exist_ok=True)
            output_path = os.path.join(product_dir, f"full_product.{crop_ext}")

            if image_type == "shelf":
                # Extra horizontal padding for temples that extend sideways
                crop_img = crop_normalized(image, product["bbox"],
                                           padding_x=padding * 2, padding_y=padding)
            else:
                crop_img = image.copy()

            if remove_bg:
                crop_jobs.append((idx, crop_img, output_path, product["confidence"]))
            else:
                if crop_img.mode == "RGBA":
                    crop_img.save(output_path, "PNG")
                else:
                    crop_img.save(output_path, "JPEG", quality=95)

            product_result = {
                "product_index": idx,
                "product_bbox": product["bbox"],
                "confidence": product["confidence"],
            }
            products_results.append(product_result)

        # Second pass: remove background using rembg (BiRefNet-general)
        if crop_jobs:
            t_bg = time.time()
            print(f"[INFO] Removing background from {len(crop_jobs)} crops (birefnet-general)...")

            for idx, crop_img, output_path, conf in crop_jobs:
                result_img = remove_background_rembg(crop_img)
                # Save as PNG to preserve transparency
                png_path = output_path.replace(".jpg", ".png")
                result_img.save(png_path, "PNG")

            print(f"[INFO] Background removal done in {time.time() - t_bg:.1f}s")

        # Build final result
        result = {
            "image": os.path.basename(image_path),
            "image_type": image_type,
            "products_found": len(products_results),
            "products": products_results,
        }

        # Save results.json
        results_dir = os.path.join(output_dir, image_name)
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(results_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(result, f, indent=2)

        # Visualize
        if visualize:
            # Add product boxes for shelf images
            if image_type == "shelf":
                for p in raw_products:
                    all_detections.append({
                        "label": "eyewear_product",
                        "bbox": p["bbox"],
                        "confidence": p["confidence"],
                    })

            annotated = draw_boxes(image, all_detections)
            annotated_path = os.path.join(results_dir, "annotated.jpg")
            annotated.save(annotated_path, "JPEG", quality=95)
            print(f"[INFO] Annotated image saved to: {annotated_path}")

        return result


def main():
    parser = argparse.ArgumentParser(description="Eyewear Section Detector")
    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--input-dir", type=str, help="Path to a folder of images")
    parser.add_argument("--output", type=str, default="./crops", help="Output directory for crops")
    parser.add_argument("--visualize", action="store_true", help="Save annotated image with boxes")
    parser.add_argument("--model", type=str, default="auto",
                        choices=["auto", "dino", "yoloworld", "yolo"],
                        help="Detection backend: auto (fastest), dino, yoloworld, or yolo (trained)")
    parser.add_argument("--padding", type=float, default=0.08,
                        help="Padding around detected products as fraction of box size (default: 0.08)")
    parser.add_argument("--remove-bg", action="store_true",
                        help="Remove background from cropped products (saves as PNG with transparency)")
    args = parser.parse_args()

    if not args.image and not args.input_dir:
        parser.error("Provide either --image or --input-dir")

    detector = EyewearDetector(backend=args.model)

    # Collect image paths
    image_paths = []
    if args.image:
        image_paths.append(args.image)
    if args.input_dir:
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        for f in sorted(Path(args.input_dir).iterdir()):
            if f.suffix.lower() in exts:
                image_paths.append(str(f))

    print(f"[INFO] Processing {len(image_paths)} image(s)")

    for i, img_path in enumerate(image_paths):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(image_paths)}] {img_path}")
        print(f"{'='*60}")

        result = detector.run(img_path, args.output, visualize=args.visualize,
                              padding=args.padding, remove_bg=args.remove_bg)
        print(json.dumps(result, indent=2))

    print(f"\n[INFO] Done. Crops saved to: {args.output}")


if __name__ == "__main__":
    main()
