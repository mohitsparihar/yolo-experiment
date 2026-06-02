"""Auto-label images using Grounding DINO to generate YOLO training data."""

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image
from models.product_finder import load_product_finder, find_products

RANDOM_SEED = 42
VAL_SPLIT = 0.2
MIN_CONFIDENCE = 0.4  # Filter low-confidence detections
DATASET_DIR = Path("training/datasets/product_finder")

# Max inference size (matches detector.py)
MAX_INFERENCE_SIZE = 1024


def resize_for_inference(image: Image.Image) -> Image.Image:
    w, h = image.size
    max_dim = max(w, h)
    if max_dim <= MAX_INFERENCE_SIZE:
        return image
    scale = MAX_INFERENCE_SIZE / max_dim
    return image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def bbox_xyxy_to_cxcywh(bbox: list[float]) -> tuple[float, float, float, float]:
    """Convert [x1, y1, x2, y2] normalized to YOLO [cx, cy, w, h] normalized."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return cx, cy, w, h


def main():
    parser = argparse.ArgumentParser(description="Auto-label images with Grounding DINO for YOLO training")
    parser.add_argument("--images", type=str, required=True, help="Path to images directory")
    parser.add_argument("--min-conf", type=float, default=MIN_CONFIDENCE,
                        help=f"Minimum confidence threshold (default: {MIN_CONFIDENCE})")
    args = parser.parse_args()

    images_dir = Path(args.images)
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_files = sorted([f for f in images_dir.iterdir() if f.suffix.lower() in exts])

    print(f"[INFO] Found {len(image_files)} images in {images_dir}")
    if not image_files:
        print("[ERROR] No images found.")
        sys.exit(1)

    # Load Grounding DINO (force dino backend for highest recall)
    print("[INFO] Loading Grounding DINO for auto-labeling...")
    load_product_finder(backend="dino")

    # Process all images
    labeled_data = []  # list of (image_path, label_lines)
    total_boxes = 0
    skipped_low_conf = 0

    for i, img_path in enumerate(image_files):
        print(f"[{i+1}/{len(image_files)}] {img_path.name}", end=" ")

        try:
            image = Image.open(img_path).convert("RGB")
            inference_image = resize_for_inference(image)
            products = find_products(inference_image)
        except Exception as e:
            print(f"— ERROR: {e}")
            continue

        # Filter by confidence and convert to YOLO format
        label_lines = []
        for p in products:
            if p["confidence"] < args.min_conf:
                skipped_low_conf += 1
                continue
            cx, cy, w, h = bbox_xyxy_to_cxcywh(p["bbox"])
            # class_id=0 for eyewear_product
            label_lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            total_boxes += 1

        if label_lines:
            labeled_data.append((img_path, label_lines))

        print(f"— {len(label_lines)} boxes (conf>{args.min_conf})")

    print(f"\n[INFO] Labeled {len(labeled_data)} images with {total_boxes} total boxes")
    print(f"[INFO] Skipped {skipped_low_conf} low-confidence detections (<{args.min_conf})")

    if len(labeled_data) < 20:
        print(f"[WARN] Only {len(labeled_data)} labeled images — minimum 20 needed for training")

    # Shuffle and split
    random.seed(RANDOM_SEED)
    random.shuffle(labeled_data)
    split_idx = int(len(labeled_data) * (1 - VAL_SPLIT))
    train_data = labeled_data[:split_idx]
    val_data = labeled_data[split_idx:]

    # Create dataset directories
    for split in ["train", "val"]:
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Save images and labels
    for split, data in [("train", train_data), ("val", val_data)]:
        for img_path, label_lines in data:
            # Copy image
            dst_img = DATASET_DIR / "images" / split / img_path.name
            shutil.copy2(img_path, dst_img)

            # Save label file
            label_filename = img_path.stem + ".txt"
            dst_label = DATASET_DIR / "labels" / split / label_filename
            dst_label.write_text("\n".join(label_lines))

    # Generate YAML config
    yaml_content = f"""# Auto-generated YOLO dataset config (pseudo-labeled by Grounding DINO)
path: {DATASET_DIR.resolve()}
train: images/train
val: images/val

nc: 1
names:
  0: eyewear_product
"""
    (DATASET_DIR / "product_finder.yaml").write_text(yaml_content)

    print(f"\n[INFO] Dataset saved to: {DATASET_DIR}")
    print(f"[INFO] Split: {len(train_data)} train / {len(val_data)} val")
    print(f"\n[INFO] Next step: python training/train_product_finder.py")


if __name__ == "__main__":
    main()
