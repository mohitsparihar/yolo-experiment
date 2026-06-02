"""Fine-tune YOLOv8 product finder on a YOLO-format dataset."""

import argparse
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO


DATASET_DIR = Path("training/datasets/product_finder")
YAML_PATH = DATASET_DIR / "product_finder.yaml"
OUTPUT_DIR = Path("trained_models/product_finder")
MIN_IMAGES = 20


def count_images(dataset_dir: Path, split: str = "train") -> int:
    """Count images in the dataset split."""
    img_dir = dataset_dir / "images" / split
    if not img_dir.exists():
        return 0
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return sum(1 for f in img_dir.iterdir() if f.suffix.lower() in exts)


def get_next_version() -> int:
    """Get the next version number for saved models."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(OUTPUT_DIR.glob("product_finder_v*.pt"))
    if not existing:
        return 1
    versions = []
    for p in existing:
        try:
            v = int(p.stem.split("_v")[-1])
            versions.append(v)
        except ValueError:
            continue
    return max(versions, default=0) + 1


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 product finder")
    parser.add_argument(
        "--data",
        type=Path,
        default=YAML_PATH,
        help=f"Path to YOLO dataset YAML (default: {YAML_PATH})",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()
    yaml_path = args.data.resolve()
    dataset_dir = yaml_path.parent

    # Check dataset
    if not yaml_path.exists():
        print(f"[ERROR] Dataset config not found: {yaml_path}")
        print("  Run: python labeling/export_labels.py --model product_finder")
        sys.exit(1)

    num_train = count_images(dataset_dir, "train")
    num_val = count_images(dataset_dir, "val")
    print(f"[INFO] Dataset: {num_train} train / {num_val} val images")

    if num_train + num_val < MIN_IMAGES:
        print(f"[ERROR] Need at least {MIN_IMAGES} labeled images, found {num_train + num_val}")
        print("  Label more images in Label Studio and re-export.")
        sys.exit(1)

    # Train
    print(f"[INFO] Training YOLOv8n — epochs={args.epochs}, imgsz={args.imgsz}, device={args.device}")
    model = YOLO("yolov8n.pt")
    results = model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
        project="runs/product_finder",
        name="train",
        exist_ok=True,
    )

    # Find best.pt
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    if not best_pt.exists():
        print("[ERROR] Training completed but best.pt not found")
        sys.exit(1)

    # Version and save
    version = get_next_version()
    versioned_name = f"product_finder_v{version}.pt"
    dest = OUTPUT_DIR / versioned_name
    shutil.copy2(best_pt, dest)

    # Update active.txt
    active_path = OUTPUT_DIR / "active.txt"
    active_path.write_text(str(dest.resolve()))

    # Get mAP from results
    try:
        metrics = results.results_dict
        map50 = metrics.get("metrics/mAP50(B)", 0.0)
    except Exception:
        map50 = 0.0

    print(f"\n✅ {versioned_name} saved — mAP50: {map50:.2f}")
    print(f"   Path: {dest.resolve()}")
    print(f"   Active model updated: {active_path}")


if __name__ == "__main__":
    main()
