"""Export annotations from Label Studio → YOLO format for training."""

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv
from label_studio_sdk import Client
from PIL import Image

load_dotenv()

LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL", "http://localhost:8080")
LABEL_STUDIO_API_KEY = os.getenv("LABEL_STUDIO_API_KEY", "")
LABEL_STUDIO_PROJECT_ID = os.getenv("LABEL_STUDIO_PROJECT_ID", "")

# Class ID mappings
SECTION_CLASS_IDS = {
    "frame_rim":    0,
    "lens_left":    1,
    "lens_right":   2,
    "nose_bridge":  3,
    "temple_left":  4,
    "temple_right": 5,
    "brand_logo":   6,
}

PRODUCT_CLASS_IDS = {
    "eyewear_product": 0,
}

RANDOM_SEED = 42
VAL_SPLIT = 0.2


def ls_bbox_to_yolo(x_pct: float, y_pct: float, w_pct: float, h_pct: float) -> tuple:
    """
    Convert Label Studio bbox (percentages) to YOLO format (normalized center).

    Label Studio: x, y, width, height as percentages (0-100)
    YOLO: cx, cy, w, h as normalized (0-1)
    """
    cx = (x_pct + w_pct / 2) / 100.0
    cy = (y_pct + h_pct / 2) / 100.0
    w = w_pct / 100.0
    h = h_pct / 100.0
    return cx, cy, w, h


def export_for_model(client: Client, project, model_type: str):
    """Export labeled data from Label Studio to YOLO dataset format."""

    if model_type == "section_detector":
        class_ids = SECTION_CLASS_IDS
        target_labels = set(SECTION_CLASS_IDS.keys())
        dataset_dir = Path("training/datasets/section_detector")
        yaml_name = "section_detector.yaml"
    elif model_type == "product_finder":
        class_ids = PRODUCT_CLASS_IDS
        target_labels = set(PRODUCT_CLASS_IDS.keys())
        dataset_dir = Path("training/datasets/product_finder")
        yaml_name = "product_finder.yaml"
    else:
        print(f"[ERROR] Unknown model type: {model_type}")
        sys.exit(1)

    # Fetch completed tasks
    tasks = project.get_labeled_tasks()
    print(f"[INFO] Found {len(tasks)} labeled tasks in project")

    # Filter tasks that have relevant labels
    valid_tasks = []
    for task in tasks:
        if not task.get("annotations"):
            continue

        # Use the latest annotation
        annotation = task["annotations"][-1]
        if annotation.get("was_cancelled"):
            continue

        results = annotation.get("result", [])
        has_relevant = False
        for r in results:
            if r.get("type") != "rectanglelabels":
                continue
            labels = r.get("value", {}).get("rectanglelabels", [])
            if any(l in target_labels for l in labels):
                has_relevant = True
                break

        if has_relevant:
            valid_tasks.append(task)

    print(f"[INFO] {len(valid_tasks)} tasks have {model_type} labels")

    if not valid_tasks:
        print("[WARN] No valid labeled data found. Nothing to export.")
        return

    # Shuffle and split
    random.seed(RANDOM_SEED)
    random.shuffle(valid_tasks)
    split_idx = int(len(valid_tasks) * (1 - VAL_SPLIT))
    train_tasks = valid_tasks[:split_idx]
    val_tasks = valid_tasks[split_idx:]

    # Clean and recreate dataset dirs
    for split in ["train", "val"]:
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    total_labels = 0

    for split, task_list in [("train", train_tasks), ("val", val_tasks)]:
        for task in task_list:
            image_url = task["data"].get("image", "")

            # Resolve image path from Label Studio URL
            if image_url.startswith("/data/local-files/?d="):
                src_path = image_url.replace("/data/local-files/?d=", "")
            else:
                print(f"[WARN] Skipping task {task['id']} — unsupported image source")
                continue

            if not Path(src_path).exists():
                print(f"[WARN] Image not found: {src_path}")
                continue

            # Copy image
            img_filename = Path(src_path).name
            dst_img = dataset_dir / "images" / split / img_filename
            shutil.copy2(src_path, dst_img)

            # Build YOLO label file
            annotation = task["annotations"][-1]
            label_lines = []

            for r in annotation.get("result", []):
                if r.get("type") != "rectanglelabels":
                    continue

                value = r.get("value", {})
                labels = value.get("rectanglelabels", [])

                for label in labels:
                    if label not in class_ids:
                        continue

                    cx, cy, w, h = ls_bbox_to_yolo(
                        value["x"], value["y"], value["width"], value["height"]
                    )
                    cid = class_ids[label]
                    label_lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                    total_labels += 1

            # Save label file
            label_filename = Path(img_filename).stem + ".txt"
            dst_label = dataset_dir / "labels" / split / label_filename
            dst_label.write_text("\n".join(label_lines))

    # Generate YAML config
    id_to_name = {v: k for k, v in class_ids.items()}
    names_dict = {i: id_to_name[i] for i in sorted(id_to_name.keys())}

    yaml_content = f"""# Auto-generated YOLO dataset config for {model_type}
path: {dataset_dir.resolve()}
train: images/train
val: images/val

nc: {len(class_ids)}
names: {names_dict}
"""
    (dataset_dir / yaml_name).write_text(yaml_content)

    print(f"\n[INFO] Exported {len(valid_tasks)} labeled images → {len(train_tasks)} train / {len(val_tasks)} val")
    print(f"[INFO] Total label annotations: {total_labels}")
    print(f"[INFO] Dataset saved to: {dataset_dir}")


def main():
    parser = argparse.ArgumentParser(description="Export Label Studio annotations to YOLO format")
    parser.add_argument("--model", type=str, required=True,
                        choices=["section_detector", "product_finder"],
                        help="Which model to export labels for")
    args = parser.parse_args()

    if not LABEL_STUDIO_API_KEY:
        print("[ERROR] Set LABEL_STUDIO_API_KEY in .env file")
        sys.exit(1)

    if not LABEL_STUDIO_PROJECT_ID:
        print("[ERROR] Set LABEL_STUDIO_PROJECT_ID in .env file (run setup_label_studio.py first)")
        sys.exit(1)

    client = Client(url=LABEL_STUDIO_URL, api_key=LABEL_STUDIO_API_KEY)
    project = client.get_project(int(LABEL_STUDIO_PROJECT_ID))

    export_for_model(client, project, args.model)


if __name__ == "__main__":
    main()
