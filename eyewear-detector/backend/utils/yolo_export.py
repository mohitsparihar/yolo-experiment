"""Export labels from DB to YOLO .txt format for training."""

import os
import random
import shutil
from pathlib import Path

from sqlmodel import Session, select

from models.db import Image, Label, engine
from utils.image_utils import bbox_xyxy_to_cxcywh


def export_yolo_dataset(
    label_mode: str,
    output_dir: str,
    val_split: float = 0.2,
) -> dict:
    """
    Export labeled images to YOLO format.

    Args:
        label_mode: "product" or "section"
        output_dir: Root dir for the dataset (e.g., training/datasets/product_detector)
        val_split: Fraction of images for validation

    Returns:
        dict with counts: {"train": N, "val": M, "total_labels": L}
    """
    output_path = Path(output_dir)

    # Clean existing
    for split in ["train", "val"]:
        (output_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_path / "labels" / split).mkdir(parents=True, exist_ok=True)

    with Session(engine) as session:
        # Get all images that have labels for this mode
        stmt = (
            select(Image)
            .where(Image.status.in_(["labeled", "needs_review"]))
        )
        images = session.exec(stmt).all()

        # Filter to images that have labels for this mode
        labeled_images = []
        for img in images:
            labels_stmt = select(Label).where(
                Label.image_id == img.id,
                Label.label_mode == label_mode,
            )
            labels = session.exec(labels_stmt).all()
            if labels:
                labeled_images.append((img, labels))

    if not labeled_images:
        return {"train": 0, "val": 0, "total_labels": 0}

    # Shuffle and split
    random.shuffle(labeled_images)
    split_idx = max(1, int(len(labeled_images) * (1 - val_split)))
    train_set = labeled_images[:split_idx]
    val_set = labeled_images[split_idx:]

    total_labels = 0

    for split_name, split_data in [("train", train_set), ("val", val_set)]:
        for img, labels in split_data:
            # Copy image
            src = Path(img.path)
            dst_img = output_path / "images" / split_name / src.name
            shutil.copy2(src, dst_img)

            # Write label file
            label_file = output_path / "labels" / split_name / (src.stem + ".txt")
            lines = []
            for lbl in labels:
                cxcywh = bbox_xyxy_to_cxcywh([lbl.x1, lbl.y1, lbl.x2, lbl.y2])
                line = f"{lbl.class_id} {cxcywh[0]:.6f} {cxcywh[1]:.6f} {cxcywh[2]:.6f} {cxcywh[3]:.6f}"
                lines.append(line)
                total_labels += 1
            label_file.write_text("\n".join(lines) + "\n")

    return {
        "train": len(train_set),
        "val": len(val_set),
        "total_labels": total_labels,
    }
