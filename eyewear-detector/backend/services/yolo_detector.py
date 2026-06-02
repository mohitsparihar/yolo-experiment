"""YOLOv8 inference for product and section detection."""

import os
from pathlib import Path
from typing import Optional

from PIL import Image as PILImage

from utils.image_utils import SECTION_CLASSES, PRODUCT_CLASSES

# Lazy-load ultralytics to avoid import cost when not needed
_models: dict = {}

MODELS_DIR = Path(__file__).parent.parent.parent / "models"


def _get_model(model_type: str):
    """Load or return cached YOLO model."""
    if model_type in _models:
        return _models[model_type]

    from ultralytics import YOLO

    model_dir = MODELS_DIR / model_type
    # Find active model weights (latest by default)
    weights = sorted(model_dir.glob("*.pt"), key=os.path.getmtime, reverse=True)
    if not weights:
        return None

    model = YOLO(str(weights[0]))
    _models[model_type] = model
    return model


def has_trained_model(model_type: str) -> bool:
    """Check if a trained YOLO model exists for this type."""
    model_dir = MODELS_DIR / model_type
    return any(model_dir.glob("*.pt"))


def detect_products_yolo(image: PILImage.Image, conf_threshold: float = 0.5) -> list[dict]:
    """Run shelf product detection with YOLO."""
    model = _get_model("product_detector")
    if model is None:
        return []

    results = model(image, conf=conf_threshold, verbose=False)

    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        img_h, img_w = result.orig_shape
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy()
            conf = float(boxes.conf[i].cpu())
            cls_id = int(boxes.cls[i].cpu())
            detections.append({
                "bbox": [
                    float(xyxy[0]) / img_w,
                    float(xyxy[1]) / img_h,
                    float(xyxy[2]) / img_w,
                    float(xyxy[3]) / img_h,
                ],
                "confidence": conf,
                "class_id": cls_id,
                "label": PRODUCT_CLASSES.get(cls_id, "unknown"),
            })

    return detections


def detect_sections_yolo(image: PILImage.Image, conf_threshold: float = 0.5) -> list[dict]:
    """Run section detection with YOLO."""
    model = _get_model("section_detector")
    if model is None:
        return []

    results = model(image, conf=conf_threshold, verbose=False)

    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        img_h, img_w = result.orig_shape
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy()
            conf = float(boxes.conf[i].cpu())
            cls_id = int(boxes.cls[i].cpu())
            detections.append({
                "label": SECTION_CLASSES.get(cls_id, "unknown"),
                "class_id": cls_id,
                "bbox": [
                    float(xyxy[0]) / img_w,
                    float(xyxy[1]) / img_h,
                    float(xyxy[2]) / img_w,
                    float(xyxy[3]) / img_h,
                ],
                "confidence": conf,
            })

    return detections


def reload_model(model_type: str):
    """Force reload a model (after training)."""
    _models.pop(model_type, None)
