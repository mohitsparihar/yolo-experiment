"""Product finder — uses Grounding DINO, YOLOWorld, or trained YOLO to find eyewear products."""

import os
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from utils.nms import nms

os.environ.setdefault("TRANSFORMERS_CACHE", "./hf_cache")

# --- Grounding DINO config ---
GROUNDING_DINO_MODEL = "IDEA-Research/grounding-dino-tiny"
TEXT_PROMPT = "sunglasses. eyeglasses. glasses."
DINO_BOX_THRESHOLD = 0.35
DINO_TEXT_THRESHOLD = 0.25

# --- YOLOWorld config ---
YOLOWORLD_MODEL = "yolov8s-worldv2.pt"
YOLOWORLD_CLASSES = ["a pair of sunglasses", "a pair of glasses", "a pair of eyeglasses", "eyewear product"]
YOLOWORLD_CONF_THRESHOLD = 0.05

# --- Shared config ---
NMS_IOU_THRESHOLD = 0.5
YOLO_CONF_THRESHOLD = 0.20  # Low enough to catch transparent/clear frames

PRODUCT_CLASS_IDS = {
    "eyewear_product": 0,
}

# Module-level singletons
_grounding_dino_model = None
_grounding_dino_processor = None
_yolo_model = None
_yoloworld_model = None
_device = None
_backend = None  # "dino", "yoloworld", or "yolo"


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --- Grounding DINO ---

def _load_grounding_dino():
    """Load Grounding DINO model (singleton)."""
    global _grounding_dino_model, _grounding_dino_processor, _device
    if _grounding_dino_model is None:
        _device = _get_device()
        print(f"[INFO] Loading Grounding DINO (product finder) on: {_device}")
        _grounding_dino_processor = AutoProcessor.from_pretrained(GROUNDING_DINO_MODEL)
        _grounding_dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            GROUNDING_DINO_MODEL
        ).to(_device)
        _grounding_dino_model.eval()
    return _grounding_dino_model, _grounding_dino_processor, _device


def _find_products_grounding_dino(image: Image.Image) -> list[dict]:
    """Find products using Grounding DINO zero-shot detection."""
    model, processor, device = _load_grounding_dino()

    inputs = processor(images=image, text=TEXT_PROMPT, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=DINO_BOX_THRESHOLD,
        text_threshold=DINO_TEXT_THRESHOLD,
        target_sizes=[image.size[::-1]],  # (height, width)
    )[0]

    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    w, h = image.size

    if len(boxes) == 0:
        return []

    norm_boxes = []
    score_list = []
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        norm_boxes.append([float(x1 / w), float(y1 / h), float(x2 / w), float(y2 / h)])
        score_list.append(float(score))

    keep_indices = nms(norm_boxes, score_list, iou_threshold=NMS_IOU_THRESHOLD)

    products = []
    for i in keep_indices:
        products.append({"bbox": norm_boxes[i], "confidence": score_list[i]})

    products.sort(key=lambda p: p["bbox"][0])
    return products


# --- YOLOWorld (zero-shot, YOLO speed) ---

def _load_yoloworld():
    """Load YOLOWorld model (singleton)."""
    global _yoloworld_model
    if _yoloworld_model is None:
        from ultralytics import YOLO
        print(f"[INFO] Loading YOLOWorld ({YOLOWORLD_MODEL})")
        _yoloworld_model = YOLO(YOLOWORLD_MODEL)
        _yoloworld_model.set_classes(YOLOWORLD_CLASSES)
    return _yoloworld_model


def _find_products_yoloworld(image: Image.Image) -> list[dict]:
    """Find products using YOLOWorld zero-shot detection."""
    model = _load_yoloworld()
    w, h = image.size

    results = model.predict(image, conf=YOLOWORLD_CONF_THRESHOLD, verbose=False)
    products = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            products.append({
                "bbox": [float(x1 / w), float(y1 / h), float(x2 / w), float(y2 / h)],
                "confidence": conf,
            })

    # Apply NMS for consistency across backends
    if products:
        boxes = [p["bbox"] for p in products]
        scores = [p["confidence"] for p in products]
        keep = nms(boxes, scores, iou_threshold=NMS_IOU_THRESHOLD)
        products = [products[i] for i in keep]

    products.sort(key=lambda p: p["bbox"][0])
    return products


# --- Trained YOLO ---

def _load_trained_yolo(model_path: str):
    """Load a trained YOLO model."""
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        print(f"[INFO] Using trained product finder: {model_path}")
        _yolo_model = YOLO(model_path)
    return _yolo_model


def _find_products_yolo(image: Image.Image) -> list[dict]:
    """Find products using trained YOLO model."""
    results = _yolo_model(image, conf=YOLO_CONF_THRESHOLD, verbose=False)
    w, h = image.size
    products = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            products.append({
                "bbox": [float(x1 / w), float(y1 / h), float(x2 / w), float(y2 / h)],
                "confidence": conf,
            })

    products.sort(key=lambda p: p["bbox"][0])
    return products


# --- Public API ---

def load_product_finder(backend: str = "auto"):
    """
    Load the product finder model.

    Args:
        backend: "auto" (fastest available), "dino", "yoloworld", or "yolo" (trained).
    """
    global _backend

    if backend == "auto":
        # Priority: trained YOLO > YOLOWorld > DINO
        active_path = Path("trained_models/product_finder/active.txt")
        if active_path.exists():
            model_path = active_path.read_text().strip()
            if Path(model_path).exists():
                _load_trained_yolo(model_path)
                _backend = "yolo"
                return
        _load_yoloworld()
        _backend = "yoloworld"

    elif backend == "dino":
        _load_grounding_dino()
        _backend = "dino"

    elif backend == "yoloworld":
        _load_yoloworld()
        _backend = "yoloworld"

    elif backend == "yolo":
        active_path = Path("trained_models/product_finder/active.txt")
        if not active_path.exists():
            raise FileNotFoundError(
                "No trained YOLO model found. Run training first or use --model auto/yoloworld/dino"
            )
        model_path = active_path.read_text().strip()
        _load_trained_yolo(model_path)
        _backend = "yolo"

    else:
        raise ValueError(f"Unknown backend: {backend}. Use auto/dino/yoloworld/yolo")


def get_backend() -> str:
    """Return the currently active backend name."""
    return _backend or "unknown"


def find_products(image: Image.Image) -> list[dict]:
    """
    Find individual eyewear products in an image.

    Returns:
        List of dicts with 'bbox' [x1, y1, x2, y2] (normalized) and 'confidence'.
    """
    if _backend == "yolo":
        return _find_products_yolo(image)
    elif _backend == "yoloworld":
        return _find_products_yoloworld(image)
    else:
        return _find_products_grounding_dino(image)


def get_grounding_dino_model():
    """Return the loaded Grounding DINO model tuple for reuse by section_finder."""
    model, processor, device = _load_grounding_dino()
    return model, processor, device
