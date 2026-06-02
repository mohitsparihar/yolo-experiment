"""Section finder — uses Grounding DINO or trained YOLO to detect eyewear sections."""

import os
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

os.environ.setdefault("TRANSFORMERS_CACHE", "./hf_cache")

SECTIONS = [
    "frame_rim",
    "lens_left",
    "lens_right",
    "nose_bridge",
    "temple_left",
    "temple_right",
    "brand_logo",
]

SECTION_PROMPTS = {
    "frame_rim":    "eyeglass frame rim. glasses frame border. frame outline.",
    "lens_left":    "left lens. left glass lens.",
    "lens_right":   "right lens. right glass lens.",
    "nose_bridge":  "nose bridge. glasses bridge. center bridge.",
    "temple_left":  "left temple arm. left glasses arm.",
    "temple_right": "right temple arm. right glasses arm.",
    "brand_logo":   "brand logo. brand name. logo text on glasses.",
}

SECTION_BOX_THRESHOLD = 0.25

SECTION_CLASS_IDS = {
    "frame_rim":    0,
    "lens_left":    1,
    "lens_right":   2,
    "nose_bridge":  3,
    "temple_left":  4,
    "temple_right": 5,
    "brand_logo":   6,
}

GROUNDING_DINO_MODEL = "IDEA-Research/grounding-dino-tiny"

# Module-level singletons
_yolo_model = None
_using_yolo = False
_grounding_dino_model = None
_grounding_dino_processor = None
_device = None


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_grounding_dino():
    """Load Grounding DINO — tries to reuse from product_finder first."""
    global _grounding_dino_model, _grounding_dino_processor, _device

    if _grounding_dino_model is not None:
        return _grounding_dino_model, _grounding_dino_processor, _device

    # Try reusing the model already loaded by product_finder
    try:
        from models.product_finder import get_grounding_dino_model
        model, processor, device = get_grounding_dino_model()
        _grounding_dino_model = model
        _grounding_dino_processor = processor
        _device = device
        print("[INFO] Reusing Grounding DINO from product_finder for section detection")
        return model, processor, device
    except Exception:
        pass

    # Load fresh
    _device = _get_device()
    print(f"[INFO] Loading Grounding DINO (section finder) on: {_device}")
    _grounding_dino_processor = AutoProcessor.from_pretrained(GROUNDING_DINO_MODEL)
    _grounding_dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        GROUNDING_DINO_MODEL
    ).to(_device)
    _grounding_dino_model.eval()
    return _grounding_dino_model, _grounding_dino_processor, _device


def load_section_finder():
    """Load the best available section finder model."""
    global _yolo_model, _using_yolo

    active_path = Path("trained_models/section_detector/active.txt")
    if active_path.exists():
        model_path = active_path.read_text().strip()
        if Path(model_path).exists():
            from ultralytics import YOLO
            print(f"[INFO] Using trained section detector: {model_path}")
            _yolo_model = YOLO(model_path)
            _using_yolo = True
            return

    _load_grounding_dino()
    _using_yolo = False


def find_sections(image: Image.Image) -> dict:
    """
    Find eyewear sections in a single product image.

    Args:
        image: PIL Image of a single eyewear product.

    Returns:
        Dict of {section_name: {'bbox': [x1,y1,x2,y2], 'confidence': float}}
        Only includes sections that were detected above threshold.
    """
    if _using_yolo and _yolo_model is not None:
        return _find_sections_yolo(image)
    return _find_sections_grounding_dino(image)


def _find_sections_yolo(image: Image.Image) -> dict:
    """Find sections using trained YOLO model."""
    results = _yolo_model(image, verbose=False)
    w, h = image.size
    id_to_name = {v: k for k, v in SECTION_CLASS_IDS.items()}
    sections = {}

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            name = id_to_name.get(cls_id)
            if name is None:
                continue

            # Keep only highest confidence per section
            if name not in sections or conf > sections[name]["confidence"]:
                sections[name] = {
                    "bbox": [float(x1 / w), float(y1 / h), float(x2 / w), float(y2 / h)],
                    "confidence": conf,
                }

    return sections


def _find_sections_grounding_dino(image: Image.Image) -> dict:
    """Find sections using Grounding DINO — single pass with combined prompt."""
    model, processor, device = _load_grounding_dino()
    w, h = image.size

    # Combine all section keywords into one prompt for a single inference pass
    # Each section separated by ". " so Grounding DINO treats them as separate queries
    combined_prompt = (
        "frame rim. left lens. right lens. nose bridge. "
        "left temple arm. right temple arm. brand logo."
    )

    # Map detected text labels back to section names
    LABEL_TO_SECTION = {
        "frame rim": "frame_rim",
        "frame": "frame_rim",
        "rim": "frame_rim",
        "left lens": "lens_left",
        "right lens": "lens_right",
        "lens": None,  # ambiguous — handled below
        "nose bridge": "nose_bridge",
        "bridge": "nose_bridge",
        "nose": "nose_bridge",
        "left temple arm": "temple_left",
        "left temple": "temple_left",
        "right temple arm": "temple_right",
        "right temple": "temple_right",
        "temple arm": None,  # ambiguous
        "temple": None,  # ambiguous
        "brand logo": "brand_logo",
        "brand": "brand_logo",
        "logo": "brand_logo",
    }

    inputs = processor(images=image, text=combined_prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=SECTION_BOX_THRESHOLD,
        text_threshold=SECTION_BOX_THRESHOLD,
        target_sizes=[image.size[::-1]],  # (height, width)
    )[0]

    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    labels = results["text_labels"]

    if len(boxes) == 0:
        return {}

    # Group detections by section, keep highest confidence per section
    sections = {}
    # For ambiguous "lens" detections, assign by x-position (left half = left, right half = right)
    ambiguous_lenses = []

    for box, score, label in zip(boxes, scores, labels):
        label_clean = label.strip().lower()
        section_name = LABEL_TO_SECTION.get(label_clean)
        x1, y1, x2, y2 = box
        conf = float(score)
        bbox = [float(x1 / w), float(y1 / h), float(x2 / w), float(y2 / h)]

        if section_name is None:
            # Handle ambiguous labels
            if "lens" in label_clean:
                ambiguous_lenses.append({"bbox": bbox, "confidence": conf})
            elif "temple" in label_clean:
                # Assign by x-position: left half = left temple, right half = right
                cx = (bbox[0] + bbox[2]) / 2
                section_name = "temple_left" if cx < 0.5 else "temple_right"
            else:
                continue

        if section_name and (section_name not in sections or conf > sections[section_name]["confidence"]):
            sections[section_name] = {"bbox": bbox, "confidence": conf}

    # Resolve ambiguous lens detections by x-position
    for det in ambiguous_lenses:
        cx = (det["bbox"][0] + det["bbox"][2]) / 2
        name = "lens_left" if cx < 0.5 else "lens_right"
        if name not in sections or det["confidence"] > sections[name]["confidence"]:
            sections[name] = det

    return sections
