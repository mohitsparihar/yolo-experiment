# Project Overview: Detection + Cropping + Background Removal

This repository contains an eyewear detection pipeline plus a fast background-removal pipeline. The system supports multiple detection backends (trained YOLO, YOLOWorld, Grounding DINO) and multiple background-removal methods (OpenCV GrabCut, U2-NetP ONNX, and rembg in the original eyewear utility code).

## High-Level Flow

1. **Detect eyewear products** in an input image (single or shelf).
2. **Crop each product** from the original image at full resolution.
3. **Remove background** from crops (optional in eyewear detector, or via the dedicated fast pipeline).

---

## Eyewear Detector (YOLO / YOLOWorld / Grounding DINO)

Location: `eyewear-detector/`

The main entrypoint is `eyewear-detector/detector.py`. It:

- Loads a product finder model via `eyewear-detector/models/product_finder.py`.
- Runs detection on a resized inference image for speed.
- Maps boxes back to original resolution to crop at full quality.
- Saves crops and an optional annotated image.

### Backends

**Trained YOLO (best quality for your custom data)**
- Backend name: `yolo`
- Expects a trained model path stored in `eyewear-detector/trained_models/product_finder/active.txt`.
- When available, this is the highest quality for your domain.

**YOLOWorld (zero-shot, fast)**
- Backend name: `yoloworld`
- Uses `yolov8s-worldv2.pt` with eyewear-specific class prompts.
- Good speed, moderate quality without training.

**Grounding DINO (zero-shot, accurate, slower)**
- Backend name: `dino`
- Uses `IDEA-Research/grounding-dino-tiny` with text prompt:
  `"sunglasses. eyeglasses. glasses."`
- Good accuracy for varied scenes, but slower than YOLOWorld.

### Auto Backend Selection

Backend `auto` chooses the best available in this order:
1. Trained YOLO (if `active.txt` points to a valid model)
2. YOLOWorld
3. Grounding DINO

---

## OpenCV Background Removal (Fast Path)

Location: `bg-removal-fast/bg_removal_fast/opencv_fast.py`

This is a **fast CPU-only** approach optimized for product images with clean-ish backgrounds:

- Estimates background color from image borders (LAB color space).
- Builds an initial foreground mask by color distance.
- Refines with **GrabCut** for better edges.
- Applies morphological open/close + blur for smoother alpha.

Best for: clean backgrounds, batch speed.

---

## U2-NetP Background Removal (Model Fallback)

Location: `bg-removal-fast/bg_removal_fast/u2netp.py`

This uses a **small U2-NetP ONNX model** for higher quality when OpenCV fails:

- Onnxruntime CPU inference.
- Input resized to 320 with padding (matching rembg preprocessing).
- Output mask is resized back to original image size.
- Default inference `max_side=640` for speed.

Model download:
- Script: `bg-removal-fast/scripts/setup_models.py`
- Output: `bg-removal-fast/models/u2netp.onnx`

---

## Hybrid Background Removal (OpenCV + U2-NetP)

Location: `bg-removal-fast/bg_removal_fast/pipeline.py`

The hybrid method runs OpenCV first, then falls back to U2-NetP when the mask looks poor.

### Heuristics Used

Defined in `bg-removal-fast/bg_removal_fast/heuristics.py`:

- **Foreground ratio**: fail if < 3% or > 95%
- **Border leakage**: fail if too much foreground touches the borders
- **Edge overlap**: low overlap between mask edge and image edges triggers fallback

This keeps speed high for clean images and improves quality for harder cases.

---

## Built-in Background Removal (Eyewear Detector)

Location: `eyewear-detector/utils/image_utils.py`

The eyewear detector now ships with two background-removal functions that can be chosen per call. Default behavior (when using `--remove-bg`) uses the fast edge-trim method.

### `remove_background_fast(image)` — Edge trimming (default)

- Pure NumPy, no ML model, **instant**.
- Samples background color from the four corners.
- Scans rows/columns from each edge and trims only those where ≥99% of pixels match background.
- Stops the trim the moment a row/column contains actual content (frame, temples, etc.).
- Safety cap: never trims more than 25% from any side.
- Returns an **RGB** crop (not RGBA) — just smaller, no alpha channel.

Best for: shelf photos with uniform backgrounds. Preserves temples that extend sideways from the frame.

### `remove_background_rembg(image)` — Accurate fallback

- Uses `rembg` with the `u2netp` model (~4MB, the smallest/fastest variant).
- Input is downscaled to 320px for inference, mask is upscaled back.
- Returns an **RGBA** image with transparent background.
- Slower (~2–3s per crop on CPU), but understands object shapes — handles transparent/clear frames that edge-trim can't detect.

Use this when crops have complex backgrounds or transparent frames where corner-based color detection fails.

### `remove_background(image)` — Default wrapper

Alias for `remove_background_fast()`. Swap out the implementation in `image_utils.py` if you want `rembg` as the default.

### Cropping padding

`crop_normalized(image, bbox, padding=0.0, padding_x=None, padding_y=None)` supports both uniform and asymmetric padding:

- `padding` — uniform padding as fraction of box size.
- `padding_x` / `padding_y` — override horizontal/vertical padding independently. Useful when temples extend sideways and need extra horizontal room.

The CLI defaults to `--padding 0.08` (8%) uniform. The `detector.py` pipeline applies `padding_x = 2 × padding` on shelf detections to preserve sideways-extending temples.

---

## Integrating the Eyewear Detector into Another Project

You can import `EyewearDetector` directly from `eyewear-detector/detector.py` in an external project without using the CLI.

### Minimal example

```python
import sys
sys.path.insert(0, "/path/to/image-detection-2/eyewear-detector")

from detector import EyewearDetector

# Load models once at startup
detector = EyewearDetector(backend="auto")  # or "yolo" | "yoloworld" | "dino"

# Run on an image
result = detector.run(
    image_path="shelf_photo.jpg",
    output_dir="./crops",
    visualize=False,      # set True to save annotated image
    padding=0.08,          # fraction of box size (default 0.08)
    remove_bg=True,        # edge-trim the shelf background
)

# result is a dict:
# {
#   "image": "shelf_photo.jpg",
#   "image_type": "shelf" | "single",
#   "products_found": int,
#   "products": [
#       {
#           "product_index": 0,
#           "product_bbox": [x1, y1, x2, y2],  # normalized [0,1]
#           "confidence": 0.97,
#       },
#       ...
#   ]
# }
```

### Output layout

```
<output_dir>/<image_name>/
├── results.json              # full detection summary
├── annotated.jpg             # (only if visualize=True)
├── product_0/
│   └── full_product.jpg      # cropped + trimmed product
├── product_1/
│   └── full_product.jpg
└── ...
```

### Programmatic usage without saving to disk

If you only need the crops and bboxes in memory, you can use the utility functions directly without calling `detector.run()`:

```python
from PIL import Image
from detector import EyewearDetector
from utils.image_utils import crop_normalized, remove_background_fast, remove_background_rembg
from models.product_finder import find_products

detector = EyewearDetector(backend="auto")

img = Image.open("shelf_photo.jpg").convert("RGB")
products = find_products(img)  # list of {"bbox": [...], "confidence": float}

for p in products:
    crop = crop_normalized(img, p["bbox"], padding_x=0.16, padding_y=0.08)
    clean = remove_background_fast(crop)       # RGB, trimmed edges
    # or: clean = remove_background_rembg(crop)  # RGBA, full matting
```

### Dependencies to install in the consuming project

```bash
pip install torch torchvision transformers ultralytics Pillow opencv-python numpy
pip install "rembg[cpu]"   # only needed if you use remove_background_rembg()
```

### Gotchas when embedding in another project

- `EyewearDetector()` loads models at construction — keep a singleton instance, don't re-create per request.
- If using the trained YOLO backend, your working directory must allow `trained_models/product_finder/active.txt` paths to resolve. Either `cd` into `eyewear-detector/` before calling, or pass absolute paths in `active.txt`.
- Padding is applied to normalized bbox coordinates *before* conversion to pixel coords. `padding_x=0.2` means 20% of the box width on each side, not 20% of the image.
- `remove_background_fast()` returns RGB; `remove_background_rembg()` returns RGBA. Check `.mode` before saving or compositing.

---

## New Fast BG Removal Package

Location: `bg-removal-fast/`

Key components:

- `bg_removal_fast/cli.py`: CLI for file/folder processing.
- `bg_removal_fast/pipeline.py`: API + hybrid logic.
- `bg_removal_fast/opencv_fast.py`: fast OpenCV mask.
- `bg_removal_fast/u2netp.py`: ONNX U2-NetP inference.
- `bg_removal_fast/heuristics.py`: fallback scoring.
- `bg_removal_fast/__main__.py`: enables `python -m bg_removal_fast`.

### Example Usage

Single image:
```bash
python -m bg_removal_fast --input IMG20260329140724.png --output-dir output_pngs
```

Batch folder:
```bash
python -m bg_removal_fast --input downloaded_images --output-dir output_pngs
```

---

## YOLO Crops + BG Removal Pipeline (No Eyewear Code Changes)

Location: `bg-removal-fast/scripts/eyewear_bg_pipeline.py`

This script:

1. Runs eyewear detector to generate product crops.
2. Background-removes each crop using the hybrid pipeline.

Outputs:
- Crops: `eyewear_bg_output/crops/...`
- BG-removed: `eyewear_bg_output/bg_removed/...`

---

## Notes / Gotchas

- **Trained YOLO requires `trained_models/product_finder/active.txt`** to exist and point to a valid model path.
- The `eyewear_bg_pipeline.py` script temporarily changes working directory to `eyewear-detector` so trained YOLO paths resolve correctly.
- The U2-NetP model file is ignored by git via `.gitignore`.

If you want this doc placed somewhere else (e.g., inside `bg-removal-fast/`), tell me and I’ll move it.
