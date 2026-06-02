# Eyewear Detector — Integration Prompt

## What This Is

The `eyewear-detector/` folder contains a working eyewear product detection system using a **trained YOLOv8 model** (59x faster than the Grounding DINO fallback). Copy it into your FastAPI + Next.js project as a package. This prompt tells you how to wire it up.

---

## What It Does

1. **Detects individual eyewear products** in shelf/display images using a trained YOLOv8 model (~0.03s per image)
2. **Infers image type** from detection count — `"shelf"` (multiple products) or `"single"` (one product)
3. **Crops products with configurable padding** (asymmetric padding supported — more horizontal to preserve temples)
4. **Optionally removes background** from crops — edge-trim (instant) or `rembg` with u2netp (accurate fallback)
5. **Saves cropped product images** to an output directory
6. **Returns structured JSON** with product bounding boxes and confidence scores

Section-level detection (frame_rim, lenses, temples, etc.) is implemented but currently commented out in `detector.py`. Uncomment the section detection block when needed.

---

## Detection Backends

The system supports 3 backends via the `backend` parameter:

| Backend | Speed | Accuracy | When to Use |
|---------|-------|----------|-------------|
| `"yolo"` (trained) | **0.03s/img** | Best | **Default when `trained_models/product_finder/active.txt` exists** |
| `"yoloworld"` | 0.07s/img | Moderate | Zero-shot fallback, no training needed |
| `"dino"` | 1.5s/img | High recall | Generating training data, max recall |

With `backend="auto"` (the default), it picks the fastest available: trained YOLO > YOLOWorld > DINO.

---

## What You're Copying

```
eyewear_detector/                  # ← rename from eyewear-detector to underscore
├── __init__.py
├── detector.py                    # EyewearDetector class — main entry point
├── models/
│   ├── __init__.py
│   ├── classifier.py              # CLIP classifier (commented out, not loaded)
│   ├── product_finder.py          # 3-backend product detector (yolo/yoloworld/dino)
│   └── section_finder.py          # Section detector (commented out)
├── utils/
│   ├── __init__.py
│   ├── image_utils.py             # crop, draw boxes, save helpers
│   └── nms.py                     # non-maximum suppression
├── routers/
│   ├── __init__.py
│   └── eyewear.py                 # FastAPI router (4 endpoints, ready to mount)
├── trained_models/
│   └── product_finder/
│       ├── product_finder_v1.pt   # ← Trained YOLOv8 weights (6MB) — COPY THIS
│       └── active.txt             # Points to the .pt file — COPY THIS
├── labeling/                      # Label Studio integration (optional)
├── training/                      # YOLO fine-tuning scripts (optional)
├── scripts/                       # benchmark, auto-labeling (optional)
├── lib/eyewear.ts                 # Next.js API helper
├── components/EyewearDetector.tsx # Next.js upload + results component
└── requirements.txt
```

**Critical files to copy** (minimum for detection to work):
- `detector.py`
- `models/` (entire folder)
- `utils/` (entire folder)
- `trained_models/product_finder/product_finder_v1.pt` + `active.txt`

**For FastAPI integration**, also copy:
- `routers/eyewear.py`

**For Next.js**, also copy:
- `lib/eyewear.ts`
- `components/EyewearDetector.tsx`

---

## Step 1 — Copy Into Your Project

```
your-project/
├── backend/
│   ├── main.py                    # your FastAPI app
│   ├── ...
│   └── eyewear_detector/          # ← paste here (rename to underscore)
│       ├── __init__.py
│       ├── detector.py
│       ├── models/
│       ├── utils/
│       ├── routers/
│       └── trained_models/
│           └── product_finder/
│               ├── product_finder_v1.pt
│               └── active.txt
├── frontend/                      # your Next.js app
│   ├── lib/eyewear.ts             # ← paste here
│   └── components/EyewearDetector.tsx  # ← paste here
```

**Important:** Rename the folder from `eyewear-detector` to `eyewear_detector` (underscore) so Python can import it.

---

## Step 2 — Fix Imports Inside the Package

The internal imports use bare module names. Update them to relative imports:

### `eyewear_detector/detector.py`
```python
# Change:
from models.product_finder import load_product_finder, find_products, get_backend
from utils.image_utils import crop_normalized, draw_boxes, save_crop

# To:
from .models.product_finder import load_product_finder, find_products, get_backend
from .utils.image_utils import crop_normalized, draw_boxes, save_crop
```

### `eyewear_detector/models/product_finder.py`
```python
# Change:
from utils.nms import nms

# To:
from ..utils.nms import nms
```

### `eyewear_detector/models/section_finder.py`
```python
# Change:
from models.product_finder import get_grounding_dino_model

# To:
from .product_finder import get_grounding_dino_model
```

### `eyewear_detector/routers/eyewear.py`
```python
# Change:
from detector import EyewearDetector
from training import retrain_models

# To:
from ..detector import EyewearDetector
from ..training import retrain_models
```

### `eyewear_detector/trained_models` path fix

The `active.txt` file contains an absolute path. Update it to match the new location:
```
# Update active.txt to point to the .pt file relative to where your app runs from,
# OR use an absolute path to the new location:
eyewear_detector/trained_models/product_finder/product_finder_v1.pt
```

---

## Step 3 — Install Dependencies

Add to your project's `requirements.txt`:

```
torch
torchvision
ultralytics
Pillow
numpy
```

These are the **only deps needed** for the trained YOLO model + edge-trim background removal. No need for `transformers` (~700MB Grounding DINO download) unless you want the `dino` fallback.

Optional (only if you want dino/yoloworld fallback):
```
transformers
```

Optional (only if you want the `rembg` background-removal fallback for tricky crops like transparent frames):
```
rembg[cpu]
```

Optional (only for training/labeling pipeline):
```
label-studio-sdk
python-dotenv
opencv-python
```

---

## Step 4A — FastAPI Integration

In your `main.py`:

```python
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from eyewear_detector.routers.eyewear import router as eyewear_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount eyewear detection endpoints
app.include_router(eyewear_router)

# Serve crop images as static files
app.mount("/eyewear-crops", StaticFiles(directory="outputs/eyewear_crops"), name="eyewear-crops")
```

This gives you:

| Method | Endpoint | What it does |
|--------|----------|--------------|
| `POST` | `/api/eyewear/detect` | Upload image → get product detections + crop paths |
| `GET`  | `/api/eyewear/crops/{image_id}/{product_index}/{section}` | Fetch a specific crop image |
| `POST` | `/api/eyewear/retrain?model=both` | Trigger YOLO retraining (background) |
| `GET`  | `/api/eyewear/model/status` | Check active model versions |

Or use `EyewearDetector` directly in your own endpoints:

```python
from eyewear_detector.detector import EyewearDetector

# Initialize ONCE at app startup (loads trained YOLO, ~0.01s)
detector = EyewearDetector(backend="auto")

@app.post("/my-custom-endpoint")
async def detect(file: UploadFile):
    # save file, then:
    result = detector.run(
        image_path=saved_path,
        output_dir="./crops",
        padding=0.08,       # 8% padding around each detection (2× on horizontal axis)
        remove_bg=True,     # trim shelf background from edges
    )
    return result
```

---

## Padding + Background Removal

### Padding

`EyewearDetector.run(..., padding=0.08)` controls how much breathing room is added around each YOLO bounding box before cropping.

- Internally, the detector applies **horizontal padding = `padding × 2`**, vertical = `padding`. This preserves sideways-extending temples that often fall just outside the YOLO box.
- `padding` is a fraction of the box's own width/height, not the whole image.
- Default: `0.08` (8% → 16% horizontal).

### Background Removal

Pass `remove_bg=True` to `run()` to strip the background from each product crop. The detector ships with two methods in `utils/image_utils.py`:

#### `remove_background_fast(image)` — Default, instant

- Pure NumPy, no ML model.
- Samples corner colors to learn the background, then trims rows/columns from the edges as long as ≥99% of pixels match that background color.
- Stops the moment any non-background content appears (frame, temple, lens reflection).
- Never trims more than 25% from any side (hard safety cap).
- Returns an **RGB** image (smaller crop, no alpha channel).
- Best for: shelf photos with uniform backgrounds.

#### `remove_background_rembg(image)` — Accurate fallback (opt-in)

- Uses `rembg` with the `u2netp` ONNX model (~4MB, smallest variant).
- Downscales to 320px for inference, upscales mask back.
- Returns an **RGBA** image with transparent background.
- Slower (~2–3s/crop on CPU), but understands object shapes — handles **transparent/clear frames** and complex backgrounds where edge-trim fails.
- Requires `pip install rembg[cpu]`.

#### Routing by confidence (optional pattern)

A useful pattern is to route by YOLO confidence: high-confidence detections use `fast`, low-confidence ones use `rembg`.

```python
from eyewear_detector.utils.image_utils import (
    crop_normalized,
    remove_background_fast,
    remove_background_rembg,
)
from eyewear_detector.models.product_finder import find_products
from PIL import Image

img = Image.open("shelf.jpg").convert("RGB")
products = find_products(img)

for p in products:
    crop = crop_normalized(img, p["bbox"], padding_x=0.16, padding_y=0.08)
    if p["confidence"] >= 0.85:
        cleaned = remove_background_fast(crop)     # RGB, instant
        cleaned.save(f"out_{p['bbox']}.jpg", "JPEG", quality=95)
    else:
        cleaned = remove_background_rembg(crop)    # RGBA, accurate
        cleaned.save(f"out_{p['bbox']}.png", "PNG")
```

#### Asymmetric padding

`crop_normalized()` accepts `padding`, `padding_x`, and `padding_y`:

```python
crop = crop_normalized(img, bbox, padding_x=0.16, padding_y=0.08)
```

Use this when you want different amounts on horizontal vs vertical axes (e.g., give temples more room sideways).

#### Parallelizing background removal

Both methods are thread-safe after the first call. For batch processing, use a thread pool:

```python
from concurrent.futures import ThreadPoolExecutor

def _process(job):
    idx, crop, path = job
    remove_background_fast(crop).save(path, "JPEG", quality=95)
    return idx

with ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(_process, jobs))
```

For `remove_background_rembg`, warm up the model by calling it once before threading — otherwise multiple threads will each try to load the ONNX model simultaneously.

## Step 4B — Next.js Integration

Copy into your Next.js app:
- `lib/eyewear.ts` → `your-app/lib/eyewear.ts`
- `components/EyewearDetector.tsx` → `your-app/components/EyewearDetector.tsx`

Add to `.env.local`:
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

Use in any page:
```tsx
import EyewearDetector from "@/components/EyewearDetector"

export default function Page() {
  return <EyewearDetector />
}
```

---

## Key Details

- **Trained YOLO model is 6MB** — loads in ~0.01s, runs inference in ~0.03s per image
- **`backend="auto"`** (default) automatically uses the trained YOLO when `trained_models/product_finder/active.txt` exists
- **No large model downloads needed** if using trained YOLO + edge-trim bg removal only. Grounding DINO (~700MB), CLIP (~600MB), and rembg u2netp (~4MB) are only downloaded if you explicitly use `backend="dino"` or `remove_background_rembg()`
- **Runs on MPS** (Apple Silicon) by default, falls back to CPU
- **Reuse the same `EyewearDetector` instance** across requests — do NOT create a new one per request
- **Images are auto-resized** to max 1024px for inference, crops are taken from original resolution
- **`remove_background_fast` returns RGB**, **`remove_background_rembg` returns RGBA** — check `.mode` before saving/compositing

## What You May Want to Customize

- `detector.py` — add fields to the result dict, change output paths, re-enable section detection, change default padding
- `routers/eyewear.py` — adjust endpoint paths, add auth, change upload/crop directories, expose `padding`/`remove_bg` as query params
- `models/product_finder.py` — tune `NMS_IOU_THRESHOLD` (default 0.5) if too many/few products detected
- `utils/image_utils.py` — tune `BG_THRESHOLD` (0.99) and the 25% safety cap in `remove_background_fast`, or swap the default `remove_background` implementation between fast/rembg
