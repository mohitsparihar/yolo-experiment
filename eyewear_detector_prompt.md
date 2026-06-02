# Eyewear Section Detector — Claude Code Prompt

## What to Build

A single Python CLI tool that:
1. Takes an image path as input (single product or shelf with multiple products)
2. Detects individual eyewear products and their sections
3. Saves cropped section images to an output folder
4. Prints a JSON summary of what was found and where

That's it. No API, no database, no frontend.

---

## Hardware

Apple M4 Mac — all models must run on **MPS** (`torch.backends.mps.is_available()`).
Fall back to CPU automatically if MPS is unavailable.

---

## Project Structure

```
eyewear-detector/
├── detector.py           # Main CLI entry point
├── models/
│   ├── classifier.py     # CLIP — detects if image is "shelf" or "single product"
│   ├── product_finder.py # Grounding DINO — finds individual products on a shelf
│   └── section_finder.py # Grounding DINO — finds sections within a single product
├── utils/
│   ├── image_utils.py    # crop, draw boxes, save image helpers
│   └── nms.py            # non-maximum suppression for overlapping boxes
├── requirements.txt
└── README.md
```

---

## Section Classes

Detect these 7 sections on every individual product:

```python
SECTIONS = [
    "frame_rim",
    "lens_left",
    "lens_right",
    "nose_bridge",
    "temple_left",
    "temple_right",
    "brand_logo"
]
```

---

## Models to Use

### 1. Image Type Classifier (`models/classifier.py`)
**Model:** `openai/clip-vit-base-patch32` from HuggingFace

Use CLIP zero-shot to decide if the image is a shelf or a single product:

```python
PROMPTS = [
    "multiple sunglasses displayed on a retail shelf or fixture",  # → shelf
    "a single pair of glasses on a plain or white background",     # → single
]
```

Return `"shelf"` or `"single"` plus a confidence score.

---

### 2. Product Finder (`models/product_finder.py`)
**Model:** `IDEA-Research/grounding-dino-tiny` from HuggingFace

Only runs on shelf images. Finds each individual pair of glasses:

```python
TEXT_PROMPT = "sunglasses. eyeglasses. glasses."
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
```

After detection, apply **NMS with IoU threshold 0.5** to remove duplicate boxes.
Return list of normalized bboxes `[x1, y1, x2, y2]`.

---

### 3. Section Finder (`models/section_finder.py`)
**Model:** `IDEA-Research/grounding-dino-tiny` (same model, reuse the loaded instance)

Runs on each individual product image (or crop from shelf).
Run one pass per section using these prompts:

```python
SECTION_PROMPTS = {
    "frame_rim":    "eyeglass frame rim. glasses frame border. frame outline.",
    "lens_left":    "left lens. left glass lens.",
    "lens_right":   "right lens. right glass lens.",
    "nose_bridge":  "nose bridge. glasses bridge. center bridge.",
    "temple_left":  "left temple arm. left glasses arm.",
    "temple_right": "right temple arm. right glasses arm.",
    "brand_logo":   "brand logo. brand name. logo text on glasses.",
}
SECTION_BOX_THRESHOLD = 0.25  # lower threshold — small parts are harder to detect
```

For each section, take only the **highest confidence box** if any are found above threshold.
Return a dict of `{section_name: bbox}` for sections that were detected.

---

## CLI Interface (`detector.py`)

```bash
# Single image
python detector.py --image path/to/image.jpg --output ./crops

# Batch — process all images in a folder
python detector.py --input-dir path/to/images/ --output ./crops

# Show annotated image with boxes drawn (optional)
python detector.py --image path/to/image.jpg --output ./crops --visualize
```

### Output folder structure

```
crops/
└── image_name/
    ├── product_0/
    │   ├── full_product.jpg       # the whole product crop (from shelf) or original
    │   ├── frame_rim.jpg
    │   ├── lens_left.jpg
    │   ├── lens_right.jpg
    │   ├── nose_bridge.jpg
    │   ├── temple_left.jpg
    │   ├── temple_right.jpg
    │   └── brand_logo.jpg         # only saved if detected
    ├── product_1/
    │   └── ...
    └── results.json               # full detection summary for this image
```

For a single product image, there is only `product_0/`.

### `results.json` format

```json
{
  "image": "frame_display.jpg",
  "image_type": "shelf",
  "products_found": 2,
  "products": [
    {
      "product_index": 0,
      "product_bbox": [0.05, 0.10, 0.48, 0.90],
      "sections_detected": ["frame_rim", "lens_left", "lens_right", "nose_bridge"],
      "sections_missing": ["temple_left", "temple_right", "brand_logo"],
      "sections": {
        "frame_rim":   {"bbox": [0.06, 0.12, 0.47, 0.88], "confidence": 0.71, "crop": "product_0/frame_rim.jpg"},
        "lens_left":   {"bbox": [0.08, 0.20, 0.25, 0.75], "confidence": 0.65, "crop": "product_0/lens_left.jpg"},
        "lens_right":  {"bbox": [0.28, 0.20, 0.45, 0.75], "confidence": 0.63, "crop": "product_0/lens_right.jpg"},
        "nose_bridge": {"bbox": [0.23, 0.35, 0.30, 0.55], "confidence": 0.41, "crop": "product_0/nose_bridge.jpg"}
      }
    }
  ]
}
```

---

## Model Loading

- Load CLIP and Grounding DINO **once at startup**, not per image
- Store loaded models in a simple module-level singleton so batch processing doesn't reload them
- Print which device is being used at startup: `[INFO] Running on: mps`
- Models download from HuggingFace on first run — set `TRANSFORMERS_CACHE=./hf_cache` so they cache locally

---

## `requirements.txt`

```
torch
torchvision
transformers
Pillow
opencv-python
numpy
```

---

## README must include

```bash
# Install
pip install -r requirements.txt

# First run (downloads models ~1.3GB total, cached after)
python detector.py --image sample.jpg --output ./crops

# Batch
python detector.py --input-dir ./my_images --output ./crops --visualize
```

- Note that first run downloads ~1.3GB of model weights (CLIP ~600MB + Grounding DINO ~700MB)
- Explain the `crops/` output structure
- Show a one-liner example of loading a crop into FashionCLIP:
  ```python
  from PIL import Image
  crop = Image.open("crops/image_name/product_0/frame_rim.jpg")
  # pass `crop` directly to your FashionCLIP pipeline
  ```

---

## Part 2 — Label Studio Integration & Retraining Pipeline

Add these scripts alongside the existing detector. No changes needed to `detector.py`.

### Additional Project Structure

```
eyewear-detector/
├── detector.py                  # (unchanged)
├── models/                      # (unchanged)
├── utils/                       # (unchanged)
│
├── labeling/
│   ├── setup_label_studio.py    # Creates Label Studio project + imports images
│   ├── export_labels.py         # Exports annotations from Label Studio → YOLO format
│   └── label_studio_config.xml  # Bounding box UI config for Label Studio
│
├── training/
│   ├── train_section_detector.py  # Fine-tune YOLOv8 section detector
│   ├── train_product_finder.py    # Fine-tune YOLOv8 shelf product detector
│   └── datasets/
│       ├── section_detector/
│       │   ├── images/train/
│       │   ├── images/val/
│       │   ├── labels/train/
│       │   ├── labels/val/
│       │   └── section_detector.yaml
│       └── product_finder/
│           ├── images/train/
│           ├── images/val/
│           ├── labels/train/
│           ├── labels/val/
│           └── product_finder.yaml
│
├── trained_models/
│   ├── section_detector/        # versioned .pt files: section_detector_v1.pt, v2.pt ...
│   └── product_finder/          # versioned .pt files: product_finder_v1.pt, v2.pt ...
│
├── scripts/
│   ├── ingest.sh                # Full pipeline: detect → push to Label Studio
│   └── retrain.sh               # Full pipeline: export labels → train → activate model
│
├── .env                         # LABEL_STUDIO_URL, LABEL_STUDIO_API_KEY, LABEL_STUDIO_PROJECT_ID
└── requirements.txt             # add: label-studio-sdk, ultralytics
```

---

## Label Studio Setup (`labeling/setup_label_studio.py`)

### Install & run Label Studio locally

```bash
pip install label-studio
label-studio start  # runs at http://localhost:8080
```

### Script behavior

```bash
python labeling/setup_label_studio.py --images ./my_images
```

This script should:
1. Connect to Label Studio using `LABEL_STUDIO_URL` and `LABEL_STUDIO_API_KEY` from `.env`
2. Create a project named `"Eyewear Section Detector"` if it doesn't exist, or reuse it by name
3. Apply the labeling config from `label_studio_config.xml` (bounding boxes with the 7 section classes + 1 product class)
4. Run `detector.py` on every image in `--images` to get pre-annotations (Grounding DINO predictions)
5. Upload each image to Label Studio with its pre-annotations already drawn — team just reviews and corrects, not draws from scratch
6. Print the Label Studio project URL when done

### `label_studio_config.xml`

Generate this Label Studio XML config for bounding box annotation with these labels:

```xml
<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true"/>
  <RectangleLabels name="label" toName="image" showInline="true">
    <Label value="eyewear_product" background="#FF6B6B"/>
    <Label value="frame_rim"       background="#4ECDC4"/>
    <Label value="lens_left"       background="#45B7D1"/>
    <Label value="lens_right"      background="#96CEB4"/>
    <Label value="nose_bridge"     background="#FFEAA7"/>
    <Label value="temple_left"     background="#DDA0DD"/>
    <Label value="temple_right"    background="#98D8C8"/>
    <Label value="brand_logo"      background="#F7DC6F"/>
  </RectangleLabels>
</View>
```

---

## Export Labels (`labeling/export_labels.py`)

```bash
python labeling/export_labels.py --model section_detector
python labeling/export_labels.py --model product_finder
```

This script should:
1. Connect to Label Studio, fetch all **completed** annotations (skip skipped/unreviewed tasks)
2. For `--model section_detector`: export only annotations that contain section labels (frame_rim, lens_*, etc.)
3. For `--model product_finder`: export only annotations that contain `eyewear_product` labels
4. Convert Label Studio JSON export → YOLO `.txt` format:
   - One `.txt` file per image, same filename as the image
   - Each line: `class_id cx cy w h` (normalized 0–1, center x/y)
   - Class IDs must match constants in `models/section_finder.py` and `models/product_finder.py`
5. Copy images + label files into the correct `training/datasets/` subfolder
6. Auto-split 80% train / 20% val (random, seeded for reproducibility)
7. Auto-generate the `.yaml` config file for that model
8. Print a summary: `Exported 84 labeled images → 67 train / 17 val`

### YOLO class ID mapping

```python
# section_detector classes
SECTION_CLASS_IDS = {
    "frame_rim":    0,
    "lens_left":    1,
    "lens_right":   2,
    "nose_bridge":  3,
    "temple_left":  4,
    "temple_right": 5,
    "brand_logo":   6,
}

# product_finder classes
PRODUCT_CLASS_IDS = {
    "eyewear_product": 0,
}
```

---

## Training Scripts

### `training/train_section_detector.py`

```bash
python training/train_section_detector.py
python training/train_section_detector.py --epochs 100 --imgsz 800  # optional overrides
```

Steps:
1. Check that at least **30 labeled images** exist in the dataset — exit with a clear message if not
2. Run: `yolo train model=yolov8n.pt data=training/datasets/section_detector/section_detector.yaml epochs=50 imgsz=640 device=mps`
3. Find the `best.pt` from the training run output
4. Auto-version it: copy to `trained_models/section_detector/section_detector_v{N}.pt` where N increments
5. Save a `trained_models/section_detector/active.txt` file containing just the path to the new best model
6. Print: `✅ section_detector_v3.pt saved — mAP50: 0.83`

### `training/train_product_finder.py`

Same structure as above but:
- Minimum **20 labeled images**
- Saves to `trained_models/product_finder/`
- Uses `product_finder.yaml`

---

## Model Activation in `detector.py`

Update `models/section_finder.py` and `models/product_finder.py` to check for trained models at startup:

```python
def load_model():
    active_path = Path("trained_models/section_detector/active.txt")
    if active_path.exists():
        model_path = active_path.read_text().strip()
        print(f"[INFO] Using trained section detector: {model_path}")
        return YOLO(model_path)  # ultralytics YOLOv8
    else:
        print("[INFO] No trained model found — using Grounding DINO (zero-shot)")
        return load_grounding_dino()
```

This means:
- **Before any training:** Grounding DINO handles detection automatically
- **After first training run:** YOLO model is used instead, no code changes needed
- **After retraining:** update `active.txt` to point to new version — detector picks it up on next run

---

## Convenience Shell Scripts

### `scripts/ingest.sh` — process new images and push to Label Studio for review

```bash
#!/bin/bash
# Usage: ./scripts/ingest.sh ./new_product_photos
set -e
echo "=== Step 1: Running detection on new images ==="
python detector.py --input-dir $1 --output ./crops --visualize

echo "=== Step 2: Uploading to Label Studio with pre-annotations ==="
python labeling/setup_label_studio.py --images $1

echo "=== Done. Open Label Studio to review: $LABEL_STUDIO_URL ==="
```

### `scripts/retrain.sh` — export labels and retrain both models

```bash
#!/bin/bash
set -e
echo "=== Step 1: Exporting labels from Label Studio ==="
python labeling/export_labels.py --model section_detector
python labeling/export_labels.py --model product_finder

echo "=== Step 2: Training section detector ==="
python training/train_section_detector.py

echo "=== Step 3: Training product finder ==="
python training/train_product_finder.py

echo "=== Done. New models are active. Run detector.py to use them. ==="
```

---

## `.env` file

```
LABEL_STUDIO_URL=http://localhost:8080
LABEL_STUDIO_API_KEY=your_api_key_here   # found in Label Studio → Account → Access Token
LABEL_STUDIO_PROJECT_ID=                 # auto-filled by setup_label_studio.py on first run
```

---

## Updated `requirements.txt`

```
torch
torchvision
transformers
Pillow
opencv-python
numpy
ultralytics
label-studio-sdk
python-dotenv
```

---

## Part 3 — FastAPI & Next.js Integration

The detector runs as a module inside your existing FastAPI app — no separate server, no HTTP calls between services. Models load once when FastAPI starts and stay in memory.

---

## How to Mount Into Your Existing FastAPI App

### Step 1 — Add the detector as a module

Copy the entire `eyewear-detector/` folder into your existing project:

```
your-app/
├── main.py                  # your existing FastAPI app
├── routers/
│   └── ...                  # your existing routes
├── eyewear_detector/        # ← drop the detector in here as a package
│   ├── __init__.py
│   ├── detector.py
│   ├── models/
│   ├── utils/
│   ├── labeling/
│   └── training/
└── ...
```

### Step 2 — Create the eyewear router

Create `routers/eyewear.py` in your existing app:

```python
from fastapi import APIRouter, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
import shutil, uuid
from pathlib import Path
from eyewear_detector.detector import EyewearDetector

router = APIRouter(prefix="/api/eyewear", tags=["eyewear"])

# Load models once at module level — stays in memory for all requests
detector = EyewearDetector()

UPLOAD_DIR = Path("uploads/eyewear")
CROPS_DIR  = Path("outputs/eyewear_crops")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CROPS_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/detect")
async def detect(file: UploadFile = File(...)):
    """
    Upload an image, get back detected sections and crop file paths.
    Handles both shelf and single product images automatically.
    """
    image_id = str(uuid.uuid4())
    image_path = UPLOAD_DIR / f"{image_id}_{file.filename}"

    with image_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    result = detector.run(
        image_path=str(image_path),
        output_dir=str(CROPS_DIR / image_id)
    )

    return JSONResponse({
        "image_id": image_id,
        "image_type": result["image_type"],
        "products": result["products"],
        "crops_base_url": f"/eyewear-crops/{image_id}"
    })


@router.get("/crops/{image_id}/{product_index}/{section}")
async def get_crop(image_id: str, product_index: int, section: str):
    """Return a specific section crop as an image file."""
    from fastapi.responses import FileResponse
    crop_path = CROPS_DIR / image_id / f"product_{product_index}" / f"{section}.jpg"
    if not crop_path.exists():
        return JSONResponse({"error": "crop not found"}, status_code=404)
    return FileResponse(str(crop_path), media_type="image/jpeg")


@router.post("/retrain")
async def retrain(background_tasks: BackgroundTasks, model: str = "both"):
    """
    Trigger retraining in the background.
    model: "section_detector" | "product_finder" | "both"
    """
    from eyewear_detector.training import retrain_models
    background_tasks.add_task(retrain_models, model=model)
    return {"status": "training started", "model": model}


@router.get("/model/status")
async def model_status():
    """Returns which model version is currently active."""
    from pathlib import Path
    def active_version(name):
        p = Path(f"eyewear_detector/trained_models/{name}/active.txt")
        return p.read_text().strip() if p.exists() else "grounding_dino (zero-shot)"

    return {
        "section_detector": active_version("section_detector"),
        "product_finder":   active_version("product_finder"),
    }
```

### Step 3 — Register the router in your existing `main.py`

```python
# your existing main.py — add these two lines
from routers.eyewear import router as eyewear_router
app.include_router(eyewear_router)

# Serve crop images as static files
from fastapi.staticfiles import StaticFiles
app.mount("/eyewear-crops", StaticFiles(directory="outputs/eyewear_crops"), name="eyewear-crops")
```

### Step 4 — Refactor `detector.py` to expose a class

Wrap the existing CLI logic in a class so it's importable:

```python
# eyewear_detector/detector.py  — add this class, keep the CLI block at bottom

class EyewearDetector:
    def __init__(self):
        # Load all models once
        self.classifier     = load_clip_classifier()
        self.product_finder = load_product_finder()   # YOLO or Grounding DINO
        self.section_finder = load_section_finder()   # YOLO or Grounding DINO

    def run(self, image_path: str, output_dir: str) -> dict:
        """Main entry point. Returns results dict + saves crops to output_dir."""
        # existing detection logic here — same as CLI but returns dict instead of printing
        ...

# Keep CLI working too
if __name__ == "__main__":
    # existing argparse CLI code unchanged
    ...
```

---

## API Endpoints Summary

| Method | Endpoint | What it does |
|--------|----------|--------------|
| `POST` | `/api/eyewear/detect` | Upload image → get sections + crop paths |
| `GET`  | `/api/eyewear/crops/{image_id}/{product_index}/{section}` | Fetch a specific crop image |
| `POST` | `/api/eyewear/retrain?model=both` | Trigger retraining in background |
| `GET`  | `/api/eyewear/model/status` | See which model version is active |

---

## Detect Response Shape

```json
{
  "image_id": "abc-123",
  "image_type": "shelf",
  "crops_base_url": "/eyewear-crops/abc-123",
  "products": [
    {
      "product_index": 0,
      "product_bbox": [0.05, 0.10, 0.48, 0.90],
      "sections": {
        "frame_rim":  { "bbox": [0.06, 0.12, 0.47, 0.88], "confidence": 0.71, "crop_url": "/api/eyewear/crops/abc-123/0/frame_rim" },
        "lens_left":  { "bbox": [0.08, 0.20, 0.25, 0.75], "confidence": 0.65, "crop_url": "/api/eyewear/crops/abc-123/0/lens_left" },
        "lens_right": { "bbox": [0.28, 0.20, 0.45, 0.75], "confidence": 0.63, "crop_url": "/api/eyewear/crops/abc-123/0/lens_right" }
      }
    }
  ]
}
```

---

## Next.js Integration

### Install

```bash
npm install
```

No extra packages needed — uses native `fetch` and standard Next.js patterns.

### API helper (`lib/eyewear.ts`)

```typescript
const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"

export interface Section {
  bbox: [number, number, number, number]
  confidence: number
  crop_url: string
}

export interface Product {
  product_index: number
  product_bbox: [number, number, number, number]
  sections: Record<string, Section>
}

export interface DetectResult {
  image_id: string
  image_type: "shelf" | "single" | "worn" | "closeup"
  crops_base_url: string
  products: Product[]
}

export async function detectEyewear(file: File): Promise<DetectResult> {
  const form = new FormData()
  form.append("file", file)
  const res = await fetch(`${API_BASE}/api/eyewear/detect`, {
    method: "POST",
    body: form,
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export function cropUrl(crop_url: string): string {
  return `${API_BASE}${crop_url}`
}
```

### Upload + Results component (`components/EyewearDetector.tsx`)

```tsx
"use client"
import { useState, useCallback } from "react"
import { detectEyewear, cropUrl, DetectResult, Product } from "@/lib/eyewear"

export default function EyewearDetector() {
  const [result, setResult]   = useState<DetectResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState<string | null>(null)

  const handleFile = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setLoading(true)
    setError(null)
    try {
      setResult(await detectEyewear(file))
    } catch (err: any) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [])

  return (
    <div>
      <input type="file" accept="image/*" onChange={handleFile} disabled={loading} />
      {loading && <p>Detecting sections...</p>}
      {error   && <p style={{ color: "red" }}>{error}</p>}
      {result  && <ResultsView result={result} />}
    </div>
  )
}

function ResultsView({ result }: { result: DetectResult }) {
  return (
    <div>
      <p>Type: <strong>{result.image_type}</strong> — {result.products.length} product(s) found</p>
      {result.products.map(p => <ProductCard key={p.product_index} product={p} />)}
    </div>
  )
}

function ProductCard({ product }: { product: Product }) {
  return (
    <div style={{ border: "1px solid #ccc", padding: 12, marginTop: 12 }}>
      <p><strong>Product {product.product_index}</strong></p>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
        {Object.entries(product.sections).map(([label, section]) => (
          <div key={label} style={{ textAlign: "center" }}>
            <img
              src={cropUrl(section.crop_url)}
              alt={label}
              style={{ width: 100, height: 80, objectFit: "cover", borderRadius: 4 }}
            />
            <p style={{ fontSize: 11, margin: "4px 0 0" }}>
              {label}<br />
              <span style={{ color: "#888" }}>{(section.confidence * 100).toFixed(0)}%</span>
            </p>
          </div>
        ))}
      </div>
    </div>
  )
}
```

### Use it in any page

```tsx
// app/products/detect/page.tsx  or  pages/products/detect.tsx
import EyewearDetector from "@/components/EyewearDetector"

export default function DetectPage() {
  return (
    <main>
      <h1>Eyewear Section Detector</h1>
      <EyewearDetector />
    </main>
  )
}
```

### `.env.local` for Next.js

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## CORS — add to your FastAPI `main.py`

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # add your production domain too
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Passing Crops to FashionCLIP (in your existing backend)

Since everything is on the same machine, your FashionCLIP service can read crop files directly by path — no need to go through the API:

```python
from PIL import Image
from eyewear_detector.detector import EyewearDetector

detector = EyewearDetector()  # reuse the same instance — don't create a new one per request

result = detector.run(image_path="product.jpg", output_dir="./crops/abc123")

for product in result["products"]:
    for section_name, section_data in product["sections"].items():
        crop = Image.open(section_data["local_crop_path"])  # direct file path, no HTTP
        embedding = your_fashionclip.embed(crop)
        matches   = your_vector_db.search(embedding, top_k=5)
        # attach matches back to result as needed
```

Make sure `detector.run()` returns both `crop_url` (for the API response) and `local_crop_path` (absolute path on disk) in each section dict.

