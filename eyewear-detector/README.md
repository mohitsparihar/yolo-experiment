# Eyewear Section Detector

Detects individual eyewear products in images and identifies their sections (frame rim, lenses, nose bridge, temples, brand logo). Works on both single product images and shelf/display images with multiple products.

## Models

- **CLIP** (`openai/clip-vit-base-patch32`) — classifies image as "shelf" or "single product"
- **Grounding DINO** (`IDEA-Research/grounding-dino-tiny`) — zero-shot detection of products and sections
- **YOLOv8** (after training) — replaces Grounding DINO with a fine-tuned model

## Install

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# First run downloads ~1.3GB of model weights (CLIP ~600MB + Grounding DINO ~700MB)
# Models are cached in ./hf_cache after first download

# Single image
python detector.py --image sample.jpg --output ./crops

# Batch — process all images in a folder
python detector.py --input-dir ./my_images --output ./crops --visualize
```

## Output Structure

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
    ├── results.json               # full detection summary
    └── annotated.jpg              # only with --visualize
```

## Using Crops with FashionCLIP

```python
from PIL import Image
crop = Image.open("crops/image_name/product_0/frame_rim.jpg")
# pass `crop` directly to your FashionCLIP pipeline
```

## Label Studio Integration (Part 2)

### Setup

```bash
pip install label-studio
label-studio start  # runs at http://localhost:8080
```

1. Get your API key from Label Studio → Account → Access Token
2. Add it to `.env`: `LABEL_STUDIO_API_KEY=your_token`

### Ingest new images with pre-annotations

```bash
./scripts/ingest.sh ./new_product_photos
```

### Retrain models after labeling

```bash
./scripts/retrain.sh
```

After retraining, the detector automatically uses the trained YOLO model instead of Grounding DINO.

## Build a General Web Dataset

If the current YOLO model was trained mostly on shelf photos, it will learn shelf context along with the glasses. To generalize, train on mixed web images: product shots, flat lays, people wearing glasses, and multi-object scenes.

### Install Playwright once

```bash
pip install -r requirements.txt
playwright install chromium
```

### Create a pseudo-labeled dataset from web search

```bash
python scripts/bootstrap_web_dataset.py \
  --query-file scripts/web_queries.txt \
  --per-query 30 \
  --dataset-dir training/datasets/product_finder_web
```

This flow:

- Uses Playwright to collect image URLs from Bing Images by default
- Downloads a small, query-diverse batch of images
- Runs Grounding DINO to auto-label eyewear boxes
- Filters low-confidence and bad-size detections
- Writes a YOLO dataset directly to `training/datasets/product_finder_web`
- Deletes temporary downloads by default

To use Google Images instead:

```bash
python scripts/bootstrap_web_dataset.py \
  --engine google \
  --query-file scripts/web_queries.txt \
  --per-query 50 \
  --min-conf 0.35 \
  --dataset-dir training/datasets/product_finder_web \
  --train
```

Google is more brittle than Bing and may show consent or anti-bot pages depending on your region and browser state.

### Train on that generated dataset

```bash
python training/train_product_finder.py \
  --data training/datasets/product_finder_web/product_finder.yaml \
  --epochs 50 \
  --imgsz 640 \
  --device mps
```

### One-shot build + train

```bash
python scripts/bootstrap_web_dataset.py \
  --query-file scripts/web_queries.txt \
  --per-query 30 \
  --dataset-dir training/datasets/product_finder_web \
  --train
```

### Run on Google Colab

If this flow is too heavy for your Mac, run it on a Colab GPU runtime. The cleanest path is a single `%%bash` cell that installs dependencies and launches the pipeline.

```bash
%%bash
git clone <your-repo-url> /content/image-detection-2
cd /content/image-detection-2
chmod +x eyewear-detector/scripts/colab_one_shot.sh

ROOT_DIR=/content/image-detection-2 \
ENGINE=bing \
PER_QUERY=80 \
MIN_CONF=0.30 \
MIN_SIDE=180 \
DEVICE=cuda \
EPOCHS=30 \
bash eyewear-detector/scripts/colab_one_shot.sh
```

Notes:

- Use a Colab GPU runtime so Grounding DINO and YOLO training use `cuda`
- `ENGINE=bing` is the recommended unattended Colab mode
- `ENGINE=google` may hit anti-bot checks and is less reliable in headless cloud sessions

### Important note

This is pseudo-labeling, not ground truth. Use it to bootstrap quickly, then manually review a small subset and retrain again. A strong loop is:

1. Train on the generated web dataset
2. Run the model on 50-100 target images
3. Correct only the bad predictions
4. Retrain on the corrected mixed dataset

## FastAPI Integration (Part 3)

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/eyewear/detect` | Upload image, get sections + crop paths |
| `GET`  | `/api/eyewear/crops/{image_id}/{product_index}/{section}` | Fetch a specific crop |
| `POST` | `/api/eyewear/retrain?model=both` | Trigger retraining in background |
| `GET`  | `/api/eyewear/model/status` | See active model versions |

### Mount in your FastAPI app

```python
from routers.eyewear import router as eyewear_router
app.include_router(eyewear_router)

from fastapi.staticfiles import StaticFiles
app.mount("/eyewear-crops", StaticFiles(directory="outputs/eyewear_crops"), name="eyewear-crops")
```

### CORS

```python
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Hardware

Optimized for Apple M4 Mac with MPS acceleration. Falls back to CPU automatically if MPS is unavailable.
