#!/usr/bin/env bash
set -euo pipefail

# Google Colab one-shot runner for:
# web search -> download -> pseudo-label with Grounding DINO -> YOLO train
#
# Typical usage from a Colab cell:
#   %%bash
#   git clone <your-repo-url> /content/image-detection-2
#   cd /content/image-detection-2
#   chmod +x eyewear-detector/scripts/colab_one_shot.sh
#   ROOT_DIR=/content/image-detection-2 \
#   ENGINE=bing \
#   PER_QUERY=80 \
#   MIN_CONF=0.30 \
#   MIN_SIDE=180 \
#   DEVICE=cuda \
#   EPOCHS=30 \
#   bash eyewear-detector/scripts/colab_one_shot.sh

ROOT_DIR="${ROOT_DIR:-/content/image-detection-2}"
EYEWEAR_DIR="${EYEWEAR_DIR:-$ROOT_DIR/eyewear-detector}"
ENGINE="${ENGINE:-bing}"
QUERY_FILE="${QUERY_FILE:-$EYEWEAR_DIR/scripts/web_queries.txt}"
DATASET_DIR="${DATASET_DIR:-$EYEWEAR_DIR/training/datasets/product_finder_web}"
PER_QUERY="${PER_QUERY:-80}"
MIN_CONF="${MIN_CONF:-0.30}"
MIN_SIDE="${MIN_SIDE:-180}"
MIN_BOX_AREA="${MIN_BOX_AREA:-0.01}"
MAX_BOX_AREA="${MAX_BOX_AREA:-0.85}"
MAX_BOXES="${MAX_BOXES:-8}"
IMGSZ="${IMGSZ:-640}"
EPOCHS="${EPOCHS:-30}"
DEVICE="${DEVICE:-cuda}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"
INSTALL_BROWSER="${INSTALL_BROWSER:-1}"
KEEP_STAGING="${KEEP_STAGING:-0}"
TRAIN_NOW="${TRAIN_NOW:-1}"
REPO_URL="${REPO_URL:-}"

if [[ ! -d "$ROOT_DIR" && -n "$REPO_URL" ]]; then
  git clone "$REPO_URL" "$ROOT_DIR"
fi

if [[ ! -d "$EYEWEAR_DIR" ]]; then
  echo "[ERROR] Eyewear project directory not found: $EYEWEAR_DIR"
  echo "[ERROR] Set ROOT_DIR or EYEWEAR_DIR correctly, or provide REPO_URL."
  exit 1
fi

cd "$EYEWEAR_DIR"

if [[ "$INSTALL_DEPS" == "1" ]]; then
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
fi

if [[ "$INSTALL_BROWSER" == "1" ]]; then
  python -m playwright install --with-deps chromium
fi

CMD=(
  python scripts/bootstrap_web_dataset.py
  --engine "$ENGINE"
  --per-query "$PER_QUERY"
  --min-conf "$MIN_CONF"
  --min-side "$MIN_SIDE"
  --min-box-area "$MIN_BOX_AREA"
  --max-box-area "$MAX_BOX_AREA"
  --max-boxes "$MAX_BOXES"
  --dataset-dir "$DATASET_DIR"
  --epochs "$EPOCHS"
  --imgsz "$IMGSZ"
  --device "$DEVICE"
)

if [[ -n "$QUERY_FILE" && -f "$QUERY_FILE" ]]; then
  CMD+=(--query-file "$QUERY_FILE")
fi

if [[ -n "${QUERY:-}" ]]; then
  CMD+=(--query "$QUERY")
fi

if [[ -n "${EXTRA_QUERY_1:-}" ]]; then
  CMD+=(--query "$EXTRA_QUERY_1")
fi

if [[ -n "${EXTRA_QUERY_2:-}" ]]; then
  CMD+=(--query "$EXTRA_QUERY_2")
fi

if [[ -n "${EXTRA_QUERY_3:-}" ]]; then
  CMD+=(--query "$EXTRA_QUERY_3")
fi

if [[ "$KEEP_STAGING" == "1" ]]; then
  CMD+=(--keep-staging)
fi

if [[ "$TRAIN_NOW" == "1" ]]; then
  CMD+=(--train)
fi

echo "[INFO] Running in $EYEWEAR_DIR"
echo "[INFO] Engine=$ENGINE PerQuery=$PER_QUERY MinConf=$MIN_CONF Device=$DEVICE Epochs=$EPOCHS"
echo "[INFO] DatasetDir=$DATASET_DIR"

if [[ "$ENGINE" == "google" ]]; then
  echo "[WARN] Google Images is less stable on Colab and may hit anti-bot checks."
  echo "[WARN] Prefer ENGINE=bing for unattended Colab runs."
fi

"${CMD[@]}"
