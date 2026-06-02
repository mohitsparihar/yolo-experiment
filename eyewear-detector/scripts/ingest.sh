#!/bin/bash
# Usage: ./scripts/ingest.sh ./new_product_photos
set -e
echo "=== Step 1: Running detection on new images ==="
python detector.py --input-dir $1 --output ./crops --visualize

echo "=== Step 2: Uploading to Label Studio with pre-annotations ==="
python labeling/setup_label_studio.py --images $1

echo "=== Done. Open Label Studio to review: $LABEL_STUDIO_URL ==="
