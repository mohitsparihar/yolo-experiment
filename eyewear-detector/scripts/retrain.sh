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
