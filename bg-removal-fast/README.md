# Fast Background Removal (Hybrid OpenCV + U2-NetP)

CPU-friendly background removal for product images. The pipeline runs a fast OpenCV pass first and falls back to a small U2-NetP model only when the mask quality looks poor.

## Quickstart

1. Install dependencies:

```bash
python -m pip install -r bg-removal-fast/requirements.txt
```

2. Make the module importable (one-time):

```bash
python -m pip install -e bg-removal-fast
```

3. Download the U2-NetP model:

```bash
python bg-removal-fast/scripts/setup_models.py
```

4. Run the CLI on a file or folder:

```bash
python -m bg_removal_fast --input IMG20260329140724.png --output-dir output_pngs
```

## Eyewear YOLO + BG Removal (no eyewear-detector changes)

Run YOLO crops using the existing eyewear detector, then background-remove each crop:

```bash
python bg-removal-fast/scripts/eyewear_bg_pipeline.py \
  --image IMG20260329140724.png \
  --output-dir eyewear_bg_output
```

Outputs:
- Crops: `eyewear_bg_output/crops/...`
- BG-removed crops: `eyewear_bg_output/bg_removed/...`

## CLI Options

```bash
python -m bg_removal_fast \
  --input /path/to/images \
  --output-dir /path/to/output \
  --mode hybrid|opencv|u2netp \
  --max-side 640 \
  --workers 4 \
  --recursive
```

## Python API

```python
from bg_removal_fast import remove_bg_hybrid, process_folder

# Single image
result, quality, used_fallback = remove_bg_hybrid(image)
result.save("out.png")

# Folder
process_folder("input_dir", "output_dir", mode="hybrid", workers=4)
```

## Notes

- Default output is transparent PNG.
- The model file is stored in `bg-removal-fast/models/u2netp.onnx` and is ignored by git.
- Heuristics check foreground ratio, border leakage, and edge overlap to decide fallback.
