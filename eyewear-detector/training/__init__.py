"""Training package — provides retrain_models for API integration."""

import subprocess
import sys


def retrain_models(model: str = "both"):
    """
    Run retraining for the specified model(s).

    Args:
        model: 'section_detector', 'product_finder', or 'both'
    """
    if model in ("section_detector", "both"):
        print("[TRAIN] Exporting section detector labels...")
        subprocess.run([sys.executable, "labeling/export_labels.py", "--model", "section_detector"], check=True)
        print("[TRAIN] Training section detector...")
        subprocess.run([sys.executable, "training/train_section_detector.py"], check=True)

    if model in ("product_finder", "both"):
        print("[TRAIN] Exporting product finder labels...")
        subprocess.run([sys.executable, "labeling/export_labels.py", "--model", "product_finder"], check=True)
        print("[TRAIN] Training product finder...")
        subprocess.run([sys.executable, "training/train_product_finder.py"], check=True)

    print(f"[TRAIN] Done retraining: {model}")
