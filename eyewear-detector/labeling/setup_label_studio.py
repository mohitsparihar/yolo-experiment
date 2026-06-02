"""Setup Label Studio project and import images with pre-annotations from detector."""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from label_studio_sdk import Client
from PIL import Image

load_dotenv()

LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL", "http://localhost:8080")
LABEL_STUDIO_API_KEY = os.getenv("LABEL_STUDIO_API_KEY", "")
PROJECT_NAME = "Eyewear Section Detector"

# Section label mapping for Label Studio pre-annotations
SECTION_LABELS = [
    "frame_rim", "lens_left", "lens_right",
    "nose_bridge", "temple_left", "temple_right", "brand_logo",
]


def get_or_create_project(client: Client) -> object:
    """Find existing project by name or create a new one."""
    projects = client.get_projects()
    for p in projects:
        if p.title == PROJECT_NAME:
            print(f"[INFO] Found existing project: {PROJECT_NAME} (id={p.id})")
            return p

    # Load labeling config
    config_path = Path(__file__).parent / "label_studio_config.xml"
    label_config = config_path.read_text()

    project = client.create_project(
        title=PROJECT_NAME,
        label_config=label_config,
    )
    print(f"[INFO] Created project: {PROJECT_NAME} (id={project.id})")

    # Save project ID to .env
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        content = env_path.read_text()
        if "LABEL_STUDIO_PROJECT_ID=" in content:
            lines = content.split("\n")
            lines = [
                f"LABEL_STUDIO_PROJECT_ID={project.id}" if l.startswith("LABEL_STUDIO_PROJECT_ID=") else l
                for l in lines
            ]
            env_path.write_text("\n".join(lines))
        else:
            with open(env_path, "a") as f:
                f.write(f"\nLABEL_STUDIO_PROJECT_ID={project.id}\n")

    return project


def run_detector_on_image(image_path: str) -> dict:
    """Run the eyewear detector on a single image and return results."""
    # Import here to avoid circular imports at module level
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from detector import EyewearDetector

    detector = EyewearDetector()
    result = detector.run(
        image_path=image_path,
        output_dir="/tmp/label_studio_preannotations",
        visualize=False,
    )
    return result


def detection_to_ls_annotation(result: dict, image_width: int, image_height: int) -> list[dict]:
    """Convert detector results to Label Studio annotation format."""
    annotations = []

    for product in result.get("products", []):
        product_bbox = product["product_bbox"]

        # Add product-level box (for shelf images)
        if result["image_type"] == "shelf":
            x1, y1, x2, y2 = product_bbox
            annotations.append({
                "from_name": "label",
                "to_name": "image",
                "type": "rectanglelabels",
                "value": {
                    "x": x1 * 100,
                    "y": y1 * 100,
                    "width": (x2 - x1) * 100,
                    "height": (y2 - y1) * 100,
                    "rectanglelabels": ["eyewear_product"],
                },
                "score": product.get("confidence", 0.5),
            })

        # Add section boxes
        for section_name, section_data in product.get("sections", {}).items():
            bbox = section_data["bbox"]

            # If shelf image, map section bbox from product-relative to image-relative
            if result["image_type"] == "shelf":
                px1, py1, px2, py2 = product_bbox
                pw = px2 - px1
                ph = py2 - py1
                sx1, sy1, sx2, sy2 = bbox
                x1 = px1 + sx1 * pw
                y1 = py1 + sy1 * ph
                x2 = px1 + sx2 * pw
                y2 = py1 + sy2 * ph
            else:
                x1, y1, x2, y2 = bbox

            annotations.append({
                "from_name": "label",
                "to_name": "image",
                "type": "rectanglelabels",
                "value": {
                    "x": x1 * 100,
                    "y": y1 * 100,
                    "width": (x2 - x1) * 100,
                    "height": (y2 - y1) * 100,
                    "rectanglelabels": [section_name],
                },
                "score": section_data.get("confidence", 0.5),
            })

    return annotations


def main():
    parser = argparse.ArgumentParser(description="Setup Label Studio and import images")
    parser.add_argument("--images", type=str, required=True, help="Path to images directory")
    args = parser.parse_args()

    if not LABEL_STUDIO_API_KEY:
        print("[ERROR] Set LABEL_STUDIO_API_KEY in .env file")
        print("  1. Open Label Studio at", LABEL_STUDIO_URL)
        print("  2. Go to Account → Access Token")
        print("  3. Add to .env: LABEL_STUDIO_API_KEY=your_token")
        sys.exit(1)

    # Connect to Label Studio
    client = Client(url=LABEL_STUDIO_URL, api_key=LABEL_STUDIO_API_KEY)
    project = get_or_create_project(client)

    # Collect images
    images_dir = Path(args.images)
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_files = sorted([f for f in images_dir.iterdir() if f.suffix.lower() in exts])
    print(f"[INFO] Found {len(image_files)} images in {images_dir}")

    if not image_files:
        print("[WARN] No images found.")
        return

    # Initialize detector once
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from detector import EyewearDetector
    detector = EyewearDetector()

    uploaded = 0
    for img_path in image_files:
        print(f"\n[INFO] Processing: {img_path.name}")

        # Run detection for pre-annotations
        try:
            result = detector.run(
                image_path=str(img_path),
                output_dir="/tmp/label_studio_preannotations",
                visualize=False,
            )
        except Exception as e:
            print(f"[WARN] Detection failed for {img_path.name}: {e}")
            result = {"products": [], "image_type": "single"}

        # Get image dimensions
        img = Image.open(img_path)
        w, h = img.size

        # Convert to Label Studio format
        annotations = detection_to_ls_annotation(result, w, h)

        # Upload image to Label Studio
        try:
            project.import_tasks([{
                "data": {"image": f"/data/local-files/?d={os.path.abspath(img_path)}"},
                "predictions": [{
                    "result": annotations,
                    "score": 0.5,
                }] if annotations else [],
            }])
            uploaded += 1
        except Exception as e:
            print(f"[WARN] Upload failed for {img_path.name}: {e}")

    print(f"\n[INFO] Uploaded {uploaded}/{len(image_files)} images to Label Studio")
    print(f"[INFO] Open project: {LABEL_STUDIO_URL}/projects/{project.id}")


if __name__ == "__main__":
    main()
