"""Single product pipeline — detect all 7 sections directly."""

import os
import uuid
from pathlib import Path

from PIL import Image as PILImage

from services.pipelines.base import BasePipeline
from services.claude_detector import detect_sections
from services.yolo_detector import detect_sections_yolo, has_trained_model
from services.fashionclip_service import get_image_embedding, get_section_embeddings, weighted_match
from utils.image_utils import crop_normalized, draw_detections, remove_background

OUTPUT_DIR = Path("outputs")
CONFIDENCE_THRESHOLD = float(os.getenv("CLAUDE_CONFIDENCE_THRESHOLD", "0.6"))
LOW_CONFIDENCE_FALLBACK = os.getenv("LOW_CONFIDENCE_FALLBACK", "true").lower() == "true"


class SinglePipeline(BasePipeline):
    def run(self, image: PILImage.Image, image_id: str) -> dict:
        return self.detect_and_match(image, image_id, product_index=0)

    def detect_and_match(
        self, image: PILImage.Image, image_id: str, product_index: int = 0,
        padding: float = 0.03, remove_bg: bool = False,
    ) -> dict:
        """Run section detection + FashionCLIP matching on a single product image."""
        model_used = "yolo"
        detections = []

        # Try YOLO first if trained model exists
        if has_trained_model("section_detector"):
            detections = detect_sections_yolo(image)
            # Fallback to Claude if low confidence
            if LOW_CONFIDENCE_FALLBACK and detections:
                avg_conf = sum(d["confidence"] for d in detections) / len(detections)
                if avg_conf < CONFIDENCE_THRESHOLD:
                    detections = detect_sections(image)
                    model_used = "claude"
            elif not detections:
                detections = detect_sections(image)
                model_used = "claude"
        else:
            detections = detect_sections(image)
            model_used = "claude"

        # Generate crops and save them
        sections = []
        crop_ext = "png" if remove_bg else "jpg"
        for det in detections:
            crop = crop_normalized(image, det["bbox"], padding=padding)
            if remove_bg:
                crop = remove_background(crop)
            crop_filename = f"{image_id}_p{product_index}_{det['label']}.{crop_ext}"
            crop_path = OUTPUT_DIR / crop_filename
            if crop.mode == "RGBA":
                crop.save(str(crop_path), "PNG")
            else:
                crop.save(str(crop_path), "JPEG", quality=95)

            emb_id = f"emb_{uuid.uuid4().hex[:12]}"

            sections.append({
                "label": det["label"],
                "class_id": det.get("class_id", 0),
                "bbox": det["bbox"],
                "confidence": det.get("confidence", 0.0),
                "crop_url": f"/outputs/{crop_filename}",
                "embedding_id": emb_id,
            })

        # FashionCLIP matching
        matches = []
        try:
            full_emb = get_image_embedding(image)
            section_embs = get_section_embeddings(image, detections)
            matches = weighted_match(full_emb, section_embs, top_k=5)
            matches = [
                {
                    "rank": i + 1,
                    "product_id": m["product_id"],
                    "score": round(m["weighted_score"], 4),
                    "matched_via": "+".join(m["component_scores"].keys()),
                }
                for i, m in enumerate(matches)
            ]
        except Exception:
            # FashionCLIP/vector DB might not be available
            pass

        # Draw annotated image
        annotated = draw_detections(image, detections)
        annotated_filename = f"{image_id}_p{product_index}_annotated.jpg"
        annotated.save(str(OUTPUT_DIR / annotated_filename))

        return {
            "product_index": product_index,
            "product_bbox": [0, 0, 1, 1],  # Full image for single product
            "sections": sections,
            "matches": matches,
            "model_used": model_used,
            "annotated_image_url": f"/outputs/{annotated_filename}",
        }
