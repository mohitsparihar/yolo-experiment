"""FastAPI router for eyewear detection API endpoints."""

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from detector import EyewearDetector

router = APIRouter(prefix="/api/eyewear", tags=["eyewear"])

# Load models once at module level — stays in memory for all requests
detector = EyewearDetector()

UPLOAD_DIR = Path("uploads/eyewear")
CROPS_DIR = Path("outputs/eyewear_crops")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CROPS_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/detect")
async def detect(file: UploadFile = File(...), padding: float = 0.03, remove_bg: bool = False):
    """
    Upload an image, get back detected sections and crop file paths.
    Handles both shelf and single product images automatically.

    Query params:
        padding: Extra padding around crops as fraction of box size (default: 0.03).
        remove_bg: Remove background from crops, saving as PNG with transparency.
    """
    image_id = str(uuid.uuid4())
    image_path = UPLOAD_DIR / f"{image_id}_{file.filename}"

    with image_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    result = detector.run(
        image_path=str(image_path),
        output_dir=str(CROPS_DIR / image_id),
        padding=padding,
        remove_bg=remove_bg,
    )

    # Add crop_url to each section for API consumers
    for product in result["products"]:
        for section_name, section_data in product.get("sections", {}).items():
            section_data["crop_url"] = (
                f"/api/eyewear/crops/{image_id}/{product['product_index']}/{section_name}"
            )

    return JSONResponse({
        "image_id": image_id,
        "image_type": result["image_type"],
        "products": result["products"],
        "crops_base_url": f"/eyewear-crops/{image_id}",
    })


@router.get("/crops/{image_id}/{product_index}/{section}")
async def get_crop(image_id: str, product_index: int, section: str):
    """Return a specific section crop as an image file."""
    # Try PNG first (background-removed), then fall back to JPEG
    png_path = CROPS_DIR / image_id / f"product_{product_index}" / f"{section}.png"
    jpg_path = CROPS_DIR / image_id / f"product_{product_index}" / f"{section}.jpg"
    if png_path.exists():
        return FileResponse(str(png_path), media_type="image/png")
    if jpg_path.exists():
        return FileResponse(str(jpg_path), media_type="image/jpeg")
    return JSONResponse({"error": "crop not found"}, status_code=404)


@router.post("/retrain")
async def retrain(background_tasks: BackgroundTasks, model: str = "both"):
    """
    Trigger retraining in the background.
    model: "section_detector" | "product_finder" | "both"
    """
    from training import retrain_models
    background_tasks.add_task(retrain_models, model=model)
    return {"status": "training started", "model": model}


@router.get("/model/status")
async def model_status():
    """Returns which model version is currently active."""
    def active_version(name):
        p = Path(f"trained_models/{name}/active.txt")
        return p.read_text().strip() if p.exists() else "grounding_dino (zero-shot)"

    return {
        "section_detector": active_version("section_detector"),
        "product_finder": active_version("product_finder"),
    }
