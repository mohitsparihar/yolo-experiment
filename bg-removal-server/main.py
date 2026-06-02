"""FastAPI server: image in → PNG with background removed (RGBA)."""

from __future__ import annotations

import io
import sys
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL import Image as PILImage

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

# Local package lives in ../bg-removal-fast
_BG_FAST_ROOT = Path(__file__).resolve().parent.parent / "bg-removal-fast"
if _BG_FAST_ROOT.is_dir() and str(_BG_FAST_ROOT) not in sys.path:
    sys.path.insert(0, str(_BG_FAST_ROOT))

from bg_removal_fast.pipeline import (  # noqa: E402
    remove_bg_hybrid,
    remove_bg_opencv,
    remove_bg_u2netp,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"

_rembg_sessions: dict[str, object] = {}


class RemovalMode(str, Enum):
    rembg = "rembg"
    carvekit = "carvekit"
    hybrid = "hybrid"
    opencv = "opencv"
    u2netp = "u2netp"


app = FastAPI(title="Background removal API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def serve_ui():
    index = STATIC_DIR / "index.html"
    if not index.is_file():
        raise HTTPException(status_code=404, detail="UI not found (static/index.html missing).")
    return FileResponse(index)


def _prepare_rembg_rgb(image: "PILImage.Image") -> "PILImage.Image":
    """RGB image for segmentation. RGBA is composited on white so the model sees opaque edges."""
    from PIL import Image

    if image.mode == "RGBA":
        base = Image.new("RGB", image.size, (255, 255, 255))
        base.paste(image, mask=image.split()[3])
        return base
    return image.convert("RGB")


def remove_bg_rembg(
    image: "PILImage.Image",
    model_name: str,
    max_input_side: int,
    *,
    alpha_matting: bool = False,
    alpha_matting_foreground_threshold: int = 240,
    alpha_matting_background_threshold: int = 10,
    alpha_matting_erode_size: int = 10,
) -> "PILImage.Image":
    """Remove background via rembg (downloads model on first use)."""
    from PIL import Image
    from rembg import remove as rembg_remove, new_session

    if model_name not in _rembg_sessions:
        _rembg_sessions[model_name] = new_session(model_name)
    session = _rembg_sessions[model_name]

    original = image
    work = _prepare_rembg_rgb(original)
    w, h = work.size
    max_dim = max(w, h)

    common_kw: dict[str, Any] = {"session": session, "post_process_mask": True}
    if alpha_matting:
        common_kw["alpha_matting"] = True
        common_kw["alpha_matting_foreground_threshold"] = alpha_matting_foreground_threshold
        common_kw["alpha_matting_background_threshold"] = alpha_matting_background_threshold
        common_kw["alpha_matting_erode_size"] = alpha_matting_erode_size

    if max_dim > max_input_side:
        scale = max_input_side / max_dim
        small = work.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
        small_result = rembg_remove(small, **common_kw)
        alpha = small_result.split()[-1].resize((w, h), Image.Resampling.LANCZOS)
        # Match colors to `work` (what the model saw), not raw RGBA originals.
        result = work.convert("RGBA")
        result.putalpha(alpha)
        return result

    return rembg_remove(work, **common_kw)


def _normalize_carve_object_type(raw: str) -> str:
    """CarveKit expects 'object' or 'hairs-like'."""
    s = (raw or "object").strip().lower().replace("_", "-")
    if s in ("hairs", "hair", "hairs-like"):
        return "hairs-like"
    return "object"


@lru_cache(maxsize=16)
def _carvekit_interface(
    object_type: str,
    batch_size_seg: int,
    batch_size_matting: int,
    device: str,
    seg_mask_size: int,
    matting_mask_size: int,
    trimap_prob_threshold: int,
    trimap_dilation: int,
    trimap_erosion_iters: int,
    fp16: bool,
):
    from carvekit.api.high import HiInterface

    return HiInterface(
        object_type=object_type,
        batch_size_seg=batch_size_seg,
        batch_size_matting=batch_size_matting,
        device=device,
        seg_mask_size=seg_mask_size,
        matting_mask_size=matting_mask_size,
        trimap_prob_threshold=trimap_prob_threshold,
        trimap_dilation=trimap_dilation,
        trimap_erosion_iters=trimap_erosion_iters,
        fp16=fp16,
    )


def remove_bg_carvekit(
    image: "PILImage.Image",
    *,
    object_type: str = "object",
    batch_size_seg: int = 5,
    batch_size_matting: int = 1,
    device: str = "cpu",
    seg_mask_size: int = 640,
    matting_mask_size: int = 2048,
    trimap_prob_threshold: int = 231,
    trimap_dilation: int = 30,
    trimap_erosion_iters: int = 5,
    fp16: bool = False,
) -> "PILImage.Image":
    """E-commerce oriented background removal (Tracer B7 + FBA matting)."""
    ot = _normalize_carve_object_type(object_type)
    iface = _carvekit_interface(
        ot,
        batch_size_seg,
        batch_size_matting,
        device,
        seg_mask_size,
        matting_mask_size,
        trimap_prob_threshold,
        trimap_dilation,
        trimap_erosion_iters,
        fp16,
    )
    return iface([image])[0]


def _pil_from_upload(file: UploadFile) -> "PILImage.Image":
    from PIL import Image

    raw = file.file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")
    try:
        return Image.open(io.BytesIO(raw))
    except Exception as exc:  # PIL.UnidentifiedImageError or OSError
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}") from exc


@app.post("/remove-background")
def remove_background(
    file: UploadFile = File(..., description="Input image (JPEG, PNG, WebP, etc.)"),
    mode: RemovalMode = Query(
        RemovalMode.carvekit,
        description="carvekit (default, e‑commerce matting), rembg, or hybrid/opencv/u2netp (bg-removal-fast).",
    ),
    max_side: int = Query(
        2048,
        ge=64,
        le=4096,
        description="Max longest side in pixels before rembg/U2-Net downscale (higher = sharper, slower, more RAM).",
    ),
    rembg_model: str = Query(
        "isnet-general-use",
        description="rembg model when mode=rembg. isnet-general-use is stronger than u2netp; birefnet-general is slowest/best.",
    ),
    alpha_matting: bool = Query(
        False,
        description="rembg only: softer edges (hair/fur/feathers); slower and more CPU.",
    ),
    alpha_matting_foreground_threshold: int = Query(240, ge=0, le=255),
    alpha_matting_background_threshold: int = Query(10, ge=0, le=255),
    alpha_matting_erode_size: int = Query(10, ge=1, le=40),
    carve_object_type: str = Query(
        "object",
        description='carvekit: "object" (products) or "hairs-like" / "hairs" (fine edges).',
    ),
    carve_device: str = Query("cpu", description="carvekit: cpu or cuda (if available)."),
    carve_seg_mask_size: int = Query(640, ge=320, le=1024, description="carvekit segmentation input size."),
    carve_matting_mask_size: int = Query(
        2048, ge=1024, le=4096, description="carvekit matting resolution (higher = sharper, more RAM)."
    ),
    carve_trimap_prob_threshold: int = Query(231, ge=0, le=255),
    carve_trimap_dilation: int = Query(30, ge=1, le=80),
    carve_trimap_erosion_iters: int = Query(5, ge=1, le=20),
    carve_batch_size_seg: int = Query(5, ge=1, le=16),
    carve_batch_size_matting: int = Query(1, ge=1, le=4),
    carve_fp16: bool = Query(False, description="carvekit: half precision on GPU."),
):
    """Return a PNG with transparency where the background was removed."""
    image = _pil_from_upload(file)

    try:
        if mode == RemovalMode.rembg:
            model = rembg_model.strip() or "isnet-general-use"
            result = remove_bg_rembg(
                image,
                model_name=model,
                max_input_side=max_side,
                alpha_matting=alpha_matting,
                alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
                alpha_matting_background_threshold=alpha_matting_background_threshold,
                alpha_matting_erode_size=alpha_matting_erode_size,
            )
        elif mode == RemovalMode.carvekit:
            result = remove_bg_carvekit(
                image,
                object_type=carve_object_type,
                batch_size_seg=carve_batch_size_seg,
                batch_size_matting=carve_batch_size_matting,
                device=carve_device.strip() or "cpu",
                seg_mask_size=carve_seg_mask_size,
                matting_mask_size=carve_matting_mask_size,
                trimap_prob_threshold=carve_trimap_prob_threshold,
                trimap_dilation=carve_trimap_dilation,
                trimap_erosion_iters=carve_trimap_erosion_iters,
                fp16=carve_fp16,
            )
        elif mode == RemovalMode.opencv:
            result = remove_bg_opencv(image)
        elif mode == RemovalMode.u2netp:
            result = remove_bg_u2netp(image, model_path=None, max_side=max_side)
        else:
            result, _quality, _fb = remove_bg_hybrid(image, model_path=None, max_side=max_side)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc)
            + " For hybrid/u2netp, download the model: "
            + "cd bg-removal-fast && python scripts/setup_models.py",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Processing failed: {exc}") from exc

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    buf.seek(0)

    base = Path(file.filename or "image").stem or "output"
    out_name = f"{base}_nobg.png"

    return StreamingResponse(
        buf,
        media_type="image/png",
        headers={"Content-Disposition": f'inline; filename="{out_name}"'},
    )
