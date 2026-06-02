"""Fast background removal (OpenCV + U2-NetP) package."""

from .pipeline import (
    remove_bg_hybrid,
    remove_bg_opencv,
    remove_bg_u2netp,
    process_folder,
)

__all__ = [
    "remove_bg_hybrid",
    "remove_bg_opencv",
    "remove_bg_u2netp",
    "process_folder",
]
