"""SQLModel database models and engine setup."""

from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel, create_engine, Session

DATABASE_URL = "sqlite:///./eyewear.db"
engine = create_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False})


class Image(SQLModel, table=True):
    id: str = Field(primary_key=True)  # UUID
    filename: str
    path: str
    image_type: Optional[str] = None  # "shelf"|"single"|"worn"|"closeup"
    type_confidence: Optional[float] = None
    status: str = "unlabeled"  # "unlabeled"|"labeled"|"needs_review"
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    labeled_at: Optional[datetime] = None


class Label(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    image_id: str = Field(foreign_key="image.id")
    label_mode: str  # "product" | "section"
    class_id: int
    label: str
    x1: float  # normalized 0-1
    y1: float
    x2: float
    y2: float
    source: str  # "claude"|"yolo"|"manual"|"extracted"
    confidence: Optional[float] = None


class Detection(SQLModel, table=True):
    id: str = Field(primary_key=True)  # UUID
    image_id: str = Field(foreign_key="image.id")
    product_index: int = 0
    label: str
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    crop_path: str = ""
    embedding_id: Optional[str] = None
    top_match_id: Optional[str] = None
    top_match_score: Optional[float] = None


class ModelVersion(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    model_type: str  # "product_detector" | "section_detector"
    version: int
    weights_path: str
    map50: float
    trained_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = False


def init_db():
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session
