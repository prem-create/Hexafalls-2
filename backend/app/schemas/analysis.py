"""
Walking Eye - AI Perception Engine
Pydantic schemas for analysis request/response contracts.

These are the data shapes that cross the API boundary.
Keeping them separate from internal domain models means
the API contract can evolve independently of internal logic.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# ============================================================
# Sub-schemas (building blocks)
# ============================================================

class BoundingBox(BaseModel):
    """Pixel-space bounding box of a detected object."""

    x: int = Field(..., description="Left edge of the bounding box (pixels)")
    y: int = Field(..., description="Top edge of the bounding box (pixels)")
    width: int = Field(..., description="Width of the bounding box (pixels)")
    height: int = Field(..., description="Height of the bounding box (pixels)")


class Center(BaseModel):
    """Center point of a detected object's bounding box."""

    x: int = Field(..., description="Horizontal center (pixels)")
    y: int = Field(..., description="Vertical center (pixels)")


class DetectedObject(BaseModel):
    """
    Represents a single object detected in the image.

    Future fields to add here without breaking existing clients:
    - depth_estimate: float
    - ocr_text: str
    - track_id: int
    - attributes: dict
    """

    id: int = Field(..., description="Temporary unique ID within this response")
    label: str = Field(..., description="Object class label (e.g. 'chair')")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Detection confidence score between 0 and 1",
    )
    bbox: BoundingBox = Field(..., description="Bounding box in pixel coordinates")
    center: Center = Field(..., description="Center point of the bounding box")

    # --- Spatial awareness (direction + proximity) ---
    direction: Optional[str] = Field(
        default=None,
        description=(
            "Where the object sits in the frame: 'left', 'center-left', "
            "'center', 'center-right', or 'right'."
        ),
    )
    proximity: Optional[str] = Field(
        default=None,
        description=(
            "Estimated closeness based on how much of the frame the object "
            "occupies: 'far', 'medium', 'close', or 'very close'."
        ),
    )
    is_hazard: Optional[bool] = Field(
        default=None,
        description="True if this object is close and in the walking path.",
    )


# ============================================================
# Top-level response schema
# ============================================================

class AnalysisResponse(BaseModel):
    """
    Full response returned by POST /analyze.

    Designed to be stable — future capabilities (depth, OCR, tracking)
    are added as optional fields so existing Flutter clients keep working.
    """

    success: bool = Field(..., description="Whether analysis completed successfully")
    processing_time_ms: float = Field(
        ..., description="Total server-side processing time in milliseconds"
    )
    model_used: str = Field(..., description="Name/identifier of the model used")
    image_width: int = Field(..., description="Width of the processed image in pixels")
    image_height: int = Field(..., description="Height of the processed image in pixels")
    objects: List[DetectedObject] = Field(
        default_factory=list,
        description="List of all detected objects",
    )
    summary: str = Field(
        ..., description="Natural-language scene description"
    )

    # --- Future optional fields (non-breaking additions) ---
    object_count: int = Field(
        default=0, description="Total number of detected objects"
    )
    scene_tags: Optional[List[str]] = Field(
        default=None,
        description="High-level scene tags (future: scene classifier)",
    )
    hazard_detected: bool = Field(
        default=False,
        description="True if any detected object is close and in the walking path.",
    )
    suggested_direction: Optional[str] = Field(
        default=None,
        description=(
            "'left' or 'right' if that side has meaningfully more open space "
            "to move toward while a hazard blocks the path; null otherwise."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "success": True,
                "processing_time_ms": 48.5,
                "model_used": "YOLOv8n",
                "image_width": 1280,
                "image_height": 720,
                "object_count": 1,
                "summary": "There is a chair ahead.",
                "objects": [
                    {
                        "id": 1,
                        "label": "chair",
                        "confidence": 0.94,
                        "bbox": {"x": 120, "y": 240, "width": 220, "height": 330},
                        "center": {"x": 230, "y": 405},
                    }
                ],
                "scene_tags": None,
            }
        }
    }


# ============================================================
# Error response schema
# ============================================================

class ErrorResponse(BaseModel):
    """Standard error shape returned on 4xx/5xx responses."""

    success: bool = Field(default=False)
    error: str = Field(..., description="Human-readable error message")
    detail: Optional[str] = Field(
        default=None, description="Technical detail for debugging"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "success": False,
                "error": "Invalid image file.",
                "detail": "Could not decode image. File may be corrupted.",
            }
        }
    }


# ============================================================
# Batch analysis (future-ready stub)
# ============================================================

class BatchAnalysisResponse(BaseModel):
    """Response shape for POST /analyze/batch (future endpoint)."""

    success: bool
    total_images: int
    results: List[AnalysisResponse]
    errors: List[ErrorResponse] = Field(default_factory=list)
