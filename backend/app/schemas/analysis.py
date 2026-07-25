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


# ============================================================
# Motion analysis sub-schemas
# ============================================================

class DistanceInfo(BaseModel):
    """
    Distance information for a tracked object.

    When metric depth is unavailable, value and unit are None and source
    is 'relative_bbox_scale' to make it explicit that no real measurement
    was performed.
    """

    value: Optional[float] = Field(
        default=None,
        description="Distance in metres (metric depth) or None when unavailable.",
    )
    unit: Optional[str] = Field(
        default=None,
        description="'meters' when metric depth is available, otherwise null.",
    )
    source: str = Field(
        ...,
        description=(
            "'depth' when derived from a depth sensor/model, "
            "'relative_bbox_scale' when bbox size is used as a proxy."
        ),
    )


class VelocityInfo(BaseModel):
    """
    Velocity information.

    For metric velocity (depth available): value in m/s, type='metric'.
    For bbox-scale fallback: value=None, type='relative',
      relative_approach_speed gives a px²/s proxy value.
    """

    value: Optional[float] = Field(
        default=None,
        description="Speed in m/s when metric depth is available, otherwise null.",
    )
    unit: Optional[str] = Field(
        default=None,
        description="'m/s' when metric, otherwise null.",
    )
    type: str = Field(
        ...,
        description="'metric' when depth-derived, 'relative' when bbox-proxy.",
    )
    relative_approach_speed: Optional[float] = Field(
        default=None,
        description=(
            "Bounding-box area change per second (px²/s). "
            "Only present when type='relative'. Not a physical velocity."
        ),
    )


class MotionInfo(BaseModel):
    """
    Temporal motion analysis result for a tracked object.

    Populated when ENABLE_TRACKING is True and the object has been observed
    across at least MIN_TRACK_HISTORY frames.  None on the first frame or
    when tracking is disabled.
    """

    state: str = Field(
        ...,
        description=(
            "Motion state: 'APPROACHING', 'MOVING_AWAY', 'STATIONARY', or 'UNKNOWN'."
        ),
    )
    direction: str = Field(
        ...,
        description=(
            "Horizontal movement direction in the image plane: "
            "'LEFT', 'RIGHT', or 'CENTER'."
        ),
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Confidence score for the motion classification (0–1). "
            "Low when few observations are available."
        ),
    )
    distance: DistanceInfo = Field(..., description="Distance/depth information.")
    velocity: VelocityInfo = Field(..., description="Velocity information.")
    observations_used: int = Field(
        ...,
        description="Number of temporal observations used to compute this result.",
    )


# ============================================================
# Alert schema  (defined before AnalysisResponse — no forward ref needed)
# ============================================================

class AlertInfo(BaseModel):
    """
    A single alert event produced by the Alert Manager for one tracked object.

    The frontend should speak `message` verbatim when `should_speak` is True.
    All other fields are available for UI display or logging.
    """

    track_id: int = Field(..., description="Track ID of the object that triggered this alert.")
    label: str = Field(..., description="Object class label (e.g. 'person').")
    alert_type: str = Field(
        ...,
        description=(
            "What triggered this alert: 'NEW_OBJECT', 'ZONE_CROSSING', "
            "'MOTION_STATE_CHANGE', 'RAPID_APPROACH', 'DISTANCE_UPDATE', "
            "'PERIODIC_REMINDER', or 'OBJECT_GONE'."
        ),
    )
    priority: str = Field(
        ...,
        description="Alert urgency: 'HIGH', 'MEDIUM', or 'LOW'.",
    )
    message: str = Field(
        ...,
        description="Natural-language alert sentence ready to speak aloud.",
    )
    should_speak: bool = Field(
        ...,
        description=(
            "True when this alert represents a meaningful new event that "
            "warrants speech output. The frontend should only speak when this is True."
        ),
    )
    distance_m: Optional[float] = Field(
        default=None,
        description="Metric distance in metres at the time of the alert, if available.",
    )
    zone: Optional[str] = Field(
        default=None,
        description=(
            "Distance zone at alert time: 'VERY_NEAR', 'NEAR', 'MEDIUM', "
            "'FAR', or 'UNKNOWN'."
        ),
    )
    motion_state: Optional[str] = Field(
        default=None,
        description=(
            "Motion state at alert time: 'APPROACHING', 'MOVING_AWAY', "
            "'STATIONARY', or 'UNKNOWN'."
        ),
    )
    velocity_ms: Optional[float] = Field(
        default=None,
        description="Radial velocity in m/s at alert time, if available.",
    )


# ============================================================
# Main object schema
# ============================================================

class DetectedObject(BaseModel):
    """
    Represents a single object detected in the image.
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

    # --- Temporal tracking ---
    track_id: Optional[int] = Field(
        default=None,
        description=(
            "Persistent track ID assigned by the IoU tracker. "
            "Stable across consecutive frames for the same physical object. "
            "None when tracking is disabled."
        ),
    )

    # --- Motion analysis ---
    motion: Optional[MotionInfo] = Field(
        default=None,
        description=(
            "Temporal motion analysis result. "
            "None on the first frame, when tracking is disabled, "
            "or when insufficient history exists."
        ),
    )


# ============================================================
# Top-level response schema
# ============================================================

class AnalysisResponse(BaseModel):
    """
    Full response returned by POST /analyze.

    Designed to be stable — new optional fields are added without breaking
    existing Flutter clients.
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

    # --- Optional / future fields ---
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

    # --- Tracking session ---
    session_id: Optional[str] = Field(
        default=None,
        description=(
            "Echo of the session_id supplied by the client. "
            "Use the same value in subsequent requests to maintain tracking continuity."
        ),
    )
    tracking_enabled: bool = Field(
        default=False,
        description="True when temporal tracking and motion analysis are active.",
    )

    # --- Intelligent alerts ---
    alerts: List[AlertInfo] = Field(
        default_factory=list,
        description=(
            "Structured alert events produced by the Alert Manager. "
            "Sorted HIGH → MEDIUM → LOW. "
            "Speak only the message of alerts where should_speak=True."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "success": True,
                "processing_time_ms": 148.5,
                "model_used": "YOLO11n + MiDaS DPT_Hybrid",
                "image_width": 1280,
                "image_height": 720,
                "object_count": 1,
                "summary": "Caution, person ahead, 3.2 metres — approaching!",
                "tracking_enabled": True,
                "session_id": "flutter-cam-001",
                "hazard_detected": True,
                "alerts": [
                    {
                        "track_id": 7,
                        "label": "person",
                        "alert_type": "ZONE_CROSSING",
                        "priority": "MEDIUM",
                        "message": "Person now nearby, approximately 3.2 metres away, approaching.",
                        "should_speak": True,
                        "distance_m": 3.2,
                        "zone": "NEAR",
                        "motion_state": "APPROACHING",
                        "velocity_ms": 1.2,
                    }
                ],
                "objects": [
                    {
                        "id": 1,
                        "track_id": 7,
                        "label": "person",
                        "confidence": 0.91,
                        "bbox": {"x": 100, "y": 120, "width": 150, "height": 380},
                        "center": {"x": 175, "y": 310},
                        "direction": "center",
                        "proximity": "close",
                        "is_hazard": True,
                    }
                ],
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
# Batch analysis
# ============================================================

class BatchImageResult(BaseModel):
    """
    Result for a single image within a batch request.
    Wraps either a successful AnalysisResponse or an error, keyed by the
    original filename so clients can correlate results back to their uploads.
    """

    index: int = Field(..., description="0-based position of this image in the batch upload")
    filename: str = Field(..., description="Original filename of the uploaded image")
    success: bool = Field(..., description="Whether analysis succeeded for this image")
    result: Optional[AnalysisResponse] = Field(
        default=None,
        description="Full analysis result when success=True",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message when success=False",
    )


class BatchAnalysisResponse(BaseModel):
    """Response shape for POST /analyze/batch."""

    success: bool = Field(..., description="True when at least one image was processed without error")
    total_images: int = Field(..., description="Number of images received in the request")
    succeeded: int = Field(..., description="Number of images analysed successfully")
    failed: int = Field(..., description="Number of images that could not be analysed")
    total_processing_time_ms: float = Field(
        ..., description="Wall-clock time for the entire batch (milliseconds)"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Echo of the session_id supplied by the client (tracking continuity)",
    )
    results: List[BatchImageResult] = Field(
        default_factory=list,
        description="Per-image results in the same order as the uploaded files",
    )
