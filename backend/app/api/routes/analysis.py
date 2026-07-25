"""
Walking Eye - AI Perception Engine
Analysis Routes.

POST /analyze       → Single image analysis
POST /analyze/batch → Batch analysis (future-ready stub)

Routes are intentionally thin — they only handle:
- Input validation
- Calling the service
- Returning the response or a proper error

No business logic lives here.
"""

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.config.settings import get_settings
from app.dependencies.model_dependency import get_analysis_service
from app.schemas.analysis import AnalysisResponse, BatchAnalysisResponse, ErrorResponse
from app.services.analysis_service import AnalysisService
from app.utilities.logger import get_logger
from app.vision.image_processor import ImageProcessingError
from app.vision.detector import DetectionError

logger = get_logger(__name__)
settings = get_settings()

router = APIRouter()


@router.post(
    "",
    response_model=AnalysisResponse,
    summary="Analyze Image",
    description=(
        "Receives an image file, detects all visible objects using YOLO, "
        "and returns structured detections with a natural-language scene summary."
    ),
    responses={
        200: {"model": AnalysisResponse},
        400: {"model": ErrorResponse, "description": "Invalid or corrupted image"},
        422: {"description": "Missing or malformed request body"},
        503: {"model": ErrorResponse, "description": "AI model not available"},
    },
)
async def analyze_image(
    image: UploadFile = File(..., description="Image file (JPEG or PNG)"),
    service: AnalysisService = Depends(get_analysis_service),
) -> AnalysisResponse:
    """
    Main perception endpoint.

    Accepts a multipart/form-data upload with field name 'image'.
    Returns detected objects and a scene summary.
    """
    # --- Validate content type ---
    if image.content_type not in settings.ALLOWED_CONTENT_TYPES and \
       image.content_type != "application/octet-stream":
        logger.warning(
            f"Rejected upload: unsupported content type '{image.content_type}'"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file type: '{image.content_type}'. "
                f"Allowed: {', '.join(settings.ALLOWED_CONTENT_TYPES)}"
            ),
        )

    # --- Read image bytes ---
    image_bytes = await image.read()

    if not image_bytes:
        logger.warning("Rejected upload: empty file received.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # --- Validate file size ---
    if len(image_bytes) > settings.max_image_bytes:
        logger.warning(
            f"Rejected upload: file size {len(image_bytes)} bytes "
            f"exceeds limit of {settings.max_image_bytes} bytes."
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"File too large. Maximum allowed size is "
                f"{settings.MAX_IMAGE_SIZE_MB} MB."
            ),
        )

    logger.info(
        f"Received image | filename: '{image.filename}' | "
        f"type: {image.content_type} | size: {len(image_bytes)} bytes"
    )

    # --- Run analysis pipeline ---
    try:
        result = await service.analyze(image_bytes)
        return result

    except ImageProcessingError as e:
        logger.warning(f"Image processing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    except DetectionError as e:
        logger.error(f"Detection pipeline failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Object detection failed. Please try again.",
        )

    except Exception as e:
        logger.error(f"Unexpected error during analysis: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred. Please try again.",
        )


@router.post(
    "/batch",
    response_model=BatchAnalysisResponse,
    summary="Batch Image Analysis (Future)",
    description=(
        "Accepts multiple images for batch processing. "
        "Currently returns a not-implemented response. "
        "Architecture is in place for future activation."
    ),
    responses={
        501: {"description": "Not yet implemented"},
    },
)
async def analyze_batch(
    service: AnalysisService = Depends(get_analysis_service),
) -> BatchAnalysisResponse:
    """
    Batch analysis endpoint stub.
    Endpoint is registered and documented; implementation is a future task.
    Returns 501 until activated.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail=(
            "Batch analysis is not yet implemented. "
            "Use POST /analyze for single-image processing."
        ),
    )
