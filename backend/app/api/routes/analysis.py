"""
Walking Eye - AI Perception Engine
Analysis Routes.

POST /analyze       → Single image analysis (with optional temporal tracking)
POST /analyze/batch → Batch image analysis — runs the full pipeline on each
                      uploaded image and returns per-image results

Routes are intentionally thin — they only handle:
- Input validation
- Calling the service
- Returning the response or a proper error

No business logic lives here.

Temporal tracking
-----------------
Clients supply a `session_id` query parameter to opt-in to tracking.
All frames sent with the same session_id share a tracker and history buffer,
enabling motion analysis across frames.  Without session_id the response
is identical to the pre-tracking behaviour (fully backwards-compatible).
"""

import time
from typing import Annotated, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile, status

from app.config.settings import get_settings
from app.dependencies.model_dependency import get_analysis_service
from app.schemas.analysis import (
    AnalysisResponse,
    BatchAnalysisResponse,
    BatchImageResult,
    ErrorResponse,
)
from app.services.analysis_service import AnalysisService
from app.utilities.logger import get_logger
from app.vision.image_processor import ImageProcessingError
from app.vision.detector import DetectionError

logger = get_logger(__name__)
settings = get_settings()

router = APIRouter()

# Maximum images accepted in a single batch request
_BATCH_MAX_IMAGES = 20


# ===========================================================================
# POST /analyze  — single image
# ===========================================================================

@router.post(
    "",
    response_model=AnalysisResponse,
    summary="Analyze Image",
    description=(
        "Receives an image file, detects all visible objects using YOLO, "
        "and returns structured detections with a natural-language scene summary. "
        "Supply `session_id` to enable temporal tracking and motion analysis across frames."
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
    session_id: Optional[str] = Query(
        default=None,
        description=(
            "Client-assigned session identifier for temporal tracking continuity. "
            "Use the same value across consecutive frames from the same camera feed "
            "to enable object tracking and motion analysis. "
            "Omit to get a single-frame response (no tracking, fully backwards-compatible)."
        ),
        max_length=128,
    ),
    service: AnalysisService = Depends(get_analysis_service),
) -> AnalysisResponse:
    """
    Main perception endpoint.

    Accepts a multipart/form-data upload with field name 'image'.
    Returns detected objects, motion info (when session_id is supplied),
    and a scene summary.
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
        + (f" | session_id: '{session_id}'" if session_id else "")
    )

    # --- Run analysis pipeline ---
    try:
        result = await service.analyze(
            image_bytes=image_bytes,
            session_id=session_id,
            timestamp=time.time(),
            # estimated_depths: None — depth estimation not implemented yet.
            # When a depth model is added (ENABLE_DEPTH=True), pass the
            # per-detection depth list here.
            estimated_depths=None,
        )
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


# ===========================================================================
# POST /analyze/batch  — multiple images
# ===========================================================================

@router.post(
    "/batch",
    response_model=BatchAnalysisResponse,
    summary="Batch Image Analysis",
    description=(
        f"Accepts up to {_BATCH_MAX_IMAGES} image files in a single request and runs "
        "the full YOLO detection + spatial awareness + scene reasoning pipeline on each one. "
        "Results are returned in the same order as the uploaded files. "
        "Images that fail validation or processing are recorded as errors without "
        "aborting the rest of the batch. "
        "Supply `session_id` to enable temporal tracking across all images in the batch — "
        "useful when the images are consecutive frames from the same camera feed."
    ),
    responses={
        200: {"model": BatchAnalysisResponse},
        400: {"model": ErrorResponse, "description": "No valid images supplied"},
        503: {"model": ErrorResponse, "description": "AI model not available"},
    },
    # Override the OpenAPI body schema so Swagger UI renders proper file-picker
    # buttons instead of text boxes for the 'images' field.
    openapi_extra={
        "requestBody": {
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "images": {
                                "type": "array",
                                "items": {"type": "string", "format": "binary"},
                                "description": (
                                    f"One or more image files (JPEG or PNG). "
                                    f"Maximum {_BATCH_MAX_IMAGES} per request."
                                ),
                            }
                        },
                        "required": ["images"],
                    }
                }
            },
            "required": True,
        }
    },
)
async def analyze_batch(
    request: Request,
    session_id: Optional[str] = Query(
        default=None,
        description=(
            "Optional session identifier shared across all images in this batch. "
            "When supplied, temporal tracking and motion analysis are applied "
            "across the batch (images are treated as consecutive frames). "
            "Omit for independent single-frame analysis of each image."
        ),
        max_length=128,
    ),
    service: AnalysisService = Depends(get_analysis_service),
) -> BatchAnalysisResponse:
    """
    Batch perception endpoint.

    Accepts multiple images via multipart/form-data (field name 'images').
    Reads uploads directly from the raw request form so that both Swagger UI
    and programmatic clients (curl, Flutter) work correctly.
    """
    batch_start = time.perf_counter()

    # --- Parse multipart form directly from the request ---
    try:
        form = await request.form()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not parse multipart form data: {e}",
        )

    # Collect all files uploaded under the 'images' field name.
    # The form may contain multiple values with the same key.
    uploads: List[UploadFile] = []
    for key, value in form.multi_items():
        if key == "images" and isinstance(value, UploadFile):
            uploads.append(value)

    if not uploads:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "No image files found. "
                "Send files as multipart/form-data with field name 'images'."
            ),
        )

    if len(uploads) > _BATCH_MAX_IMAGES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Too many images. Maximum allowed per batch is {_BATCH_MAX_IMAGES}, "
                f"received {len(uploads)}."
            ),
        )

    logger.info(
        f"Batch request: {len(uploads)} image(s)"
        + (f" | session_id: '{session_id}'" if session_id else "")
    )

    results: List[BatchImageResult] = []
    base_timestamp = time.time()

    for idx, upload in enumerate(uploads):
        filename = upload.filename or f"image_{idx}"
        frame_timestamp = base_timestamp + idx * 0.5   # 0.5 s ≈ 2 FPS spacing

        # ---- Validate content type ----
        if upload.content_type not in settings.ALLOWED_CONTENT_TYPES and \
           upload.content_type != "application/octet-stream":
            logger.warning(
                f"Batch[{idx}] '{filename}': rejected — "
                f"unsupported content type '{upload.content_type}'"
            )
            results.append(BatchImageResult(
                index=idx,
                filename=filename,
                success=False,
                error=(
                    f"Unsupported file type: '{upload.content_type}'. "
                    f"Allowed: {', '.join(settings.ALLOWED_CONTENT_TYPES)}"
                ),
            ))
            continue

        image_bytes = await upload.read()

        if not image_bytes:
            results.append(BatchImageResult(
                index=idx,
                filename=filename,
                success=False,
                error="Uploaded file is empty.",
            ))
            continue

        if len(image_bytes) > settings.max_image_bytes:
            results.append(BatchImageResult(
                index=idx,
                filename=filename,
                success=False,
                error=(
                    f"File too large. Maximum allowed size is "
                    f"{settings.MAX_IMAGE_SIZE_MB} MB."
                ),
            ))
            continue

        # ---- Run full pipeline ----
        try:
            analysis = await service.analyze(
                image_bytes=image_bytes,
                session_id=session_id,
                timestamp=frame_timestamp,
                estimated_depths=None,
            )
            results.append(BatchImageResult(
                index=idx,
                filename=filename,
                success=True,
                result=analysis,
            ))
            logger.info(
                f"Batch[{idx}] '{filename}': OK — "
                f"{analysis.object_count} object(s), "
                f"{analysis.processing_time_ms:.1f} ms"
            )

        except (ImageProcessingError, DetectionError) as e:
            logger.warning(f"Batch[{idx}] '{filename}': pipeline error — {e}")
            results.append(BatchImageResult(
                index=idx,
                filename=filename,
                success=False,
                error=str(e),
            ))

        except Exception as e:
            logger.error(
                f"Batch[{idx}] '{filename}': unexpected error — {e}",
                exc_info=True,
            )
            results.append(BatchImageResult(
                index=idx,
                filename=filename,
                success=False,
                error="An unexpected error occurred while processing this image.",
            ))

    succeeded = sum(1 for r in results if r.success)
    failed = len(results) - succeeded
    total_ms = (time.perf_counter() - batch_start) * 1000

    logger.info(
        f"Batch complete: {succeeded}/{len(uploads)} succeeded, "
        f"{failed} failed | total {total_ms:.1f} ms"
    )

    return BatchAnalysisResponse(
        success=succeeded > 0,
        total_images=len(uploads),
        succeeded=succeeded,
        failed=failed,
        total_processing_time_ms=round(total_ms, 2),
        session_id=session_id,
        results=results,
    )
