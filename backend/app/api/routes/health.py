"""
Walking Eye - AI Perception Engine
Health & Root Routes.

GET /        → Server info
GET /health  → Detailed health check (model status, uptime, config)
"""

import time
from datetime import datetime, timezone

from fastapi import APIRouter, Request

from app.config.settings import get_settings
from app.utilities.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

router = APIRouter()

# Record startup time for uptime calculation
_startup_time: float = time.time()


@router.get("/", summary="Server Info", tags=["Health"])
async def root():
    """
    Returns basic server identification info.
    Useful as a quick ping to confirm the backend is reachable.
    """
    return {
        "name": "Walking Eye - Perception Engine",
        "version": "1.0.0",
        "description": "AI-powered object detection and scene understanding backend.",
        "docs": "/docs",
        "health": "/health",
    }


@router.get("/health", summary="Health Check", tags=["Health"])
async def health_check(request: Request):
    """
    Returns detailed health status including model availability and uptime.
    Flutter app can poll this on startup to confirm backend readiness.
    """
    uptime_seconds = round(time.time() - _startup_time, 1)

    # Check model state from app.state
    model_manager = getattr(request.app.state, "model_manager", None)
    model_loaded = model_manager is not None and model_manager.is_loaded
    model_info = model_manager.get_model_info() if model_loaded else {}

    status = "healthy" if model_loaded else "degraded"

    logger.debug(f"Health check | status: {status} | uptime: {uptime_seconds}s")

    return {
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "uptime_seconds": uptime_seconds,
        "model": {
            "loaded": model_loaded,
            **model_info,
        },
        "config": {
            "confidence_threshold": settings.CONFIDENCE_THRESHOLD,
            "max_image_dimension": settings.MAX_IMAGE_DIMENSION,
            "max_image_size_mb": settings.MAX_IMAGE_SIZE_MB,
            "debug": settings.DEBUG,
        },
    }
