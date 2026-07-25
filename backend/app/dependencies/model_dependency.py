"""
Walking Eye - AI Perception Engine
FastAPI Dependency Injection.

Provides the YOLO model and AnalysisService to route handlers
via FastAPI's Depends() mechanism.

This keeps routes thin — they declare what they need,
and the DI system wires it up automatically.
"""

from fastapi import Depends, HTTPException, Request, status
from ultralytics import YOLO

from app.services.analysis_service import AnalysisService
from app.utilities.logger import get_logger

logger = get_logger(__name__)


def get_model(request: Request) -> YOLO:
    """
    Retrieves the loaded YOLO model from application state.

    Injected into routes via FastAPI Depends().
    Raises 503 if the model isn't loaded (startup failure scenario).

    Args:
        request: FastAPI Request object (auto-injected by FastAPI).

    Returns:
        Loaded YOLO model instance.

    Raises:
        HTTPException 503: If the model is not available.
    """
    model_manager = getattr(request.app.state, "model_manager", None)

    if model_manager is None or not model_manager.is_loaded:
        logger.error("Model requested but ModelManager is not loaded.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI model is not available. The service may still be starting up.",
        )

    return model_manager.model


def get_analysis_service(
    model: YOLO = Depends(get_model),
) -> AnalysisService:
    """
    Constructs and returns an AnalysisService with the loaded model injected.

    A new AnalysisService instance is created per request — it's lightweight
    (no model loading), so this is safe and keeps things stateless.

    Args:
        model: Injected YOLO model via get_model dependency.

    Returns:
        Configured AnalysisService instance.
    """
    return AnalysisService(model=model)
