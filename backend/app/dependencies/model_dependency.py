"""
Walking Eye - AI Perception Engine
FastAPI Dependency Injection.

Provides the YOLO model, DepthEstimator, AnalysisService, and TrackerStore
to route handlers via FastAPI's Depends() mechanism.

This keeps routes thin — they declare what they need and the DI system
wires it up automatically.
"""

from fastapi import Depends, HTTPException, Request, status
from ultralytics import YOLO

from app.services.analysis_service import AnalysisService
from app.tracking.tracker_store import TrackerStore
from app.utilities.logger import get_logger

logger = get_logger(__name__)


def get_model(request: Request) -> YOLO:
    """
    Retrieves the loaded YOLO model from application state.
    Raises 503 if the model isn't loaded.
    """
    model_manager = getattr(request.app.state, "model_manager", None)

    if model_manager is None or not model_manager.is_loaded:
        logger.error("Model requested but ModelManager is not loaded.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI model is not available. The service may still be starting up.",
        )

    return model_manager.model


def get_depth_estimator(request: Request):
    """
    Retrieves the DepthEstimator from application state.
    Returns None when depth estimation is disabled or failed to load —
    callers degrade gracefully to bbox-proxy mode.
    """
    model_manager = getattr(request.app.state, "model_manager", None)
    if model_manager is None:
        return None
    return model_manager.depth_estimator


def get_tracker_store(request: Request) -> TrackerStore:
    """
    Retrieves the global TrackerStore from application state.
    Returns None when tracking is disabled.
    """
    return getattr(request.app.state, "tracker_store", None)


def get_analysis_service(
    model: YOLO = Depends(get_model),
    tracker_store: TrackerStore = Depends(get_tracker_store),
    depth_estimator=Depends(get_depth_estimator),
) -> AnalysisService:
    """
    Constructs and returns an AnalysisService with all dependencies injected.
    A new instance is created per request (lightweight — no model loading).
    """
    return AnalysisService(
        model=model,
        tracker_store=tracker_store,
        depth_estimator=depth_estimator,
    )
