"""
Walking Eye - AI Perception Engine
Main application entry point.

Handles FastAPI app creation, lifespan (startup/shutdown),
middleware registration, and router mounting.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import analysis, health
from app.config.settings import get_settings
from app.core.model_manager import ModelManager
from app.middleware.logging_middleware import LoggingMiddleware
from app.middleware.timing_middleware import TimingMiddleware
from app.utilities.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.
    Handles startup (model loading, tracker initialisation) and shutdown.
    All heavy objects are loaded ONCE and stored in app.state.
    """
    # --- STARTUP ---
    logger.info("Starting Walking Eye Perception Engine...")
    logger.info(f"Environment: {'DEBUG' if settings.DEBUG else 'PRODUCTION'}")

    # Load YOLO + optional MiDaS depth model
    model_manager = ModelManager(
        model_path=settings.MODEL_PATH,
        enable_depth=settings.ENABLE_DEPTH,
    )
    model_manager.load()
    app.state.model_manager = model_manager

    if settings.ENABLE_DEPTH:
        if model_manager.depth_enabled:
            logger.info("Depth estimation: ENABLED (MiDaS DPT_Hybrid)")
        else:
            logger.warning(
                "Depth estimation: REQUESTED but model failed to load — "
                "falling back to bbox-proxy mode."
            )
    else:
        logger.info("Depth estimation: DISABLED (ENABLE_DEPTH=False)")

    # Initialise the TrackerStore (session-level multi-object tracking)
    if settings.ENABLE_TRACKING:
        from app.tracking.tracker_store import TrackerStore
        tracker_store = TrackerStore(
            max_sessions=settings.MAX_TRACKER_SESSIONS,
            history_size=settings.HISTORY_SIZE,
            iou_threshold=settings.TRACKER_IOU_THRESHOLD,
            max_age=settings.TRACKER_MAX_AGE,
            min_track_history=settings.MIN_TRACK_HISTORY,
            stationary_depth_threshold=settings.STATIONARY_DEPTH_THRESHOLD,
            approaching_depth_threshold=settings.APPROACHING_DEPTH_THRESHOLD,
            stationary_scale_threshold=settings.STATIONARY_SCALE_THRESHOLD,
            approaching_scale_threshold=settings.APPROACHING_SCALE_THRESHOLD,
            direction_noise_threshold_px=settings.DIRECTION_NOISE_THRESHOLD_PX,
            # Alert manager settings
            alert_min_interval_s=settings.ALERT_MIN_INTERVAL_S,
            alert_zone_far_m=settings.ALERT_ZONE_FAR_M,
            alert_zone_medium_m=settings.ALERT_ZONE_MEDIUM_M,
            alert_zone_near_m=settings.ALERT_ZONE_NEAR_M,
            alert_distance_change_threshold_m=settings.ALERT_DISTANCE_CHANGE_THRESHOLD_M,
            alert_rapid_approach_threshold_ms=settings.ALERT_RAPID_APPROACH_THRESHOLD_MS,
            alert_track_disappear_frames=settings.ALERT_TRACK_DISAPPEAR_FRAMES,
        )
        app.state.tracker_store = tracker_store
        logger.info("Temporal tracking and motion analysis: ENABLED")
    else:
        app.state.tracker_store = None
        logger.info("Temporal tracking and motion analysis: DISABLED")

    logger.info("Application startup complete. Ready to serve.")

    yield

    # --- SHUTDOWN ---
    logger.info("Shutting down Walking Eye Perception Engine...")
    app.state.model_manager = None
    app.state.tracker_store = None
    logger.info("Shutdown complete.")


def create_app() -> FastAPI:
    """Application factory."""
    app = FastAPI(
        title="Walking Eye - Perception Engine",
        description=(
            "AI-powered backend for real-time object detection, depth estimation, "
            "scene understanding, and temporal motion analysis. "
            "Designed to serve as the perception layer for an AI assistant."
        ),
        version="1.2.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS — permissive for MVP/development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Custom middleware (last added = first executed)
    app.add_middleware(TimingMiddleware)
    app.add_middleware(LoggingMiddleware)

    # Routers
    app.include_router(health.router, tags=["Health"])
    app.include_router(analysis.router, prefix="/analyze", tags=["Analysis"])

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="debug" if settings.DEBUG else "info",
    )
