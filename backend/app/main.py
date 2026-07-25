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
    Handles startup (model loading) and shutdown (cleanup).
    YOLO model is loaded ONCE here and stored in app.state.
    """
    # --- STARTUP ---
    logger.info("Starting Walking Eye Perception Engine...")
    logger.info(f"Environment: {'DEBUG' if settings.DEBUG else 'PRODUCTION'}")

    model_manager = ModelManager(model_path=settings.MODEL_PATH)
    model_manager.load()
    app.state.model_manager = model_manager

    logger.info("Application startup complete. Ready to serve.")

    yield

    # --- SHUTDOWN ---
    logger.info("Shutting down Walking Eye Perception Engine...")
    app.state.model_manager = None
    logger.info("Shutdown complete.")


def create_app() -> FastAPI:
    """
    Application factory.
    Creates and configures the FastAPI instance.
    """
    app = FastAPI(
        title="Walking Eye - Perception Engine",
        description=(
            "AI-powered backend for real-time object detection and scene understanding. "
            "Designed to serve as the perception layer for an AI assistant."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # --- CORS ---
    # Permissive for MVP/development. Tighten per environment in production.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Custom Middleware (order matters: last added = first executed) ---
    app.add_middleware(TimingMiddleware)
    app.add_middleware(LoggingMiddleware)

    # --- Routers ---
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
