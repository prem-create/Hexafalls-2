"""
Walking Eye - AI Perception Engine
Centralized logging configuration.

Provides a consistent logger factory used across all modules.
Logs to both console and rotating file.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.config.settings import get_settings

settings = get_settings()

# Track which loggers have already been configured to avoid duplicate handlers
_configured_loggers: set[str] = set()


def _ensure_log_dir() -> Path:
    """Create the log directory if it doesn't exist."""
    log_dir = Path(settings.LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _build_formatter() -> logging.Formatter:
    """Returns a consistent log formatter for all handlers."""
    return logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger for the given module name.

    Usage:
        from app.utilities.logger import get_logger
        logger = get_logger(__name__)

    Args:
        name: Typically __name__ of the calling module.

    Returns:
        A fully configured Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if logger was already configured
    if name in _configured_loggers:
        return logger

    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(log_level)

    formatter = _build_formatter()

    # --- Console Handler ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # --- Rotating File Handler ---
    try:
        log_dir = _ensure_log_dir()
        file_handler = RotatingFileHandler(
            filename=log_dir / "walking_eye.log",
            maxBytes=10 * 1024 * 1024,  # 10 MB per file
            backupCount=5,              # Keep last 5 rotated files
            encoding="utf-8",
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError as e:
        # If file logging fails (e.g. read-only fs), warn but continue
        logger.warning(f"Could not set up file logging: {e}")

    # Prevent log messages from propagating to the root logger (avoids duplicates)
    logger.propagate = False

    _configured_loggers.add(name)
    return logger
