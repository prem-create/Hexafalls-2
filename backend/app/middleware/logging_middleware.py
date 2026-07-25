"""
Walking Eye - AI Perception Engine
Request Logging Middleware.

Logs every incoming request and outgoing response.
Provides a consistent audit trail across all endpoints.
"""

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.utilities.logger import get_logger

logger = get_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs request method, path, status code, and duration for every call.
    Assigns a unique request ID to each request for traceability.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())[:8]  # Short ID for readability
        method = request.method
        path = request.url.path

        logger.info(f"[{request_id}] → {method} {path}")

        start = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.error(
                f"[{request_id}] ✗ {method} {path} | "
                f"UNHANDLED ERROR: {e} | {duration_ms:.1f} ms"
            )
            raise

        duration_ms = (time.perf_counter() - start) * 1000
        status_code = response.status_code

        log_fn = logger.warning if status_code >= 400 else logger.info
        log_fn(
            f"[{request_id}] ← {method} {path} | "
            f"status: {status_code} | {duration_ms:.1f} ms"
        )

        # Attach request ID to response headers for client-side tracing
        response.headers["X-Request-ID"] = request_id
        return response
