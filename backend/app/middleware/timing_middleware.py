"""
Walking Eye - AI Perception Engine
Timing Middleware.

Adds an X-Process-Time-Ms header to every response.
Useful for Flutter client performance monitoring and debugging.
"""

import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class TimingMiddleware(BaseHTTPMiddleware):
    """
    Injects X-Process-Time-Ms into every response header.
    Measures wall-clock time from request receipt to response send.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"
        return response
