# ============================================================
# Walking Eye - AI Perception Engine
# Dockerfile
#
# Multi-stage build:
#   Stage 1 (builder): installs dependencies into a venv
#   Stage 2 (runtime): copies only the venv + app code
#
# Result: lean final image with no build tools or cache.
# ============================================================

# --- Stage 1: Builder ---
FROM python:3.11-slim AS builder

# Install OS-level build dependencies needed by OpenCV and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Create isolated virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# --- Stage 2: Runtime ---
FROM python:3.11-slim AS runtime

LABEL maintainer="Walking Eye Project"
LABEL description="AI Perception Engine - Object Detection Backend"
LABEL version="1.0.0"

# Install only runtime OS dependencies (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# Set working directory
WORKDIR /app

# Create required directories with correct ownership
RUN mkdir -p logs uploads models && \
    chown -R appuser:appuser /app

# Copy application source code
COPY --chown=appuser:appuser app/ ./app/

# Switch to non-root user
USER appuser

# Expose the application port
EXPOSE 8000

# Health check — Docker will mark container unhealthy if this fails
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
    || exit 1

# Start the server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
