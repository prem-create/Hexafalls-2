# Walking Eye — AI Perception Engine

Production-quality Python backend for real-time object detection and scene understanding.
Built to serve as the perception layer for an AI assistant ("Walking Eye").

---

## Architecture

```
backend/
├── app/
│   ├── api/
│   │   └── routes/
│   │       ├── health.py        # GET / and GET /health
│   │       └── analysis.py      # POST /analyze and POST /analyze/batch
│   ├── config/
│   │   └── settings.py          # Env-driven config via pydantic-settings
│   ├── core/
│   │   └── model_manager.py     # YOLO lifecycle: load once, reuse forever
│   ├── dependencies/
│   │   └── model_dependency.py  # FastAPI DI: injects model + service into routes
│   ├── middleware/
│   │   ├── logging_middleware.py # Request/response logging with request IDs
│   │   └── timing_middleware.py  # X-Process-Time-Ms response header
│   ├── models/                   # Internal domain models (future use)
│   ├── reasoning/
│   │   └── scene_analyzer.py    # Natural-language scene summary generation
│   ├── schemas/
│   │   └── analysis.py          # Pydantic request/response contracts
│   ├── services/
│   │   └── analysis_service.py  # Orchestrates: image → detect → reason → respond
│   ├── utilities/
│   │   └── logger.py            # Centralized logging factory
│   ├── vision/
│   │   ├── image_processor.py   # OpenCV: decode, validate, resize
│   │   └── detector.py          # YOLO inference wrapper
│   └── main.py                  # App factory, lifespan, middleware, routers
├── models/                      # YOLO .pt weight files (auto-downloaded)
├── logs/                        # Rotating log files
├── uploads/                     # Uploaded images (future persistence)
├── tests/                       # Test suite
├── requirements.txt
├── .env.example
├── Dockerfile
└── docker-compose.yml
```

---

## Quick Start

### Option 1 — Local (recommended for development)

**1. Clone and enter the backend directory**
```bash
cd backend
```

**2. Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up environment**
```bash
cp .env.example .env
# Edit .env if needed — defaults work out of the box
```

**5. Create required directories**
```bash
mkdir -p models logs uploads
```

**6. Run the server**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The server starts at `http://localhost:8000`.
YOLO model downloads automatically on first run (~6 MB for yolov8n).

---

### Option 2 — Docker

**1. Build and start**
```bash
cp .env.example .env
docker compose up --build
```

**2. Run in background**
```bash
docker compose up -d
```

**3. View logs**
```bash
docker compose logs -f
```

**4. Stop**
```bash
docker compose down
```

---

## API Reference

### GET /
Server identification ping.

```json
{
  "name": "Walking Eye - Perception Engine",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "/health"
}
```

---

### GET /health
Detailed health check. Poll this on Flutter app startup.

```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00+00:00",
  "uptime_seconds": 42.1,
  "model": { "loaded": true, "path": "models/yolov8n.pt", "type": "YOLOv8" },
  "config": { "confidence_threshold": 0.45, "max_image_dimension": 1280 }
}
```

---

### POST /analyze
Analyze a single image.

**Request**
```
Content-Type: multipart/form-data
Field: image (file) — JPEG or PNG
```

**Response**
```json
{
  "success": true,
  "processing_time_ms": 48.5,
  "model_used": "YOLOv8",
  "image_width": 1280,
  "image_height": 720,
  "object_count": 1,
  "summary": "There is a chair on your left.",
  "hazard_detected": false,
  "objects": [
    {
      "id": 1,
      "label": "chair",
      "confidence": 0.94,
      "bbox": { "x": 120, "y": 240, "width": 220, "height": 330 },
      "center": { "x": 230, "y": 405 },
      "direction": "left",
      "proximity": "medium",
      "is_hazard": false
    }
  ]
}
```

---

## Spatial Awareness (direction + proximity)

Every detection is annotated with:

- **`direction`** — `left` / `center-left` / `center` / `center-right` / `right`, based on where the object's center falls across the frame.
- **`proximity`** — `far` / `medium` / `close` / `very close`, estimated from how much of the frame the object's bounding box fills (a cheap, model-free stand-in for real depth estimation).
- **`is_hazard`** — `true` if the object is close/very-close *and* roughly in the walking path (not off to the far side).

The top-level `summary` puts any hazard warning first — e.g. `"Stop, person ahead, very close! There is a chair on your right."` — and `hazard_detected` lets a client trigger a distinct alert sound/vibration without parsing the sentence.

This logic lives in `app/reasoning/spatial_analyzer.py` and is intentionally pure geometry (no extra model, no added latency) so it's cheap to demo live.

**Error responses**

| Status | Reason |
|--------|--------|
| 400 | Missing image, unsupported format, file too large, corrupted image |
| 503 | Model not loaded (service starting up) |
| 500 | Internal server error |

---

### POST /analyze/batch
Future endpoint. Returns 501 until implemented.

---

### GET /docs
Interactive Swagger UI — auto-generated by FastAPI.

---

## Configuration

All settings are controlled via environment variables or `.env` file.

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |
| `DEBUG` | `false` | Enable debug mode + auto-reload |
| `MODEL_PATH` | `models/yolov8n.pt` | Path to YOLO weights |
| `CONFIDENCE_THRESHOLD` | `0.45` | Minimum detection confidence |
| `MAX_IMAGE_SIZE_MB` | `10.0` | Max upload size in MB |
| `MAX_IMAGE_DIMENSION` | `1280` | Longest edge resize limit |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `ALLOWED_ORIGINS` | `["*"]` | CORS allowed origins |

---

## YOLO Model Options

Swap models by changing `MODEL_PATH` in `.env`:

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| `yolov8n.pt` | 6 MB | Fastest | Good |
| `yolov8s.pt` | 22 MB | Fast | Better |
| `yolov8m.pt` | 50 MB | Medium | Great |
| `yolov8l.pt` | 87 MB | Slower | Excellent |
| `yolov8x.pt` | 137 MB | Slowest | Best |

For mobile-facing APIs, `yolov8n.pt` or `yolov8s.pt` are recommended.

---

## Extending the Backend

The architecture is designed for zero-friction feature additions:

- **OCR** — add `vision/ocr_processor.py`, set `ENABLE_OCR=true`
- **Depth estimation** — add `vision/depth_estimator.py`, set `ENABLE_DEPTH=true`
- **Object tracking** — add `vision/tracker.py`, set `ENABLE_TRACKING=true`
- **LLM scene summaries** — replace `reasoning/scene_analyzer.py`'s `_rule_based_summary` with an API call
- **Navigation hints** — add `reasoning/spatial_analyzer.py`, inject into `AnalysisService`

No route changes, no schema breakage — just new modules.

---

## Tech Stack

- Python 3.11+
- FastAPI 0.111
- Ultralytics YOLOv8
- OpenCV (headless)
- Pydantic v2
- Uvicorn
