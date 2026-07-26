# 🧠 Walking Eye — AI Perception & Motion Engine

> **Production-grade Python backend providing real-time YOLOv8 object detection, MiDaS monocular metric depth estimation, spatial positioning, temporal motion tracking, and hazard assessment.**

---

## 📌 Overview

The **Walking Eye Backend** is a high-performance FastAPI microservice acting as the intelligence layer for visual assistance applications. It ingests video/camera frame images from client devices, runs deep learning vision pipelines, tracks objects across frames, calculates spatial vectors and closing speeds, and returns structured spatial data along with natural-language audio summaries.

---

## 🏗️ Architecture & Component Overview

```
backend/
├── app/
│   ├── api/
│   │   └── routes/
│   │       ├── health.py             # GET / and GET /health
│   │       └── analysis.py           # POST /analyze and POST /analyze/batch
│   ├── config/
│   │   └── settings.py               # Pydantic environment configuration (.env)
│   ├── core/
│   │   └── model_manager.py          # YOLO & MiDaS model lifecycle manager (load once, reuse)
│   ├── dependencies/
│   │   └── model_dependency.py       # FastAPI dependency injection for routers
│   ├── middleware/
│   │   ├── logging_middleware.py     # Unique request ID assignment & log formatting
│   │   └── timing_middleware.py      # X-Process-Time-Ms performance header
│   ├── reasoning/
│   │   ├── scene_analyzer.py         # Natural-language spoken text generation
│   │   └── spatial_analyzer.py       # 2D/3D spatial direction & proximity bucketizer
│   ├── schemas/
│   │   └── analysis.py               # Pydantic request & response serialization models
│   ├── services/
│   │   └── analysis_service.py       # Main pipeline orchestrator (Detect -> Depth -> Track -> Reason)
│   ├── tracking/
│   │   ├── tracker_store.py          # Session-level temporal object tracker store
│   │   └── alert_manager.py          # Hazard alert prioritization & deduplication manager
│   ├── utilities/
│   │   └── logger.py                 # Centralized logging factory
│   ├── vision/
│   │   ├── image_processor.py        # OpenCV image decode, validate & resize utilities
│   │   ├── detector.py               # YOLOv8 object detection wrapper
│   │   └── depth_estimator.py        # MiDaS DPT_Hybrid monocular depth estimation engine
│   └── main.py                       # App factory, lifespan, CORS, middleware & routers
├── models/                           # Model weights cache directory (auto-downloaded)
├── logs/                             # Rotating application log files
├── tests/                            # Pytest test suite
├── requirements.txt                  # Python dependencies
├── .env.example                      # Environment defaults template
├── Dockerfile                        # Multi-stage Docker image definition
└── docker-compose.yml                # Docker compose stack configuration
```

---

## ⚡ Perception & Intelligence Pipeline

1. **Image Preprocessing (`vision/image_processor.py`)**: Validates byte payloads, decodes JPEG/PNG buffers using OpenCV, and resizes frames up to `MAX_IMAGE_DIMENSION` (default: 1280px) while maintaining aspect ratio.
2. **Object Detection (`vision/detector.py`)**: Runs YOLOv8 inference, extracting bounding boxes, confidence scores, and class labels (e.g. `person`, `chair`, `car`, `door`).
3. **Monocular Depth Estimation (`vision/depth_estimator.py`)**: Runs Intel MiDaS (`DPT_Hybrid`) when `ENABLE_DEPTH=true` to generate a pixel-level metric depth map, producing physical distances in meters ($m$). *(Falls back to bounding-box area proxy if disabled/unavailable)*.
4. **Spatial 2D/3D Analysis (`reasoning/spatial_analyzer.py`)**:
   * **Direction**: Categorizes object position into `left`, `center-left`, `center`, `center-right`, or `right`.
   * **Proximity**: Bucketizes distances into `very close` ($<1.5m$), `close` ($1.5-3m$), `medium` ($3-6m$), or `far` ($>6m$).
   * **Priority Ranking**: Scores objects by combining depth and centrality to ensure the most critical items are spoken first.
5. **Temporal Multi-Object Tracking (`tracking/tracker_store.py`)**: When `session_id` is supplied, tracks objects over frame sequences using IoU matching and metric depth histories. Calculates:
   * **Trajectory & Vector**: `approaching`, `receding`, or `stationary`.
   * **Closing Speed**: Measured in meters per second ($m/s$).
6. **Hazard Alert Management (`tracking/alert_manager.py`)**: Detects high-risk trajectory overlaps (e.g., a person approaching rapidly within $1.5m$ of the walking path) and throttles alert frequencies to eliminate TTS audio spam.
7. **Natural Language Summarizer (`reasoning/scene_analyzer.py`)**: Synthesizes structured results into concise spoken feedback (e.g. *"Stop! Person approaching rapidly 1.2m ahead on your left"*).

---

## 🚀 Quick Start Guide

### Option 1: Local Python Environment

1. **Enter directory**:
   ```bash
   cd backend
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize configuration**:
   ```bash
   cp .env.example .env
   ```

5. **Run Uvicorn server**:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```
   *The server starts at `http://0.0.0.0:8000`. Access Swagger API docs at `http://localhost:8000/docs`.*

---

### Option 2: Docker / Docker Compose

Build and launch containerized server:
```bash
cp .env.example .env
docker compose up --build -d
```

Check live logs:
```bash
docker compose logs -f
```

Stop stack:
```bash
docker compose down
```

---

## 🔌 API Endpoint Documentation

### `GET /`
Server identification ping.

**Response**:
```json
{
  "name": "Walking Eye - Perception Engine",
  "version": "1.2.0",
  "docs": "/docs",
  "health": "/health"
}
```

---

### `GET /health`
Comprehensive health check for client initialization.

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2026-07-26T10:00:00+00:00",
  "uptime_seconds": 128.4,
  "model": {
    "loaded": true,
    "path": "models/yolov8n.pt",
    "type": "YOLOv8"
  },
  "depth_estimation": {
    "enabled": true,
    "model_loaded": true,
    "type": "MiDaS DPT_Hybrid"
  },
  "tracking": {
    "enabled": true,
    "active_sessions": 1
  }
}
```

---

### `POST /analyze`
Primary endpoint for real-time frame perception.

**Form Parameters**:
* `image` *(file, required)*: JPEG or PNG binary frame.
* `session_id` *(string, optional)*: Client session token for tracking continuity.

**Sample Response**:
```json
{
  "success": true,
  "processing_time_ms": 42.8,
  "model_used": "YOLOv8",
  "image_width": 1280,
  "image_height": 720,
  "object_count": 2,
  "summary": "Stop! Person approaching rapidly 1.4m ahead on your left. Chair 2.5m on your right.",
  "hazard_detected": true,
  "objects": [
    {
      "id": 1,
      "label": "person",
      "confidence": 0.92,
      "bbox": { "x": 200, "y": 150, "width": 300, "height": 500 },
      "center": { "x": 350, "y": 400 },
      "direction": "left",
      "proximity": "very close",
      "depth_m": 1.4,
      "is_hazard": true,
      "motion": { "state": "approaching", "closing_speed_mps": 0.8 }
    },
    {
      "id": 2,
      "label": "chair",
      "confidence": 0.87,
      "bbox": { "x": 800, "y": 300, "width": 200, "height": 300 },
      "center": { "x": 900, "y": 450 },
      "direction": "right",
      "proximity": "close",
      "depth_m": 2.5,
      "is_hazard": false,
      "motion": { "state": "stationary", "closing_speed_mps": 0.0 }
    }
  ]
}
```

---

## 🛠️ Environment Configuration (`.env`)

| Key | Default | Description |
| :--- | :--- | :--- |
| `HOST` | `0.0.0.0` | Server bind host IP address |
| `PORT` | `8000` | Server HTTP port |
| `DEBUG` | `false` | Development mode with auto-reload |
| `MODEL_PATH` | `models/yolov8n.pt` | Path to YOLO weights (`yolov8n.pt`, `yolov8s.pt`, etc.) |
| `ENABLE_DEPTH` | `true` | Enable/disable MiDaS monocular metric depth estimation |
| `ENABLE_TRACKING` | `true` | Enable/disable temporal multi-object tracking across session streams |
| `CONFIDENCE_THRESHOLD` | `0.45` | Minimum object detection confidence threshold |
| `MAX_IMAGE_DIMENSION` | `1280` | Maximum frame resizing dimension limit |
| `LOG_LEVEL` | `INFO` | Application log verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

---

## 🎯 Model Weight Options

| YOLO Model | Weights Size | Speed | Detection Accuracy |
| :--- | :--- | :--- | :--- |
| `yolov8n.pt` *(default)* | ~6 MB | **Ultra Fast** (~15ms) | Good |
| `yolov8s.pt` | ~22 MB | **Fast** (~30ms) | Higher |
| `yolov8m.pt` | ~50 MB | **Medium** (~70ms) | High |
| `yolov8l.pt` | ~87 MB | **Slower** (~140ms) | Very High |
