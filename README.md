# 👁️ Walking Eye — AI Perception & Spatial Assistance Engine

> **Empowering visually impaired individuals with real-time AI perception, metric depth estimation, temporal motion tracking, and natural-language voice feedback.**

---

## 📌 Project Overview

**Walking Eye** is an end-to-end assistive technology system designed to act as an artificial pair of eyes for blind and visually impaired users. By combining real-time smartphone camera processing with a high-performance Python AI engine, Walking Eye provides immediate auditory awareness of the user's surroundings, pinpoints obstacles, measures distances, detects approaching hazards, and responds to natural voice commands.

The project consists of two primary components:
1. **[Frontend (Mobile App)](./Frontend/blind_eye)**: A Flutter-based mobile client featuring adaptive high-frequency frame streaming, Text-To-Speech (TTS) spoken navigation, Speech-To-Text (STT) voice command recognition, tactile UI controls, and emergency device finder audio alerts.
2. **[Backend (Perception Engine)](./backend)**: A FastAPI-powered microservice integrating YOLOv8 object detection, MiDaS monocular metric depth estimation, session-aware temporal object tracking, spatial direction analysis, and automated hazard prioritization.

---

## 🌟 Key Features

* **⚡ Adaptive Real-Time Frame Streaming**: Dynamically scales request rate based on latency exponential moving averages (EMA) to maintain real-time performance without overloading network or client battery.
* **🎯 YOLOv8 Object Detection**: High-speed, multi-class object detection identifying pedestrians, vehicles, furniture, doors, stairs, and common daily objects.
* **📏 Monocular Metric Depth Estimation (MiDaS DPT_Hybrid)**: Computes estimated physical distances in meters ($m$) to detected objects directly from 2D RGB frames without requiring LiDAR sensors.
* **🧭 2D & 3D Spatial Awareness**: Categorizes object locations into 5 horizontal direction zones (`left`, `center-left`, `center`, `center-right`, `right`) combined with proximity tiers (`very close`, `close`, `medium`, `far`).
* **🔄 Temporal Multi-Object Tracking & Motion Dynamics**: Tracks objects across continuous frames per user session (`session_id`), calculating movement trajectories, delta distance, closing speed, and vector directions (`approaching`, `receding`, `stationary`).
* **⚠️ Intelligent Hazard Alert System**: Evaluates danger levels based on object trajectory and proximity, immediately prioritizing urgent collision risks (e.g. "Stop! Person approaching rapidly 1.2m ahead on your left").
* **🎙️ Voice Command & Interactive Querying**: Hands-free voice interface allowing users to ask questions like *"What is in front of me?"*, *"Is it safe to walk?"*, or trigger emergency features like *"Find my phone"*.
* **🔊 Audio & Tactile Accessibility**: Optimized for low-vision/blind users with full Text-to-Speech integration, distinct haptic/sound warning cues, and high-contrast accessible layouts.

---

## 🏗️ System Architecture

```
                                  WALKING EYE SYSTEM ARCHITECTURE
                                  

 ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
 │                                   FLUTTER MOBILE CLIENT                                     │
 │                                                                                             │
 │  ┌──────────────────┐   YUV420 → JPEG   ┌───────────────────┐    HTTP Multipart POST       │
 │  │ Camera Feed      ├──────────────────►│ Adaptive Streamer ├──────────────────────────┐   │
 │  │ (Live Video)     │                   │ (Latency EMA)     │                          │   │
 │  └──────────────────┘                   └───────────────────┘                          │   │
 │                                                                                        ▼   │
 │  ┌──────────────────┐   Voice Command   ┌───────────────────┐                          │   │
 │  │ Speech-To-Text   ├──────────────────►│ Voice Command     │                          │   │
 │  │ (STT Service)    │                   │ Handler           │                          │   │
 │  └──────────────────┘                   └─────────┬─────────┘                          │   │
 │                                                   │                                    │   │
 │  ┌──────────────────┐   Spoken Feedback ┌─────────▼─────────┐                          │   │
 │  │ Text-To-Speech   │◄──────────────────┤ Response Manager  │◄─────────────────────┐   │   │
 │  │ (TTS Engine)     │                   │ & Audio Player    │                      │   │   │
 │  └──────────────────┘                   └───────────────────┘                      │   │   │
 └────────────────────────────────────────────────────────────────────────────────────┼───┘   │
                                                                                      │       │
                                                                       JSON Response  │       │
                                                                                      │       │
 ┌────────────────────────────────────────────────────────────────────────────────────┼───────┐
 │                                   FASTAPI BACKEND                                  │       │
 │                                                                                    │       │
 │  ┌──────────────────┐                   ┌───────────────────┐                      │       │
 │  │ API Router       │◄──────────────────┼ Multipart Decoder │◄─────────────────────┘       │
 │  │ (/analyze)       │                   │ & OpenCV Resize   │                              │
 │  └────────┬─────────┘                   └───────────────────┘                              │
 │           │                                                                                │
 │           ▼                                                                                │
 │  ┌──────────────────────────────────────────────────────────────────────────────────────┐  │
 │  │                                  AI PERCEPTION PIPELINE                              │  │
 │  │                                                                                      │  │
 │  │   ┌───────────────────────────┐                ┌─────────────────────────────────┐   │  │
 │  │   │ YOLOv8 Object Detection   │                │ MiDaS Metric Depth Estimation   │   │  │
 │  │   │ (Bounding Boxes & Labels) │                │ (Monocular Depth Map in Metres) │   │  │
 │  │   └─────────────┬─────────────┘                └────────────────┬────────────────┘   │  │
 │  │                 │                                               │                    │  │
 │  │                 └───────────────────────┬───────────────────────┘                    │  │
 │  │                                         ▼                                            │  │
 │  │                           ┌───────────────────────────┐                              │  │
 │  │                           │ Spatial 2D/3D Analyzer    │                              │  │
 │  │                           │ (Direction & Proximity)   │                              │  │
 │  │                           └─────────────┬─────────────┘                              │  │
 │  │                                         ▼                                            │  │
 │  │                           ┌───────────────────────────┐                              │  │
 │  │                           │ Temporal Tracker Store    │                              │  │
 │  │                           │ (Motion & Closing Speed)  │                              │  │
 │  │                           └─────────────┬─────────────┘                              │  │
 │  │                                         ▼                                            │  │
 │  │                           ┌───────────────────────────┐                              │  │
 │  │                           │ Natural Language          │                              │  │
 │  │                           │ Scene & Hazard Summarizer ├──────────────────────────────┘  │
 │  │                           └───────────────────────────┘                                 │
 │  └──────────────────────────────────────────────────────────────────────────────────────┘  │
 └────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
Hexafalls-2/
├── README.md                          # Master project documentation (you are here)
│
├── Frontend/                          # Flutter Mobile Client Application
│   └── blind_eye/
│       ├── lib/
│       │   ├── config/                # Endpoint URLs, timeouts & stream intervals
│       │   ├── models/                # Data models (AnalysisResult, SpatialObject, etc.)
│       │   ├── screens/               # Main CameraScreen, UI overlay & accessibility controls
│       │   ├── services/              # API, Text-to-Speech (TTS) & Voice Command (STT) services
│       │   └── utils/                 # Image conversion (YUV to JPEG isolates)
│       ├── pubspec.yaml               # Flutter package dependencies
│       └── README.md                  # Frontend-specific setup and architecture docs
│
└── backend/                           # FastAPI AI Perception Engine
    ├── app/
    │   ├── api/routes/                # API Endpoints (/analyze, /health)
    │   ├── config/                    # Pydantic environment configuration
    │   ├── core/                      # Model lifecycle & state manager
    │   ├── middleware/                # Logging, request ID & processing timer middleware
    │   ├── reasoning/                 # Spatial 2D/3D analyzer & scene summarizer
    │   ├── schemas/                   # Pydantic request & response models
    │   ├── services/                  # Perception pipeline orchestrator
    │   ├── tracking/                  # Session-aware IoU multi-object tracker & alert manager
    │   └── vision/                    # OpenCV image processor, YOLO detector & MiDaS depth engine
    ├── models/                        # Pre-trained YOLOv8 / MiDaS weight store
    ├── requirements.txt               # Python package dependencies
    ├── Dockerfile                     # Containerization spec
    ├── docker-compose.yml             # Single-command stack launch
    └── README.md                  # Backend-specific installation and API reference
```

---

## 🚀 Quick Start Guide

### Prerequisites
* **Backend**: Python 3.11+ (or Docker / Docker Compose)
* **Frontend**: Flutter SDK (>=3.0.0), Android Studio / Xcode, or connected physical Android/iOS device
* **Network**: Mobile device and host machine connected to the same Wi-Fi network (for local dev)

---

### Step 1: Start the Backend

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Set up virtual environment and install dependencies:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate

   pip install -r requirements.txt
   ```

3. Copy the environment configuration:
   ```bash
   cp .env.example .env
   ```

4. Run the development server:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```
   *The server will start at `http://0.0.0.0:8000`. YOLOv8 weights auto-download on first startup.*

*Alternatively, run via Docker:*
```bash
docker compose up --build
```

---

### Step 2: Configure & Run the Flutter App

1. Find your host computer's local Wi-Fi IP address:
   * **Windows**: Run `ipconfig` (look for `IPv4 Address` under your active Wi-Fi adapter, e.g. `192.168.1.45`)
   * **macOS/Linux**: Run `ifconfig` or `ip a`

2. Open [`Frontend/blind_eye/lib/config/app_config.dart`](file:///c:/Users/bhavesh/Hexafalls-2/Frontend/blind_eye/lib/config/app_config.dart) and update `baseUrl`:
   ```dart
   class AppConfig {
     static const String baseUrl = 'http://192.168.1.45:8000'; // Replace with your host IP
     ...
   }
   ```

3. Navigate to the frontend directory:
   ```bash
   cd Frontend/blind_eye
   ```

4. Install dependencies and run on connected device:
   ```bash
   flutter pub get
   flutter run
   ```

---

## 🔌 API Endpoint Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Root health ping & API specs link |
| `GET` | `/health` | Health check reporting model load state, tracking state, and uptime |
| `POST` | `/analyze` | Primary endpoint. Accepts multipart image frame + `session_id`. Returns objects, depths, trajectories, and spoken natural language summary |
| `POST` | `/analyze/batch` | Batch frame processing (reserved for video file analysis) |
| `GET` | `/docs` | Interactive Swagger UI documentation |

---

## 🛠️ Technology Stack

| Domain | Technology / Framework | Usage |
| :--- | :--- | :--- |
| **Mobile App (Frontend)** | Flutter 3.x / Dart | Cross-platform mobile client |
| **Speech & Audio** | `flutter_tts`, `speech_to_text` | Hands-free voice commands & auditory navigation |
| **Camera & Vision Client** | `camera`, `image` package | Frame capture & YUV->JPEG isolate conversion |
| **Backend Framework** | FastAPI, Pydantic v2, Uvicorn | High-throughput async web framework |
| **Computer Vision Engine** | Ultralytics YOLOv8, OpenCV | Real-time object classification & localization |
| **Depth Estimation** | PyTorch / MiDaS DPT_Hybrid | Monocular metric depth estimation |
| **Tracking Engine** | Custom IoU & Distance Tracker | Session-aware motion analysis & vector calculation |
| **Deployment & Ops** | Docker, Docker Compose | Containerized backend deployment |

---

## 📄 License & Acknowledgments

This project was created for accessibility assistance and research. Free to use, adapt, and build upon.

* **Ultralytics**: YOLOv8 models
* **Intel ISL**: MiDaS Monocular Depth Estimation
* **Flutter & FastAPI communities**
