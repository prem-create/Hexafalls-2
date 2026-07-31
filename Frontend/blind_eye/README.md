# 📱 Meera The Walk Assistant — Flutter Mobile Client

> **Accessible real-time visual perception, speech interface, and auditory spatial guidance for blind and visually impaired users.**

---

## 📌 Overview

The **Meera Frontend** (`blind_eye`) is a high-performance Flutter mobile application designed for visual perception assistance. The app continuously captures camera frames, sends them to the Python FastAPI backend, and translates structured spatial data into immediate, natural-language voice feedback (TTS) and haptic/auditory warnings.

---

## 🌟 Key Features

* **📷 Live Camera Frame Streamer**: Captures continuous camera frames and converts YUV420 camera buffers into optimized JPEG payloads using background isolates to ensure zero UI thread stutter.
* **⏱️ Adaptive Latency-Based Streaming**: Automatically adjusts frame transmission intervals (from 100ms up to 1500ms) using an Exponential Moving Average (EMA) of network round-trip time.
* **🗣️ Text-To-Speech (TTS) Navigation**: Speaks natural scene descriptions and immediate hazard alerts in real time (e.g. *"Chair 1.5 meters ahead on your left"*).
* **🎙️ Speech-To-Text (STT) Voice Assistant**: Integrated voice command listener allowing users to ask questions hands-free:
  * *"What is in front of me?"*
  * *"Is my path clear?"*
  * *"Repeat last alert"*
  * *"Find my phone"* (Triggers loud emergency ringtone)
* **🆔 Session Tracking Continuity**: Generates a stable session ID (`flutter-<timestamp>`) per app launch, enabling backend temporal multi-object tracking across continuous streams.
* **🎨 Accessible & Tactile UI**: High-contrast dark design with large touch targets, live status indicators, bounding box visual overlays (for sighted guides/testing), and clear auditory status feedback.

---

## 📁 Directory Architecture

```
lib/
├── config/
│   └── app_config.dart          # Server Base URL, endpoints, timeouts, stream interval
├── models/
│   ├── analysis_result.dart     # Bounding box, direction, proximity, depth & trajectory parser
│   └── voice_command.dart       # Voice command intent definitions
├── screens/
│   └── camera_screen.dart       # Main UI camera preview, adaptive loop, overlays & controls
├── services/
│   ├── api_service.dart         # HTTP Multipart POST handler for backend /analyze requests
│   ├── speech_service.dart      # Flutter TTS wrapper with alert prioritization & speech queuing
│   └── voice_command_service.dart# Speech-to-text listener & command pattern matcher
└── utils/
    └── image_converter.dart     # High-speed YUV420 to JPEG conversion routines
```

---

## ⚙️ Configuration & Environment

All connection settings reside in [`lib/config/app_config.dart`](./lib/config/app_config.dart).

```dart
class AppConfig {
  /// Base URL pointing to the FastAPI backend.
  /// Replace with your host machine's Wi-Fi IP address during local development.
  static const String baseUrl = 'http://192.168.1.45:8000';

  static const String analyzeEndpoint = '$baseUrl/analyze';
  static const String healthEndpoint = '$baseUrl/health';

  /// Timeout for HTTP requests in seconds
  static const int requestTimeoutSeconds = 30;

  /// Default interval between camera frame transmissions (milliseconds)
  static const int streamIntervalMs = 1500;

  /// Toggle temporal multi-object tracking session header
  static const bool enableTracking = true;
}
```

---

## 🚀 Getting Started

### 1. Prerequisites
* **Flutter SDK**: `^3.0.0`
* **Dart SDK**: `^3.0.0`
* **Android Studio** or **Xcode**
* Physical Android or iOS device with working camera and microphone.

---

### 2. Physical Device Setup (Local Development)

1. Connect your mobile phone and development machine to the **same Wi-Fi network**.
2. Find your computer's local IP address:
   * **Windows**: Run `ipconfig` in CMD/PowerShell $\rightarrow$ look for `IPv4 Address` under your Wi-Fi adapter.
   * **macOS/Linux**: Run `ifconfig` or `ip a`.
3. Open `lib/config/app_config.dart` and update `baseUrl` to `http://<YOUR_LOCAL_IP>:8000`.

---

### 3. Build & Run

Install dependencies:
```bash
flutter pub get
```

Run on connected physical device:
```bash
flutter run
```

---

## 📱 Mobile App Permissions

The app requires runtime permissions declared in platform manifests:

* **Camera**: Required for live obstacle detection & perception (`android.permission.CAMERA`, `NSCameraUsageDescription`).
* **Microphone**: Required for STT voice command recognition (`android.permission.RECORD_AUDIO`, `NSSpeechRecognitionUsageDescription`).
* **Internet**: Required to send image streams to local or cloud backend (`android.permission.INTERNET`).

---

## 🛠️ Key Component Breakdown

### `camera_screen.dart`
The central hub of the mobile client. Manages:
* Initialization of the device camera controller.
* The adaptive camera stream loop (transmitting frames only when the previous HTTP request has resolved to avoid backlog).
* Bounding box canvas painter overlaid on the camera feed.
* Voice assistant state (listening vs. speaking).

### `speech_service.dart`
Encapsulates `flutter_tts`. Handles:
* Immediate speech interrupts for high-priority hazard alerts.
* Queued speaking for general scene descriptions.
* Pitch, volume, and speech rate configuration.

### `voice_command_service.dart`
Wraps `speech_to_text` to capture user voice commands. Recognizes key phrases and executes callbacks for detection toggles, re-querying scene status, or triggering the phone locator ringtone player.

---

## ❓ Troubleshooting

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| **"Cannot connect to server"** | App using `localhost` or wrong IP | Set `baseUrl` in `app_config.dart` to your computer's Wi-Fi IP address (`192.168.x.x`), not `localhost`. |
| **Camera feed black/frozen** | Missing camera permissions | Ensure camera permissions are granted in system settings. |
| **High latency / slow TTS** | High backend processing time | Lower camera resolution in `camera_screen.dart` or check backend hardware acceleration / YOLO model size (`yolov8n`). |
| **Speech recognition fails** | Microphone unavailable | Ensure microphone permissions are allowed and device has working STT service. |
