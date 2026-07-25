import 'dart:async';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';

import '../config/app_config.dart';
import '../models/analysis_result.dart';
import '../services/api_service.dart';
import '../services/speech_service.dart';

// ── Session ID ───────────────────────────────────────────────────────────────
//
// A stable, app-lifetime token sent with every frame so the backend can
// maintain tracking continuity across frames within this session.
// Generated once when the class is first loaded; survives hot-restarts
// because it lives at class scope rather than in widget state.
//
// Format: "flutter-<timestamp-ms>" — unique per app launch, no package
// dependency needed.
final String _kSessionId =
    'flutter-${DateTime.now().millisecondsSinceEpoch}';

// ─────────────────────────────────────────────────────────────────────────────

class CameraScreen extends StatefulWidget {
  final List<CameraDescription> cameras;
  const CameraScreen({super.key, required this.cameras});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  late CameraController _controller;
  bool _initialized = false;
  bool _streaming = false;
  bool _requestInFlight = false;

  AnalysisResult? _lastResult;
  String? _error;

  Timer? _streamTimer;
  final SpeechService _speech = SpeechService();

  @override
  void initState() {
    super.initState();
    _speech.init();
    _initCamera();
  }

  Future<void> _initCamera() async {
    _controller = CameraController(
      widget.cameras.first,
      ResolutionPreset.medium,
      enableAudio: false,
    );
    try {
      await _controller.initialize();
      if (mounted) setState(() => _initialized = true);
    } catch (e) {
      if (mounted) setState(() => _error = 'Camera error: $e');
    }
  }

  @override
  void dispose() {
    _streamTimer?.cancel();
    _controller.dispose();
    _speech.dispose();
    super.dispose();
  }

  // ── Streaming via Timer + takePicture ────────────────────────────────────

  void _startStream() {
    setState(() {
      _streaming = true;
      _error = null;
    });

    _streamTimer = Timer.periodic(
      Duration(milliseconds: AppConfig.streamIntervalMs),
      (_) => _captureAndAnalyze(),
    );
  }

  void _stopStream() {
    _streamTimer?.cancel();
    _streamTimer = null;
    if (mounted) setState(() => _streaming = false);
  }

  void _toggleStream() {
    if (_streaming) {
      _stopStream();
    } else {
      _startStream();
    }
  }

  Future<void> _captureAndAnalyze() async {
    if (_requestInFlight || !_controller.value.isInitialized) return;
    _requestInFlight = true;

    try {
      final file = await _controller.takePicture();
      final result = await ApiService.analyzeFile(
        File(file.path),
        // Pass the stable session token so the backend links this frame
        // to the ongoing tracking session.
        sessionId: AppConfig.enableTracking ? _kSessionId : null,
      );
      if (mounted) setState(() { _lastResult = result; _error = null; });
      _announce(result);
    } catch (e) {
      if (mounted) {
        setState(() => _error = e.toString().replaceFirst('Exception: ', ''));
      }
    } finally {
      _requestInFlight = false;
    }
  }

  // ── Speech / haptics ─────────────────────────────────────────────────────

  /// Speaks the most important detection aloud, prioritising objects that are:
  ///   1. In the centre of the walking path
  ///   2. Closest to the camera (by metric depth when available)
  ///   3. Hazards
  ///
  /// The backend already sorts objects by priority and injects distance +
  /// motion riders into [result.summary].  This method speaks that summary
  /// directly and optionally appends a distance sentence for the primary
  /// approaching object when metric depth is available and not already spoken.
  void _announce(AnalysisResult result) {
    if (result.hazardDetected) {
      HapticFeedback.heavyImpact();
    }

    String speechText = result.summary;

    // The backend summary already contains distance ("3.2 meters") and motion
    // riders ("coming!") when depth + tracking are active.
    // We only augment here when metric depth is available but the summary
    // doesn't already contain a meters figure — avoids double-speaking.
    if (AppConfig.enableTracking && result.trackingEnabled) {
      final primary = _primaryObject(result);
      if (primary != null) {
        final motion = primary.motion;
        final hasMetres = speechText.contains('metre') || speechText.contains('meter');
        if (!hasMetres && motion != null && motion.distance.isMetric) {
          final dist = motion.distance.value!.toStringAsFixed(1);
          speechText = '$speechText $dist meters.';
        }
      }
    }

    _speech.speak(speechText, isHazard: result.hazardDetected);
  }

  /// Returns the highest-priority detected object:
  ///   - Prefer hazards in the centre path with metric depth
  ///   - Fall back to the first object in the (already-priority-sorted) list
  DetectedObject? _primaryObject(AnalysisResult result) {
    if (result.objects.isEmpty) return null;

    // 1. Closest hazard in the centre path with metric depth
    final centreHazards = result.objects.where((o) =>
        (o.isHazard ?? false) &&
        (o.direction == 'center' ||
         o.direction == 'center-left' ||
         o.direction == 'center-right') &&
        o.motion?.distance.isMetric == true);
    if (centreHazards.isNotEmpty) return centreHazards.first;

    // 2. Any hazard with depth
    final hazardsWithDepth = result.objects.where((o) =>
        (o.isHazard ?? false) && o.motion?.distance.isMetric == true);
    if (hazardsWithDepth.isNotEmpty) return hazardsWithDepth.first;

    // 3. Any hazard
    final hazards = result.objects.where((o) => o.isHazard ?? false);
    if (hazards.isNotEmpty) return hazards.first;

    // 4. First object (backend already sorted by priority)
    return result.objects.first;
  }

  // ── Gallery picker ───────────────────────────────────────────────────────

  Future<void> _pickFromGallery() async {
    if (_streaming) _stopStream();

    final picker = ImagePicker();
    final picked = await picker.pickImage(source: ImageSource.gallery);
    if (picked == null) return;

    setState(() { _error = null; _requestInFlight = true; });
    try {
      final result = await ApiService.analyzeFile(
        File(picked.path),
        sessionId: null, // gallery picks are single-shot; no tracking session
      );
      if (mounted) setState(() => _lastResult = result);
      _announce(result);
    } catch (e) {
      if (mounted) {
        setState(() => _error = e.toString().replaceFirst('Exception: ', ''));
      }
    } finally {
      if (mounted) setState(() => _requestInFlight = false);
    }
  }

  // ── Build ────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: SafeArea(
        child: Stack(
          children: [
            // Camera preview
            if (_initialized)
              Positioned.fill(child: CameraPreview(_controller))
            else if (_error != null)
              Center(
                child: Text(
                  _error!,
                  style: const TextStyle(color: Colors.red),
                  textAlign: TextAlign.center,
                ),
              )
            else
              const Center(
                child: CircularProgressIndicator(color: Colors.white),
              ),

            // Results overlay
            if (_lastResult != null)
              Positioned(
                left: 0,
                right: 0,
                bottom: 100,
                child: _ResultsOverlay(result: _lastResult!),
              ),

            // Error toast
            if (_error != null && _initialized)
              Positioned(
                left: 16,
                right: 16,
                top: 16,
                child: Container(
                  padding: const EdgeInsets.all(10),
                  decoration: BoxDecoration(
                    color: Colors.red.withOpacity(0.8),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Text(
                    _error!,
                    style: const TextStyle(color: Colors.white, fontSize: 13),
                  ),
                ),
              ),

            // In-flight indicator
            if (_requestInFlight)
              const Positioned(
                top: 16,
                right: 16,
                child: SizedBox(
                  width: 18,
                  height: 18,
                  child: CircularProgressIndicator(
                    color: Colors.white,
                    strokeWidth: 2,
                  ),
                ),
              ),

            // Tracking active indicator (top-left dot)
            if (_lastResult != null && _lastResult!.trackingEnabled)
              const Positioned(
                top: 16,
                left: 16,
                child: _TrackingBadge(),
              ),

            // Bottom controls
            Positioned(
              left: 0,
              right: 0,
              bottom: 0,
              child: _BottomBar(
                streaming: _streaming,
                onToggleStream: _initialized ? _toggleStream : null,
                onGallery: _initialized ? _pickFromGallery : null,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// ── Tracking active badge ────────────────────────────────────────────────────

class _TrackingBadge extends StatelessWidget {
  const _TrackingBadge();

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: Colors.green.withOpacity(0.75),
        borderRadius: BorderRadius.circular(12),
      ),
      child: const Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(Icons.track_changes, color: Colors.white, size: 12),
          SizedBox(width: 4),
          Text(
            'Tracking',
            style: TextStyle(
              color: Colors.white,
              fontSize: 11,
              fontWeight: FontWeight.w600,
            ),
          ),
        ],
      ),
    );
  }
}

// ── Results overlay ──────────────────────────────────────────────────────────

class _ResultsOverlay extends StatelessWidget {
  final AnalysisResult result;
  const _ResultsOverlay({required this.result});

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 12),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: result.hazardDetected
            ? Colors.red.withOpacity(0.55)
            : Colors.black.withOpacity(0.65),
        borderRadius: BorderRadius.circular(12),
        border: result.hazardDetected
            ? Border.all(color: Colors.redAccent, width: 2)
            : null,
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          // Hazard banner
          if (result.hazardDetected)
            const Padding(
              padding: EdgeInsets.only(bottom: 4),
              child: Text(
                '⚠ HAZARD',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 13,
                  fontWeight: FontWeight.bold,
                  letterSpacing: 1,
                ),
              ),
            ),

          // Scene summary
          Text(
            result.summary,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 16,
              fontWeight: FontWeight.w600,
            ),
          ),

          // Object chips (label + confidence + motion badge)
          if (result.objects.isNotEmpty) ...[
            const SizedBox(height: 8),
            Wrap(
              spacing: 6,
              runSpacing: 4,
              children: result.objects.map((obj) => _ObjectChip(obj: obj)).toList(),
            ),
          ],

          // Footer: timing + object count + tracking status
          const SizedBox(height: 6),
          Row(
            children: [
              Text(
                '${result.processingTimeMs.toStringAsFixed(0)} ms'
                ' · ${result.objects.length} object'
                '${result.objects.length == 1 ? '' : 's'}',
                style: const TextStyle(color: Colors.white38, fontSize: 11),
              ),
              if (result.trackingEnabled) ...[
                const SizedBox(width: 6),
                const Text(
                  '· tracking on',
                  style: TextStyle(color: Colors.greenAccent, fontSize: 11),
                ),
              ],
            ],
          ),
        ],
      ),
    );
  }
}

// ── Object chip ──────────────────────────────────────────────────────────────

class _ObjectChip extends StatelessWidget {
  final DetectedObject obj;
  const _ObjectChip({required this.obj});

  @override
  Widget build(BuildContext context) {
    final baseColor = _confidenceColor(obj.confidence);
    final motion = obj.motion;

    // Choose chip accent colour based on motion state
    final motionColor = _motionColor(motion);

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color: baseColor.withOpacity(0.15),
        border: Border.all(color: motionColor ?? baseColor),
        borderRadius: BorderRadius.circular(20),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            '${obj.label} ${(obj.confidence * 100).toStringAsFixed(0)}%',
            style: TextStyle(
              color: baseColor,
              fontSize: 12,
              fontWeight: FontWeight.w500,
            ),
          ),

          // Motion state badge — only shown when reliable
          if (motion != null && motion.isReliable && !motion.isUnknown) ...[
            const SizedBox(width: 4),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 5, vertical: 2),
              decoration: BoxDecoration(
                color: (motionColor ?? baseColor).withOpacity(0.25),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Text(
                motion.stateLabel,
                style: TextStyle(
                  color: motionColor ?? baseColor,
                  fontSize: 10,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),

            // Metric speed label (only when depth is available)
            if (motion.speedLabel != null) ...[
              const SizedBox(width: 3),
              Text(
                motion.speedLabel!,
                style: const TextStyle(
                  color: Colors.white54,
                  fontSize: 10,
                ),
              ),
            ],
          ],

          // Track ID (small, subtle)
          if (obj.trackId != null) ...[
            const SizedBox(width: 4),
            Text(
              '#${obj.trackId}',
              style: const TextStyle(color: Colors.white24, fontSize: 9),
            ),
          ],
        ],
      ),
    );
  }

  Color _confidenceColor(double c) {
    if (c >= 0.8) return Colors.greenAccent;
    if (c >= 0.6) return Colors.orangeAccent;
    return Colors.redAccent;
  }

  /// Returns an accent colour that signals urgency based on motion state.
  Color? _motionColor(MotionInfo? motion) {
    if (motion == null || !motion.isReliable) return null;
    switch (motion.state) {
      case 'APPROACHING':  return Colors.orangeAccent;
      case 'MOVING_AWAY':  return Colors.lightBlueAccent;
      case 'STATIONARY':   return Colors.white54;
      default:             return null;
    }
  }
}

// ── Bottom bar ───────────────────────────────────────────────────────────────

class _BottomBar extends StatelessWidget {
  final bool streaming;
  final VoidCallback? onToggleStream;
  final VoidCallback? onGallery;

  const _BottomBar({
    required this.streaming,
    required this.onToggleStream,
    required this.onGallery,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      color: Colors.black87,
      padding: const EdgeInsets.symmetric(vertical: 20, horizontal: 32),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
        children: [
          IconButton(
            onPressed: onGallery,
            icon: const Icon(
              Icons.photo_library_outlined,
              color: Colors.white,
              size: 28,
            ),
          ),
          GestureDetector(
            onTap: onToggleStream,
            child: AnimatedContainer(
              duration: const Duration(milliseconds: 200),
              width: 68,
              height: 68,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: streaming ? Colors.red : Colors.white,
                border: Border.all(
                  color: streaming ? Colors.red : Colors.white,
                  width: 3,
                ),
              ),
              child: Icon(
                streaming ? Icons.stop : Icons.play_arrow,
                color: streaming ? Colors.white : Colors.black,
                size: 32,
              ),
            ),
          ),
          const SizedBox(width: 48),
        ],
      ),
    );
  }
}
