import 'dart:async';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';

import '../config/app_config.dart';
import '../models/analysis_result.dart';
import '../models/voice_command.dart';
import '../services/api_service.dart';
import '../services/speech_service.dart';
import '../services/voice_command_service.dart';

// ── Session ID ───────────────────────────────────────────────────────────────
//
// A stable, app-lifetime token sent with every streamed frame so the backend
// can maintain tracking continuity (motion/direction/distance) across frames
// within this session. Generated once when the class is first loaded.
final String _kSessionId =
    'flutter-${DateTime.now().millisecondsSinceEpoch}';

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
  late final VoiceCommandService _voiceCommands;
  bool _voiceListening = false;
  bool _phoneRinging = false;
  bool _startDetectionRequested = false;
  String? _voiceTranscript;

  @override
  void initState() {
    super.initState();
    _speech.init();
    _voiceCommands = VoiceCommandService(
      onCommand: _handleVoiceCommand,
      onListeningStateChanged: (isListening) {
        if (mounted) setState(() => _voiceListening = isListening);
      },
      onError: _showVoiceError,
      onTranscript: (transcript) {
        if (mounted) setState(() => _voiceTranscript = transcript);
      },
    );
    _initCamera();
    unawaited(_voiceCommands.startListening());
  }

  Future<void> _initCamera() async {
    _controller = CameraController(
      widget.cameras.first,
      ResolutionPreset.medium,
      enableAudio: false,
    );
    try {
      await _controller.initialize();
      if (mounted) {
        setState(() => _initialized = true);
        if (_startDetectionRequested) {
          _startDetectionRequested = false;
          _startStream();
        }
      }
    } catch (e) {
      if (mounted) setState(() => _error = 'Camera error: $e');
    }
  }

  @override
  void dispose() {
    _streamTimer?.cancel();
    _controller.dispose();
    _speech.dispose();
    unawaited(_voiceCommands.dispose());
    super.dispose();
  }

  Future<void> _toggleVoiceDetection() async {
    if (_voiceListening) {
      await _voiceCommands.stopListening();
      return;
    }
    await _voiceCommands.startListening();
  }

  Future<void> _handleVoiceCommand(VoiceCommand command) async {
    switch (command) {
      case VoiceCommand.findPhone:
        await _voiceCommands.respondToPhoneFinder();
        if (mounted) setState(() => _phoneRinging = _voiceCommands.isRinging);
        return;
      case VoiceCommand.startDetection:
        if (_initialized && !_streaming) {
          _startStream();
        } else if (!_initialized) {
          _startDetectionRequested = true;
        }
        return;
    }
  }

  Future<void> _stopRingtone() async {
    await _voiceCommands.stopRingtone();
    if (mounted) setState(() => _phoneRinging = false);
  }

  void _showVoiceError(String message) {
    if (mounted) setState(() => _error = message);
  }

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
    if (mounted) {
      setState(() {
        _streaming = false;
      });
    }
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
        // Stable session token lets the backend link this frame to the
        // ongoing tracking session (direction of movement, distance, speed).
        sessionId: AppConfig.enableTracking ? _kSessionId : null,
      );
      if (mounted) {
        setState(() {
          _lastResult = result;
          _error = null;
        });
      }
      _announce(result);
    } catch (e) {
      if (mounted) {
        setState(() => _error = e.toString().replaceFirst('Exception: ', ''));
      }
    } finally {
      _requestInFlight = false;
    }
  }

  void _announce(AnalysisResult result) {
    if (result.hazardDetected) {
      HapticFeedback.heavyImpact();
    }
    _speech.speak(
      _buildNarration(result),
      isHazard: result.hazardDetected,
    );
  }

  String _buildNarration(AnalysisResult result) {
    final parts = <String>[result.summary];

    parts.add(
      'Detected ${result.objectCount} object${result.objectCount == 1 ? '' : 's'}.',
    );

    if (result.suggestedDirection != null) {
      parts.add('Suggested direction is ${result.suggestedDirection}.');
    }

    final highlights = result.objects
        .take(2)
        .map(_describeObjectForSpeech)
        .where((text) => text.isNotEmpty)
        .toList();
    if (highlights.isNotEmpty) {
      parts.add('Key objects: ${highlights.join('; ')}.');
    }

    return parts.join(' ');
  }

  String _describeObjectForSpeech(DetectedObject object) {
    final parts = <String>[object.label];

    if (object.direction != null && object.direction!.isNotEmpty) {
      parts.add(object.direction!.replaceAll('-', ' '));
    }

    if (object.proximity != null && object.proximity!.isNotEmpty) {
      parts.add(object.proximity!);
    }

    if (object.isHazard == true) {
      parts.add('hazard');
    }

    // Motion analysis — only present when tracking recognised this object
    // from a previous frame with enough confidence (isReliable). Adds the
    // direction of movement and, when metric depth is available, distance.
    final motion = object.motion;
    if (motion != null && motion.isReliable && !motion.isUnknown) {
      if (motion.isApproaching) {
        parts.add('approaching');
      } else if (motion.isMovingAway) {
        parts.add('moving away');
      } else if (motion.isStationary) {
        parts.add('stationary');
      }

      if (motion.distance.isMetric) {
        parts.add('${motion.distance.value!.toStringAsFixed(1)} meters');
      }
    }

    return parts.join(', ');
  }

  Future<void> _pickFromGallery() async {
    if (_streaming) _stopStream();
    final picker = ImagePicker();
    final picked = await picker.pickImage(source: ImageSource.gallery);
    if (picked == null) return;

    setState(() {
      _error = null;
      _requestInFlight = true;
    });
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

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: SafeArea(
        child: Stack(
          children: [
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

            if (_lastResult != null)
              Positioned(
                left: 0,
                right: 0,
                bottom: 96,
                child: _ResultsOverlay(result: _lastResult!),
              ),

            Positioned(
              left: 16,
              right: 16,
              top: 16,
              child: _VoiceStatus(
                listening: _voiceListening,
                ringing: _phoneRinging,
                transcript: _voiceTranscript,
                onStopRingtone: _stopRingtone,
              ),
            ),

            if (_error != null && _initialized)
              Positioned(
                left: 16,
                right: 16,
                top: 84,
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

            Positioned(
              left: 0,
              right: 0,
              bottom: 0,
              child: _BottomBar(
                streaming: _streaming,
                voiceListening: _voiceListening,
                onToggleStream: _initialized ? _toggleStream : null,
                onGallery: _initialized ? _pickFromGallery : null,
                onToggleVoiceDetection: _toggleVoiceDetection,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _VoiceStatus extends StatelessWidget {
  const _VoiceStatus({
    required this.listening,
    required this.ringing,
    required this.transcript,
    required this.onStopRingtone,
  });

  final bool listening;
  final bool ringing;
  final String? transcript;
  final VoidCallback onStopRingtone;

  @override
  Widget build(BuildContext context) {
    if (!listening && !ringing) return const SizedBox.shrink();

    final label = ringing ? 'Phone finder is ringing' : 'Voice detection active';
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: (ringing ? Colors.orange : Colors.teal).withOpacity(0.9),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        children: [
          Icon(ringing ? Icons.ring_volume : Icons.mic,
              color: Colors.white, size: 20),
          const SizedBox(width: 8),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  label,
                  style: const TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.w700,
                  ),
                ),
                if (!ringing && transcript != null)
                  Text(
                    transcript!,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: const TextStyle(
                      color: Colors.white70,
                      fontSize: 12,
                    ),
                  ),
              ],
            ),
          ),
          if (ringing)
            TextButton(
              onPressed: onStopRingtone,
              child: const Text(
                'STOP',
                style: TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
        ],
      ),
    );
  }
}

// ── Results overlay ──────────────────────────────────────────
// Styled to match the compact card design used in the other app build.

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

          // Object chips (label + confidence + direction/proximity badge)
          if (result.objects.isNotEmpty) ...[
            const SizedBox(height: 8),
            Wrap(
              spacing: 6,
              runSpacing: 4,
              children: result.objects.map((obj) => _ObjectChip(obj: obj)).toList(),
            ),
          ],

          // Scene tags, if the backend supplied any
          if (result.sceneTags != null && result.sceneTags!.isNotEmpty) ...[
            const SizedBox(height: 6),
            Wrap(
              spacing: 6,
              runSpacing: 4,
              children: result.sceneTags!
                  .map((tag) => Text(
                        '#$tag',
                        style: const TextStyle(
                          color: Colors.tealAccent,
                          fontSize: 11,
                          fontWeight: FontWeight.w500,
                        ),
                      ))
                  .toList(),
            ),
          ],

          // Footer: model + timing + object count + suggested direction
          const SizedBox(height: 6),
          Wrap(
            spacing: 6,
            runSpacing: 2,
            crossAxisAlignment: WrapCrossAlignment.center,
            children: [
              Text(
                result.modelUsed,
                style: const TextStyle(color: Colors.white38, fontSize: 11),
              ),
              Text(
                '· ${result.processingTimeMs.toStringAsFixed(0)} ms'
                ' · ${result.objectCount} object'
                '${result.objectCount == 1 ? '' : 's'}',
                style: const TextStyle(color: Colors.white38, fontSize: 11),
              ),
              if (result.suggestedDirection != null)
                Text(
                  '· move ${result.suggestedDirection}',
                  style: const TextStyle(
                    color: Colors.greenAccent,
                    fontSize: 11,
                  ),
                ),
              if (result.trackingEnabled)
                const Text(
                  '· tracking on',
                  style: TextStyle(color: Colors.tealAccent, fontSize: 11),
                ),
            ],
          ),
        ],
      ),
    );
  }
}

// ── Object chip ───────────────────────────────────────────────

class _ObjectChip extends StatelessWidget {
  final DetectedObject obj;
  const _ObjectChip({required this.obj});

  @override
  Widget build(BuildContext context) {
    final baseColor = _confidenceColor(obj.confidence);
    final motion = obj.motion;
    final hasReliableMotion =
        motion != null && motion.isReliable && !motion.isUnknown;
    final motionColor = hasReliableMotion ? _motionColor(motion) : null;

    // Prefer the motion badge (direction of movement + speed) when the
    // backend has reliable tracking data for this object; otherwise fall
    // back to the static direction/proximity badge as before.
    final badge = hasReliableMotion ? null : _badgeLabel(obj);

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color: baseColor.withOpacity(0.15),
        border: Border.all(
          color: (obj.isHazard ?? false)
              ? Colors.redAccent
              : (motionColor ?? baseColor),
        ),
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

          // Motion state badge (approaching / moving away / stationary),
          // only shown when the backend ran tracking for this object.
          if (hasReliableMotion) ...[
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

            // Metric speed label, only when depth is available
            if (motion.speedLabel != null) ...[
              const SizedBox(width: 3),
              Text(
                motion.speedLabel!,
                style: const TextStyle(color: Colors.white54, fontSize: 10),
              ),
            ],
          ],

          // Direction / proximity badge — fallback when no motion data
          if (badge != null) ...[
            const SizedBox(width: 4),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 5, vertical: 2),
              decoration: BoxDecoration(
                color: baseColor.withOpacity(0.25),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Text(
                badge,
                style: TextStyle(
                  color: baseColor,
                  fontSize: 10,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
          ],

          // Track ID (small, subtle) — only present while tracking
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

  String? _badgeLabel(DetectedObject object) {
    final parts = <String>[];
    if (object.direction != null && object.direction!.isNotEmpty) {
      parts.add(object.direction!.replaceAll('-', ' '));
    }
    if (object.proximity != null && object.proximity!.isNotEmpty) {
      parts.add(object.proximity!);
    }
    if (parts.isEmpty) return null;
    return parts.join(' · ');
  }

  Color _confidenceColor(double c) {
    if (c >= 0.8) return Colors.greenAccent;
    if (c >= 0.6) return Colors.orangeAccent;
    return Colors.redAccent;
  }

  /// Accent colour that signals urgency based on motion state.
  Color? _motionColor(MotionInfo? motion) {
    if (motion == null || !motion.isReliable) return null;
    switch (motion.state) {
      case 'APPROACHING':
        return Colors.orangeAccent;
      case 'MOVING_AWAY':
        return Colors.lightBlueAccent;
      case 'STATIONARY':
        return Colors.white54;
      default:
        return null;
    }
  }
}

// ── Bottom bar ───────────────────────────────────────────────

class _BottomBar extends StatelessWidget {
  final bool streaming;
  final bool voiceListening;
  final VoidCallback? onToggleStream;
  final VoidCallback? onGallery;
  final VoidCallback onToggleVoiceDetection;

  const _BottomBar({
    required this.streaming,
    required this.voiceListening,
    required this.onToggleStream,
    required this.onGallery,
    required this.onToggleVoiceDetection,
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
            icon: const Icon(Icons.photo_library_outlined,
                color: Colors.white, size: 28),
          ),
          IconButton(
            tooltip: voiceListening
                ? 'Pause voice detection'
                : 'Start voice detection',
            onPressed: onToggleVoiceDetection,
            icon: Icon(
              voiceListening ? Icons.mic : Icons.mic_none,
              color: voiceListening ? Colors.tealAccent : Colors.white,
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
                    color: streaming ? Colors.red : Colors.white, width: 3),
              ),
              child: Icon(
                streaming ? Icons.stop : Icons.play_arrow,
                color: streaming ? Colors.white : Colors.black,
                size: 32,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
