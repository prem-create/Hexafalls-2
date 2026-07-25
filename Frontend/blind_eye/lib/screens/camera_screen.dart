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

  // ── Streaming via Timer + takePicture ──────────────────────

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
    if (mounted)
      setState(() {
        _streaming = false;
      });
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
      final result = await ApiService.analyzeFile(File(file.path));
      if (mounted)
        setState(() {
          _lastResult = result;
          _error = null;
        });
      _announce(result);
    } catch (e) {
      if (mounted)
        setState(() => _error = e.toString().replaceFirst('Exception: ', ''));
    } finally {
      _requestInFlight = false;
    }
  }

  /// Speaks the scene summary aloud and, for hazards, adds a haptic
  /// buzz so the warning registers even if audio is muted or missed.
  void _announce(AnalysisResult result) {
    if (result.hazardDetected) {
      HapticFeedback.heavyImpact();
    }
    _speech.speak(result.summary, isHazard: result.hazardDetected);
  }

  // ── Gallery picker ─────────────────────────────────────────

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
      final result = await ApiService.analyzeFile(File(picked.path));
      if (mounted) setState(() => _lastResult = result);
      _announce(result);
    } catch (e) {
      if (mounted)
        setState(() => _error = e.toString().replaceFirst('Exception: ', ''));
    } finally {
      if (mounted) setState(() => _requestInFlight = false);
    }
  }

  // ── Build ───────────────────────────────────────────────────

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
                  child: Text(_error!,
                      style: const TextStyle(color: Colors.red),
                      textAlign: TextAlign.center))
            else
              const Center(
                  child: CircularProgressIndicator(color: Colors.white)),

            // Results overlay
            if (_lastResult != null)
              Positioned(
                left: 0,
                right: 0,
                bottom: 100,
                child: _ResultsOverlay(result: _lastResult!),
              ),

            // Voice-command status
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

            // Error toast
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
                  child: Text(_error!,
                      style:
                          const TextStyle(color: Colors.white, fontSize: 13)),
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
                      color: Colors.white, strokeWidth: 2),
                ),
              ),

            // Bottom controls
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

    final label =
        ringing ? 'Phone finder is ringing' : 'Voice detection active';
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
                Text(label,
                    style: const TextStyle(
                        color: Colors.white, fontWeight: FontWeight.w700)),
                if (!ringing && transcript != null)
                  Text(transcript!,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style:
                          const TextStyle(color: Colors.white70, fontSize: 12)),
              ],
            ),
          ),
          if (ringing)
            TextButton(
              onPressed: onStopRingtone,
              child: const Text('STOP',
                  style: TextStyle(
                      color: Colors.white, fontWeight: FontWeight.bold)),
            ),
        ],
      ),
    );
  }
}

// ── Results overlay ──────────────────────────────────────────

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
          Text(
            result.summary,
            style: const TextStyle(
                color: Colors.white, fontSize: 16, fontWeight: FontWeight.w600),
          ),
          if (result.objects.isNotEmpty) ...[
            const SizedBox(height: 8),
            Wrap(
              spacing: 6,
              runSpacing: 4,
              children: result.objects.map((obj) {
                final color = _confidenceColor(obj.confidence);
                return Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                  decoration: BoxDecoration(
                    color: color.withOpacity(0.2),
                    border: Border.all(color: color),
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: Text(
                    '${obj.label} ${(obj.confidence * 100).toStringAsFixed(0)}%',
                    style: TextStyle(
                        color: color,
                        fontSize: 12,
                        fontWeight: FontWeight.w500),
                  ),
                );
              }).toList(),
            ),
          ],
          const SizedBox(height: 6),
          Text(
            '${result.processingTimeMs.toStringAsFixed(0)} ms · ${result.objects.length} object${result.objects.length == 1 ? '' : 's'}',
            style: const TextStyle(color: Colors.white38, fontSize: 11),
          ),
        ],
      ),
    );
  }

  Color _confidenceColor(double c) {
    if (c >= 0.8) return Colors.greenAccent;
    if (c >= 0.6) return Colors.orangeAccent;
    return Colors.redAccent;
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
