import 'dart:async';

import 'package:flutter_ringtone_player/flutter_ringtone_player.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:speech_to_text/speech_recognition_error.dart';
import 'package:speech_to_text/speech_recognition_result.dart';
import 'package:speech_to_text/speech_to_text.dart';

import '../models/voice_command.dart';

typedef VoiceCommandHandler = FutureOr<void> Function(VoiceCommand command);
typedef VoiceListeningStateHandler = void Function(bool isListening);
typedef VoiceErrorHandler = void Function(String message);
typedef VoiceTranscriptHandler = void Function(String transcript);

/// Flutter-only voice-command service for the hackathon prototype.
///
/// Android ends system speech-recognition sessions after short pauses. While
/// active, this service restarts each completed session to provide a hands-free
/// experience without a user-facing microphone button.
class VoiceCommandService {
  VoiceCommandService({
    required this.onCommand,
    required this.onListeningStateChanged,
    required this.onError,
    required this.onTranscript,
  });

  final VoiceCommandHandler onCommand;
  final VoiceListeningStateHandler onListeningStateChanged;
  final VoiceErrorHandler onError;
  final VoiceTranscriptHandler onTranscript;

  final SpeechToText _speechToText = SpeechToText();
  final FlutterTts _tts = FlutterTts();
  final FlutterRingtonePlayer _ringtonePlayer = FlutterRingtonePlayer();

  bool _initialized = false;
  bool _listeningRequested = false;
  bool _handlingCommand = false;
  bool _ringing = false;
  bool _disposed = false;
  Timer? _restartTimer;

  bool get isListening => _listeningRequested && _speechToText.isListening;
  bool get isRinging => _ringing;

  Future<bool> startListening() async {
    if (_disposed) return false;
    _listeningRequested = true;

    try {
      final available = await _initialize();
      if (!available) {
        _stopWithError(
          'Speech recognition is unavailable. Check microphone permission.',
        );
        return false;
      }

      await _beginListening();
      return true;
    } catch (_) {
      _stopWithError('Unable to start voice detection. Please try again.');
      return false;
    }
  }

  /// Stops automatic session restarts and releases the current recognizer.
  Future<void> stopListening() async {
    _listeningRequested = false;
    _restartTimer?.cancel();
    try {
      await _speechToText.stop();
    } catch (_) {
      // The native session may already have stopped.
    }
    onListeningStateChanged(false);
  }

  Future<void> respondToPhoneFinder() async {
    try {
      await _tts.stop();
      await _tts.setLanguage('en-US');
      await _tts.setSpeechRate(0.5);
      await _tts.setVolume(1.0);
      await _tts.awaitSpeakCompletion(true);
      await _tts.speak('Here I am');
      await _ringtonePlayer.playRingtone(looping: true, asAlarm: true);
      _ringing = true;
    } catch (_) {
      onError('The phone finder response could not be played.');
    }
  }

  Future<void> stopRingtone() async {
    try {
      await _ringtonePlayer.stop();
    } catch (_) {
      onError('Unable to stop the ringtone.');
    } finally {
      _ringing = false;
    }
  }

  Future<bool> _initialize() async {
    if (_initialized) return _speechToText.isAvailable;
    final available = await _speechToText.initialize(
      onStatus: _onStatus,
      onError: _onSpeechError,
    );
    _initialized = available;
    return available;
  }

  Future<void> _beginListening() async {
    if (!_listeningRequested || _handlingCommand || _speechToText.isListening) {
      return;
    }

    await _speechToText.listen(
      onResult: _onResult,
      listenFor: const Duration(seconds: 30),
      pauseFor: const Duration(seconds: 3),
      partialResults: true,
      cancelOnError: false,
    );
    onListeningStateChanged(_speechToText.isListening);
    onTranscript('Listening for a voice command');
  }

  void _onStatus(String status) {
    final isListening = status == 'listening';
    onListeningStateChanged(isListening && _listeningRequested);

    if (_listeningRequested && !isListening && !_handlingCommand) {
      _scheduleRestart();
    }
  }

  void _onSpeechError(SpeechRecognitionError error) {
    onListeningStateChanged(false);
    if (!_listeningRequested || _disposed) return;

    if (error.permanent) {
      _stopWithError(
        'Microphone access was denied or speech recognition is unavailable.',
      );
      return;
    }
    _scheduleRestart();
  }

  void _onResult(SpeechRecognitionResult result) {
    final transcript = result.recognizedWords.trim();
    if (transcript.isEmpty) return;
    onTranscript(transcript);

    final command = _matchCommand(transcript);
    if (command != null) {
      unawaited(_handleCommand(command));
    }
  }

  VoiceCommand? _matchCommand(String text) {
    final normalized = text
        .toLowerCase()
        .replaceAll(RegExp(r'[^a-z0-9\s]'), ' ')
        .replaceAll(RegExp(r'\s+'), ' ')
        .trim();
    final words = normalized.split(' ').toSet();

    final findPhone = words.contains('where') &&
        words.contains('my') &&
        words.contains('phone') &&
        (words.contains('meera') || words.contains('mira'));
    if (findPhone) return VoiceCommand.findPhone;
    if (normalized.contains('start detection')) {
      return VoiceCommand.startDetection;
    }
    return null;
  }

  Future<void> _handleCommand(VoiceCommand command) async {
    if (_handlingCommand || _disposed) return;
    _handlingCommand = true;
    _restartTimer?.cancel();

    try {
      await _speechToText.stop();
      onListeningStateChanged(false);
      await onCommand(command);
    } catch (_) {
      onError('The voice command could not be completed.');
    } finally {
      _handlingCommand = false;
      if (_listeningRequested && !_disposed) _scheduleRestart();
    }
  }

  void _scheduleRestart() {
    _restartTimer?.cancel();
    _restartTimer = Timer(const Duration(milliseconds: 700), () {
      unawaited(_beginListening());
    });
  }

  void _stopWithError(String message) {
    _listeningRequested = false;
    _restartTimer?.cancel();
    onListeningStateChanged(false);
    onError(message);
  }

  Future<void> dispose() async {
    if (_disposed) return;
    _disposed = true;
    await stopListening();
    await stopRingtone();
    await _tts.stop();
  }
}
