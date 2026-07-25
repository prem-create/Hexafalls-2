/// Walking Eye - Speech Service
///
/// Speaks scene summaries aloud so the user never has to look at the
/// screen. Frames can arrive much faster than a sentence takes to say
/// (e.g. 2 frames/sec vs. a 3-4 second sentence), so this service
/// deliberately does NOT speak every result it's given. Instead:
///
///   - Hazard warnings ALWAYS interrupt immediately, no matter what
///     is currently being said — safety must never wait in line.
///   - Non-hazard results only speak if the engine is currently idle.
///     If it's busy, the newest result is held in a single "pending"
///     slot (overwriting any older pending one) and gets spoken as
///     soon as the current sentence finishes — never queued up, so
///     the narration never falls behind and "catches up" out of sync
///     with what the camera is currently seeing.
///   - A pending result is discarded instead of spoken if it's gone
///     stale (older than [maxPendingAgeMs]) by the time its turn
///     comes up, since a fresher one will be along within a second.
///
/// Requires the `flutter_tts` package. Add this to pubspec.yaml:
///   dependencies:
///     flutter_tts: ^4.0.2
library;

import 'package:flutter_tts/flutter_tts.dart';

class SpeechService {
  final FlutterTts _tts = FlutterTts();
  bool _initialized = false;
  bool _isSpeaking = false;

  /// Avoids re-speaking the exact same sentence back-to-back while the
  /// camera holds steady on an unchanged scene.
  String? _lastSpoken;

  /// When the last hazard warning was actually spoken, used to enforce
  /// [hazardCooldownMs] between repeats of the *same* persistent hazard.
  DateTime? _lastHazardSpokenAt;

  /// How often the same ongoing hazard is allowed to repeat. A hazard
  /// that hasn't gone away (e.g. someone standing right in front of you)
  /// should still be re-announced periodically, but not on every single
  /// incoming frame.
  static const int hazardCooldownMs = 2000;

  /// The single most recent non-hazard result waiting for the engine
  /// to go idle. Only ever holds one entry — a newer result always
  /// overwrites an older, not-yet-spoken one.
  String? _pendingText;
  DateTime? _pendingSince;

  /// How stale a pending result is allowed to get before it's dropped
  /// instead of spoken.
  static const int maxPendingAgeMs = 2000;

  Future<void> init() async {
    if (_initialized) return;
    await _tts.setLanguage('en-US');
    await _tts.setSpeechRate(0.5);
    await _tts.setVolume(1.0);
    await _tts.setPitch(1.0);

    _tts.setStartHandler(() {
      _isSpeaking = true;
    });
    _tts.setCompletionHandler(() {
      _isSpeaking = false;
      _speakPendingIfAny();
    });
    _tts.setCancelHandler(() {
      _isSpeaking = false;
    });
    _tts.setErrorHandler((msg) {
      _isSpeaking = false;
      _speakPendingIfAny();
    });

    _initialized = true;
  }

  /// Called with every analysis result, as often as frames arrive.
  /// Decides internally whether to speak now, hold as pending, or
  /// skip — callers don't need to do any rate-limiting themselves.
  Future<void> speak(String text, {bool isHazard = false}) async {
    if (!_initialized) await init();

    if (isHazard) {
      final now = DateTime.now();
      final sameAsLast = text == _lastSpoken;

      if (sameAsLast) {
        if (_isSpeaking) {
          // Already saying this exact warning — let it finish instead
          // of cutting itself off every time a new (identical) frame
          // result arrives.
          return;
        }
        if (_lastHazardSpokenAt != null &&
            now.difference(_lastHazardSpokenAt!).inMilliseconds <
                hazardCooldownMs) {
          // Said this less than hazardCooldownMs ago — skip this repeat,
          // it'll come around again soon if the hazard is still there.
          return;
        }
      }

      // Either a new/changed hazard (always interrupts immediately —
      // safety first) or the same one after its cooldown has passed.
      _pendingText = null;
      _pendingSince = null;
      _lastSpoken = text;
      _lastHazardSpokenAt = now;
      await _tts.stop();
      _isSpeaking = true;
      await _tts.speak(text);
      return;
    }

    if (text == _lastSpoken) {
      // Unchanged scene — nothing new to say.
      return;
    }

    if (!_isSpeaking) {
      _lastSpoken = text;
      _isSpeaking = true;
      await _tts.speak(text);
    } else {
      // Engine is busy — hold this as the latest pending result,
      // replacing whatever was pending before.
      _pendingText = text;
      _pendingSince = DateTime.now();
    }
  }

  void _speakPendingIfAny() {
    final text = _pendingText;
    final since = _pendingSince;
    _pendingText = null;
    _pendingSince = null;

    if (text == null || since == null) return;

    final ageMs = DateTime.now().difference(since).inMilliseconds;
    if (ageMs > maxPendingAgeMs) {
      // Stale — a fresher result will arrive shortly, skip this one.
      return;
    }
    if (text == _lastSpoken) {
      return;
    }

    _lastSpoken = text;
    _isSpeaking = true;
    _tts.speak(text);
  }

  Future<void> stop() async {
    await _tts.stop();
    _isSpeaking = false;
    _lastSpoken = null;
    _lastHazardSpokenAt = null;
    _pendingText = null;
    _pendingSince = null;
  }

  void dispose() {
    _tts.stop();
  }
}
