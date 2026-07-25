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
      final isSimilar = _shouldSkipSpeech(text, _lastSpoken, distanceThreshold: 0.6);

      if (isSimilar) {
        if (_isSpeaking) {
          // Already saying a similar warning — let it finish instead
          // of cutting itself off on every minor frame update.
          return;
        }
        if (_lastHazardSpokenAt != null &&
            now.difference(_lastHazardSpokenAt!).inMilliseconds <
                hazardCooldownMs) {
          // Cooldown has not elapsed, skip repeat.
          return;
        }
      }

      // Either a new/changed hazard (always interrupts immediately) or the same one after its cooldown/movement.
      _pendingText = null;
      _pendingSince = null;
      _lastSpoken = text;
      _lastHazardSpokenAt = now;
      await _tts.stop();
      _isSpeaking = true;
      await _tts.speak(text);
      return;
    }

    if (_shouldSkipSpeech(text, _lastSpoken)) {
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
    if (_shouldSkipSpeech(text, _lastSpoken)) {
      return;
    }

    _lastSpoken = text;
    _isSpeaking = true;
    _tts.speak(text);
  }

  /// Normalizes spoken text by ignoring minor distance updates to prevent constant chattering.
  /// Returns true if the speech should be skipped (i.e. scene unchanged or minor differences).
  bool _shouldSkipSpeech(String newText, String? lastText, {double distanceThreshold = 0.6}) {
    if (lastText == null) return false;
    if (newText == lastText) return true;

    // Match numbers preceding 'metre' (e.g. '2.5 metres')
    final regex = RegExp(r'(\d+(?:\.\d+)?)\s*metre');
    final newNormalized = newText.replaceAll(regex, '[METRES]');
    final lastNormalized = lastText.replaceAll(regex, '[METRES]');

    if (newNormalized != lastNormalized) {
      return false; // Structural change (e.g. different object, direction, or motion state)
    }

    // Inspect individual distance changes
    final newMatches = regex.allMatches(newText).toList();
    final lastMatches = regex.allMatches(lastText).toList();

    if (newMatches.length != lastMatches.length) {
      return false; // Mismatched number of distance values
    }

    for (int i = 0; i < newMatches.length; i++) {
      final newVal = double.tryParse(newMatches[i].group(1) ?? '');
      final lastVal = double.tryParse(lastMatches[i].group(1) ?? '');
      if (newVal != null && lastVal != null) {
        if ((newVal - lastVal).abs() >= distanceThreshold) {
          return false; // Significant distance change, announce it!
        }
      }
    }

    return true; // Minor fluctuation, skip speaking
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
