/// Matches the JSON response from POST /analyze

class BoundingBox {
  final int x, y, width, height;
  const BoundingBox({
    required this.x,
    required this.y,
    required this.width,
    required this.height,
  });
  factory BoundingBox.fromJson(Map<String, dynamic> j) => BoundingBox(
        x: j['x'], y: j['y'], width: j['width'], height: j['height'],
      );
}

class DetectionCenter {
  final int x;
  final int y;

  const DetectionCenter({required this.x, required this.y});

  factory DetectionCenter.fromJson(Map<String, dynamic> j) => DetectionCenter(
        x: j['x'],
        y: j['y'],
      );
}

// ─────────────────────────────────────────────────────────────────────────
// Motion analysis sub-models (optional — only present when the backend ran
// temporal tracking, i.e. a session_id was sent and it recognises the
// object from a previous frame). All fields are nullable/optional so a
// backend response without this data still parses exactly as before.
// ─────────────────────────────────────────────────────────────────────────

/// Distance information for a tracked object.
/// [value] is in metres when [source] == 'depth', null otherwise.
class DistanceInfo {
  /// Metric distance in metres, or null when unavailable.
  final double? value;

  /// 'meters' when metric depth is available, otherwise null.
  final String? unit;

  /// 'depth' when derived from a depth sensor/model;
  /// 'relative_bbox_scale' when bbox size is used as a proxy.
  final String source;

  const DistanceInfo({
    required this.value,
    required this.unit,
    required this.source,
  });

  factory DistanceInfo.fromJson(Map<String, dynamic> j) => DistanceInfo(
        value: (j['value'] as num?)?.toDouble(),
        unit: j['unit'] as String?,
        source: j['source'] as String,
      );

  bool get isMetric => source == 'depth' && value != null;
}

/// Velocity information for a tracked object.
/// [value] is in m/s when [type] == 'metric', null when type == 'relative'.
class VelocityInfo {
  /// Speed in m/s when metric depth is available, otherwise null.
  final double? value;

  /// 'm/s' when metric, otherwise null.
  final String? unit;

  /// 'metric' when depth-derived, 'relative' when bbox-proxy.
  final String type;

  /// Bounding-box area change per second (px²/s) when no metric depth.
  /// Not a physical velocity — used only as a relative motion indicator.
  final double? relativeApproachSpeed;

  const VelocityInfo({
    required this.value,
    required this.unit,
    required this.type,
    this.relativeApproachSpeed,
  });

  factory VelocityInfo.fromJson(Map<String, dynamic> j) => VelocityInfo(
        value: (j['value'] as num?)?.toDouble(),
        unit: j['unit'] as String?,
        type: j['type'] as String,
        relativeApproachSpeed:
            (j['relative_approach_speed'] as num?)?.toDouble(),
      );

  bool get isMetric => type == 'metric' && value != null;
}

/// Full temporal motion analysis result for one tracked object.
///
/// [state]     — 'APPROACHING' | 'MOVING_AWAY' | 'STATIONARY' | 'UNKNOWN'
/// [direction] — 'LEFT' | 'RIGHT' | 'CENTER'
class MotionInfo {
  final String state;
  final String direction;
  final double confidence;
  final DistanceInfo distance;
  final VelocityInfo velocity;
  final int observationsUsed;

  const MotionInfo({
    required this.state,
    required this.direction,
    required this.confidence,
    required this.distance,
    required this.velocity,
    required this.observationsUsed,
  });

  factory MotionInfo.fromJson(Map<String, dynamic> j) => MotionInfo(
        state: j['state'] as String,
        direction: j['direction'] as String,
        confidence: (j['confidence'] as num).toDouble(),
        distance: DistanceInfo.fromJson(j['distance'] as Map<String, dynamic>),
        velocity: VelocityInfo.fromJson(j['velocity'] as Map<String, dynamic>),
        observationsUsed: j['observations_used'] as int,
      );

  // ── Convenience helpers ────────────────────────────────────

  bool get isApproaching => state == 'APPROACHING';
  bool get isMovingAway => state == 'MOVING_AWAY';
  bool get isStationary => state == 'STATIONARY';
  bool get isUnknown => state == 'UNKNOWN';

  /// Returns true when we have enough observations and confidence to
  /// trust the classification (used to decide whether to surface it in TTS).
  bool get isReliable => observationsUsed >= 2 && confidence >= 0.50;

  /// Human-readable speed string, e.g. "1.8 m/s" or null.
  String? get speedLabel {
    if (velocity.isMetric) {
      return '${velocity.value!.toStringAsFixed(1)} ${velocity.unit}';
    }
    return null;
  }

  /// Short label for UI badges, e.g. "↓ Approaching".
  String get stateLabel {
    switch (state) {
      case 'APPROACHING':
        return '↓ Approaching';
      case 'MOVING_AWAY':
        return '↑ Moving away';
      case 'STATIONARY':
        return '• Stationary';
      default:
        return '';
    }
  }
}

class DetectedObject {
  final int id;
  final String label;
  final double confidence;
  final BoundingBox bbox;
  final DetectionCenter center;

  /// Where the object sits in the frame:
  /// 'left', 'center-left', 'center', 'center-right', 'right'.
  final String? direction;

  /// Estimated closeness: 'far', 'medium', 'close', 'very close'.
  final String? proximity;

  /// True if this object is close and in the walking path.
  final bool? isHazard;

  /// Persistent track ID across consecutive frames. Null when tracking
  /// is disabled or on the very first frame.
  final int? trackId;

  /// Temporal motion analysis result. Null on the first frame,
  /// when tracking is disabled, or when insufficient history exists.
  final MotionInfo? motion;

  const DetectedObject({
    required this.id,
    required this.label,
    required this.confidence,
    required this.bbox,
    required this.center,
    this.direction,
    this.proximity,
    this.isHazard,
    this.trackId,
    this.motion,
  });

  factory DetectedObject.fromJson(Map<String, dynamic> j) {
    final motionJson = j['motion'] as Map<String, dynamic>?;
    return DetectedObject(
      id: j['id'],
      label: j['label'],
      confidence: (j['confidence'] as num).toDouble(),
      bbox: BoundingBox.fromJson(j['bbox']),
      center: DetectionCenter.fromJson(j['center']),
      direction: j['direction'] as String?,
      proximity: j['proximity'] as String?,
      isHazard: j['is_hazard'] as bool?,
      trackId: j['track_id'] as int?,
      motion: motionJson != null ? MotionInfo.fromJson(motionJson) : null,
    );
  }
}

class AnalysisResult {
  final bool success;
  final double processingTimeMs;
        final String modelUsed;
  final int imageWidth;
  final int imageHeight;
        final int objectCount;
  final String summary;
  final List<DetectedObject> objects;
        final List<String>? sceneTags;

  /// True if any detected object is close and in the walking path.
  /// Use this to trigger a distinct alert (vibration/sound) without
  /// having to parse the summary sentence.
  final bool hazardDetected;

  /// 'left' or 'right' if that side has meaningfully more open space to
  /// move toward while a hazard blocks the path; null otherwise. Already
  /// baked into `summary` as "Move left/right." — this field is exposed
  /// separately in case the UI wants to show an arrow icon, etc.
  final String? suggestedDirection;

  /// Echo of the session_id sent by the client.
  final String? sessionId;

  /// True when the backend ran temporal tracking + motion analysis
  /// (distance, approaching/moving-away/stationary state, speed) for
  /// this frame. False when no session_id was sent or tracking isn't
  /// supported by the backend — in that case `motion` on every object
  /// stays null and nothing changes from prior behaviour.
  final bool trackingEnabled;

  const AnalysisResult({
    required this.success,
    required this.processingTimeMs,
    required this.modelUsed,
    required this.imageWidth,
    required this.imageHeight,
    required this.objectCount,
    required this.summary,
    required this.objects,
    this.sceneTags,
    this.hazardDetected = false,
    this.suggestedDirection,
    this.sessionId,
    this.trackingEnabled = false,
  });

  factory AnalysisResult.fromJson(Map<String, dynamic> j) => AnalysisResult(
        success: j['success'],
        processingTimeMs: (j['processing_time_ms'] as num).toDouble(),
        modelUsed: j['model_used'] as String? ?? 'Unknown model',
        imageWidth: j['image_width'],
        imageHeight: j['image_height'],
        objectCount: (j['object_count'] as num?)?.toInt() ??
          ((j['objects'] as List?)?.length ?? 0),
        summary: j['summary'],
        objects: (j['objects'] as List)
            .map((o) => DetectedObject.fromJson(o))
            .toList(),
        sceneTags: (j['scene_tags'] as List?)
          ?.map((tag) => tag.toString())
          .toList(),
        hazardDetected: j['hazard_detected'] as bool? ?? false,
        suggestedDirection: j['suggested_direction'] as String?,
        sessionId: j['session_id'] as String?,
        trackingEnabled: j['tracking_enabled'] as bool? ?? false,
      );

  // ── Convenience helpers ────────────────────────────────────

  /// Objects currently approaching the camera whose motion classification
  /// is reliable enough to act on (e.g. speak or highlight).
  List<DetectedObject> get approachingObjects => objects
      .where((o) =>
          o.motion != null && o.motion!.isApproaching && o.motion!.isReliable)
      .toList();

  /// The closest approaching hazard object, if any.
  DetectedObject? get primaryApproachingHazard {
    final hazards = objects.where((o) =>
        (o.isHazard ?? false) &&
        o.motion != null &&
        o.motion!.isApproaching &&
        o.motion!.isReliable);
    return hazards.isEmpty ? null : hazards.first;
  }
}
