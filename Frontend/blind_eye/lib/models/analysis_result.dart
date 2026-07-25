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

class DetectedObject {
  final int id;
  final String label;
  final double confidence;
  final BoundingBox bbox;

  /// Where the object sits in the frame:
  /// 'left', 'center-left', 'center', 'center-right', 'right'.
  final String? direction;

  /// Estimated closeness: 'far', 'medium', 'close', 'very close'.
  final String? proximity;

  /// True if this object is close and in the walking path.
  final bool? isHazard;

  const DetectedObject({
    required this.id,
    required this.label,
    required this.confidence,
    required this.bbox,
    this.direction,
    this.proximity,
    this.isHazard,
  });

  factory DetectedObject.fromJson(Map<String, dynamic> j) => DetectedObject(
        id: j['id'],
        label: j['label'],
        confidence: (j['confidence'] as num).toDouble(),
        bbox: BoundingBox.fromJson(j['bbox']),
        direction: j['direction'] as String?,
        proximity: j['proximity'] as String?,
        isHazard: j['is_hazard'] as bool?,
      );
}

class AnalysisResult {
  final bool success;
  final double processingTimeMs;
  final int imageWidth;
  final int imageHeight;
  final String summary;
  final List<DetectedObject> objects;

  /// True if any detected object is close and in the walking path.
  /// Use this to trigger a distinct alert (vibration/sound) without
  /// having to parse the summary sentence.
  final bool hazardDetected;

  /// 'left' or 'right' if that side has meaningfully more open space to
  /// move toward while a hazard blocks the path; null otherwise. Already
  /// baked into `summary` as "Move left/right." — this field is exposed
  /// separately in case the UI wants to show an arrow icon, etc.
  final String? suggestedDirection;

  const AnalysisResult({
    required this.success,
    required this.processingTimeMs,
    required this.imageWidth,
    required this.imageHeight,
    required this.summary,
    required this.objects,
    this.hazardDetected = false,
    this.suggestedDirection,
  });

  factory AnalysisResult.fromJson(Map<String, dynamic> j) => AnalysisResult(
        success: j['success'],
        processingTimeMs: (j['processing_time_ms'] as num).toDouble(),
        imageWidth: j['image_width'],
        imageHeight: j['image_height'],
        summary: j['summary'],
        objects: (j['objects'] as List)
            .map((o) => DetectedObject.fromJson(o))
            .toList(),
        hazardDetected: j['hazard_detected'] as bool? ?? false,
        suggestedDirection: j['suggested_direction'] as String?,
      );
}
