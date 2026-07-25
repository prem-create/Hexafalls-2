import 'dart:isolate';
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:image/image.dart' as img;

/// Data passed into the isolate — must be plain data, no Flutter objects
class _ConvertParams {
  final int width;
  final int height;
  final Uint8List yBytes;
  final Uint8List uBytes;
  final Uint8List vBytes;
  final int yRowStride;
  final int uvRowStride;
  final int uvPixelStride;

  _ConvertParams({
    required this.width,
    required this.height,
    required this.yBytes,
    required this.uBytes,
    required this.vBytes,
    required this.yRowStride,
    required this.uvRowStride,
    required this.uvPixelStride,
  });
}

/// Top-level function required by Isolate.run
Uint8List? _convertInIsolate(_ConvertParams p) {
  try {
    final image = img.Image(width: p.width, height: p.height);

    for (int y = 0; y < p.height; y++) {
      for (int x = 0; x < p.width; x++) {
        final int yIndex = y * p.yRowStride + x;
        final int uvIndex = (y ~/ 2) * p.uvRowStride + (x ~/ 2) * p.uvPixelStride;

        final int yVal = p.yBytes[yIndex];
        final int uVal = p.uBytes[uvIndex];
        final int vVal = p.vBytes[uvIndex];

        final int r = (yVal + 1.370705 * (vVal - 128)).round().clamp(0, 255);
        final int g = (yVal - 0.337633 * (uVal - 128) - 0.698001 * (vVal - 128)).round().clamp(0, 255);
        final int b = (yVal + 1.732446 * (uVal - 128)).round().clamp(0, 255);

        image.setPixelRgb(x, y, r, g, b);
      }
    }

    return Uint8List.fromList(img.encodeJpg(image, quality: 80));
  } catch (_) {
    return null;
  }
}

/// Converts a CameraImage (YUV420) to JPEG bytes in a background isolate.
/// Returns null if conversion fails.
Future<Uint8List?> convertYUV420toJpeg(CameraImage cameraImage) async {
  final yPlane = cameraImage.planes[0];
  final uPlane = cameraImage.planes[1];
  final vPlane = cameraImage.planes[2];

  final params = _ConvertParams(
    width: cameraImage.width,
    height: cameraImage.height,
    yBytes: Uint8List.fromList(yPlane.bytes),
    uBytes: Uint8List.fromList(uPlane.bytes),
    vBytes: Uint8List.fromList(vPlane.bytes),
    yRowStride: yPlane.bytesPerRow,
    uvRowStride: uPlane.bytesPerRow,
    uvPixelStride: uPlane.bytesPerPixel ?? 1,
  );

  return Isolate.run(() => _convertInIsolate(params));
}
