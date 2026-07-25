import 'dart:io';
import 'dart:typed_data';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';
import 'dart:convert';

import '../config/app_config.dart';
import '../models/analysis_result.dart';

class ApiService {
  /// Sends a File to the backend (used by gallery picker).
  ///
  /// [sessionId] — pass the app's stable session token so the backend can
  /// maintain tracking continuity across frames. Null disables tracking.
  static Future<AnalysisResult> analyzeFile(
    File imageFile, {
    String? sessionId,
  }) async {
    final bytes = await imageFile.readAsBytes();
    return analyzeBytes(bytes, sessionId: sessionId);
  }

  /// Sends raw JPEG bytes to the backend (used by live stream).
  ///
  /// [sessionId] — same session token used across every frame in the
  /// current streaming session so the backend tracker can link frames.
  static Future<AnalysisResult> analyzeBytes(
    Uint8List bytes, {
    String? sessionId,
  }) async {
    try {
      final uri = _buildUri(AppConfig.analyzeEndpoint, sessionId: sessionId);
      final request = http.MultipartRequest('POST', uri);

      request.files.add(
        http.MultipartFile.fromBytes(
          'image',
          bytes,
          filename: 'frame.jpg',
          contentType: MediaType('image', 'jpeg'),
        ),
      );

      final streamedResponse = await request.send().timeout(
        const Duration(seconds: AppConfig.requestTimeoutSeconds),
      );

      final response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        final json = jsonDecode(response.body) as Map<String, dynamic>;
        return AnalysisResult.fromJson(json);
      } else {
        final json = jsonDecode(response.body);
        throw Exception(json['detail'] ?? 'Server error ${response.statusCode}');
      }
    } on SocketException {
      throw Exception(
        'Cannot reach backend.\n'
        'Check IP in app_config.dart and WiFi connection.',
      );
    } on Exception {
      rethrow;
    }
  }

  // ── Private helpers ──────────────────────────────────────────────────────

  /// Builds a [Uri] for the given [endpoint], appending [sessionId] as
  /// a query parameter when it is non-null and tracking is enabled.
  static Uri _buildUri(String endpoint, {String? sessionId}) {
    final base = Uri.parse(endpoint);
    if (sessionId != null && AppConfig.enableTracking) {
      return base.replace(
        queryParameters: {
          ...base.queryParameters,
          'session_id': sessionId,
        },
      );
    }
    return base;
  }
}
