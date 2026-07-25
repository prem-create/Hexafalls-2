import 'dart:io';
import 'dart:typed_data';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';
import 'dart:convert';

import '../config/app_config.dart';
import '../models/analysis_result.dart';

class ApiService {
  /// Sends a File to the backend (used by gallery picker).
  static Future<AnalysisResult> analyzeFile(File imageFile) async {
    final bytes = await imageFile.readAsBytes();
    return analyzeBytes(bytes);
  }

  /// Sends raw JPEG bytes to the backend (used by live stream).
  static Future<AnalysisResult> analyzeBytes(Uint8List bytes) async {
    try {
      final request = http.MultipartRequest(
        'POST',
        Uri.parse(AppConfig.analyzeEndpoint),
      );

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
}
