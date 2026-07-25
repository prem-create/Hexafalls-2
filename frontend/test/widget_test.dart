import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:walking_eye/main.dart';

void main() {
  testWidgets('WalkingEyeApp builds with a camera parameter', (WidgetTester tester) async {
    final camera = CameraDescription(
      name: 'test-camera',
      lensDirection: CameraLensDirection.back,
      sensorOrientation: 90,
    );

    await tester.pumpWidget(WalkingEyeApp(cameras: [camera]));
    await tester.pump();

    expect(find.byType(MaterialApp), findsOneWidget);
  });
}
