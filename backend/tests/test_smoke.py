"""Smoke tests: verify all new depth/spatial modules import and configure correctly."""

def test_depth_estimator_imports():
    from app.vision.depth_estimator import DepthEstimator, DepthEstimationError
    de = DepthEstimator()
    assert de.model_type == "DPT_Hybrid"
    assert de.person_height_m == 1.70
    assert not de.is_loaded   # not loaded without calling .load()


def test_spatial_analyzer_with_depth():
    from app.reasoning.spatial_analyzer import SpatialAnalyzer
    from app.vision.detector import Detection

    det = Detection(
        label="person", confidence=0.9,
        x=260, y=80, width=120, height=320,
        center_x=320, center_y=240,
    )
    sa = SpatialAnalyzer()
    results = sa.analyze([det], frame_width=640, frame_height=480, depths=[3.0])
    assert len(results) == 1
    assert results[0].depth_m == 3.0
    assert results[0].proximity == "close"   # 3.0m → close bucket
    assert results[0].direction == "center"


def test_spatial_proximity_depth_buckets():
    from app.reasoning.spatial_analyzer import SpatialAnalyzer
    from app.vision.detector import Detection

    sa = SpatialAnalyzer()
    det = Detection(label="chair", confidence=0.8,
                    x=200, y=100, width=80, height=100,
                    center_x=240, center_y=150)

    for depth, expected in [
        (1.0, "very close"),
        (2.5, "close"),
        (5.0, "medium"),
        (10.0, "far"),
    ]:
        r = sa.analyze([det], 640, 480, depths=[depth])
        assert r[0].proximity == expected, f"depth={depth} → expected {expected}, got {r[0].proximity}"


def test_spatial_priority_centre_over_side():
    """Centre object should rank higher than side object at same depth."""
    from app.reasoning.spatial_analyzer import SpatialAnalyzer
    from app.vision.detector import Detection

    sa = SpatialAnalyzer()
    centre = Detection(label="person", confidence=0.9,
                       x=280, y=100, width=80, height=200,
                       center_x=320, center_y=200)
    side = Detection(label="chair", confidence=0.9,
                     x=20, y=100, width=80, height=200,
                     center_x=60, center_y=200)

    results = sa.analyze([side, centre], 640, 480, depths=[3.0, 3.0])
    # Centre object (person) should come first despite being second in input
    assert results[0].detection.label == "person"


def test_scene_analyzer_includes_distance():
    """Hazard warning must include the distance in metres when depth is available."""
    from app.reasoning.spatial_analyzer import SpatialAnalyzer, SpatialDetection
    from app.reasoning.scene_analyzer import SceneAnalyzer
    from app.vision.detector import Detection

    det = Detection(label="person", confidence=0.9,
                    x=280, y=80, width=80, height=250,
                    center_x=320, center_y=205)

    sa = SpatialAnalyzer()
    sd_list = sa.analyze([det], 640, 480, depths=[2.8])

    scene = SceneAnalyzer()
    summary, _ = scene.summarize(
        sd_list, 640, 480,
        det_to_depth={id(det): 2.8},
    )
    assert "2.8 meter" in summary, f"Expected distance in summary, got: {summary}"


def test_settings_depth_keys():
    from app.config.settings import get_settings
    s = get_settings()
    assert isinstance(s.ENABLE_DEPTH, bool)
    assert s.DEPTH_MODEL_TYPE in ("DPT_Hybrid", "DPT_Large", "MiDaS_small")
    assert s.DEPTH_SCALE_FACTOR > 0
    assert s.PERSON_HEIGHT_M > 0
    assert s.DEPTH_PATCH_SIZE >= 1


def test_model_manager_depth_flag():
    from app.core.model_manager import ModelManager
    mm = ModelManager(model_path="models/yolov8n.pt", enable_depth=False)
    assert mm.depth_estimator is None
    assert not mm.depth_enabled


def test_scene_analyzer_individual_motion_and_coming_going():
    """Test that SceneAnalyzer includes distance and coming/going away in the summary."""
    from app.reasoning.spatial_analyzer import SpatialAnalyzer
    from app.reasoning.scene_analyzer import SceneAnalyzer
    from app.vision.detector import Detection
    from app.tracking.motion_analyzer import MotionResult, MotionState, MotionDirection, DistanceInfo, VelocityInfo

    det1 = Detection(label="person", confidence=0.9,
                    x=400, y=80, width=80, height=250,
                    center_x=450, center_y=205)  # center-right / right
    det2 = Detection(label="chair", confidence=0.8,
                    x=50, y=80, width=80, height=250,
                    center_x=80, center_y=205)   # left

    sa = SpatialAnalyzer()
    sd_list = sa.analyze([det1, det2], 640, 480, depths=[3.5, 4.2])

    scene = SceneAnalyzer()
    
    motion_results = {
        10: MotionResult(
            state=MotionState.APPROACHING,
            direction=MotionDirection.CENTER,
            confidence=0.9,
            distance=DistanceInfo(value=3.5, unit="meters", source="depth"),
            velocity=VelocityInfo(value=0.5, unit="m/s", type="metric"),
            observations_used=3
        ),
        20: MotionResult(
            state=MotionState.MOVING_AWAY,
            direction=MotionDirection.CENTER,
            confidence=0.8,
            distance=DistanceInfo(value=4.2, unit="meters", source="depth"),
            velocity=VelocityInfo(value=0.3, unit="m/s", type="metric"),
            observations_used=3
        )
    }

    class MockTrackedDetection:
        def __init__(self, detection, track_id):
            self.detection = detection
            self.track_id = track_id

    tracked_detections = [
        MockTrackedDetection(det1, 10),
        MockTrackedDetection(det2, 20)
    ]

    summary, _ = scene.summarize(
        sd_list, 640, 480,
        motion_results=motion_results,
        tracked_detections=tracked_detections,
        det_to_depth={id(det1): 3.5, id(det2): 4.2}
    )

    assert "3.5 meters" in summary
    assert "4.2 meters" in summary
    assert "coming" in summary
    assert "going away" in summary
    assert "person" in summary
    assert "chair" in summary


def test_alert_manager_uses_coming_and_going_away():
    """Test that AlertManager builds messages with coming and going away terminology."""
    from app.alerting.alert_manager import AlertManager, AlertType, DistanceZone

    am = AlertManager()
    
    # Test APPROACHING state
    msg_coming = am._build_message(
        label="person",
        alert_type=AlertType.DISTANCE_UPDATE,
        depth_m=3.5,
        zone=DistanceZone.NEAR,
        motion_state="APPROACHING",
        velocity_ms=0.8
    )
    assert "coming" in msg_coming
    assert "approaching" not in msg_coming

    # Test MOVING_AWAY state
    msg_going_away = am._build_message(
        label="person",
        alert_type=AlertType.DISTANCE_UPDATE,
        depth_m=4.2,
        zone=DistanceZone.NEAR,
        motion_state="MOVING_AWAY",
        velocity_ms=0.5
    )
    assert "going away" in msg_going_away
    assert "moving away" not in msg_going_away


