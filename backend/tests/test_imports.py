"""Smoke test: verify all new modules import cleanly and the app factory works."""

def test_app_factory_creates_successfully():
    from app.main import create_app
    app = create_app()
    assert app is not None
    assert app.title == "Walking Eye - Perception Engine"


def test_tracking_package_imports():
    from app.tracking.tracker import IoUTracker, Track, TrackedDetection
    from app.tracking.temporal_buffer import TemporalBuffer, Observation
    from app.tracking.motion_analyzer import MotionAnalyzer, MotionState, MotionDirection, MotionResult
    from app.tracking.tracker_store import TrackerStore, SessionState


def test_schemas_have_new_fields():
    from app.schemas.analysis import (
        AnalysisResponse,
        DetectedObject,
        MotionInfo,
        DistanceInfo,
        VelocityInfo,
    )
    # DetectedObject must have track_id and motion fields
    fields = DetectedObject.model_fields
    assert "track_id" in fields
    assert "motion" in fields

    # AnalysisResponse must have session_id and tracking_enabled
    resp_fields = AnalysisResponse.model_fields
    assert "session_id" in resp_fields
    assert "tracking_enabled" in resp_fields


def test_settings_have_tracking_keys():
    from app.config.settings import get_settings
    s = get_settings()
    # Verify all new tracking keys are present and have the right types
    assert isinstance(s.ENABLE_TRACKING, bool)
    assert s.HISTORY_SIZE == 5
    assert s.MIN_TRACK_HISTORY == 2
    assert s.TRACKER_IOU_THRESHOLD == 0.30
    assert s.TRACKER_MAX_AGE == 5
    assert s.MAX_TRACKER_SESSIONS == 50
    assert s.STATIONARY_DEPTH_THRESHOLD > 0
    assert s.APPROACHING_DEPTH_THRESHOLD > 0
    assert s.STATIONARY_SCALE_THRESHOLD > 0
    assert s.APPROACHING_SCALE_THRESHOLD > 0
    assert s.DIRECTION_NOISE_THRESHOLD_PX >= 0


def test_tracker_store_creates_and_counts_sessions():
    from app.tracking.tracker_store import TrackerStore
    store = TrackerStore(max_sessions=5, history_size=3, iou_threshold=0.3,
                         max_age=5, min_track_history=2,
                         stationary_depth_threshold=0.08,
                         approaching_depth_threshold=0.08,
                         stationary_scale_threshold=0.05,
                         approaching_scale_threshold=0.05,
                         direction_noise_threshold_px=5)
    assert store.active_session_count == 0
    store.get_or_create_session("test-session")
    assert store.active_session_count == 1
    store.remove_session("test-session")
    assert store.active_session_count == 0


def test_analysis_service_accepts_tracker_store():
    """AnalysisService must accept tracker_store=None without error (tracking disabled path)."""
    # We can't load YOLO in a unit test, but we can verify the signature
    import inspect
    from app.services.analysis_service import AnalysisService
    sig = inspect.signature(AnalysisService.__init__)
    assert "tracker_store" in sig.parameters


def test_scene_analyzer_accepts_motion_kwargs():
    """SceneAnalyzer.summarize must accept motion_results and tracked_detections kwargs."""
    import inspect
    from app.reasoning.scene_analyzer import SceneAnalyzer
    sig = inspect.signature(SceneAnalyzer.summarize)
    assert "motion_results" in sig.parameters
    assert "tracked_detections" in sig.parameters
