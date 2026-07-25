"""
Walking Eye - AI Perception Engine
Depth Estimator.

Wraps MiDaS DPT_Hybrid (loaded via torch.hub) to produce per-object
approximate depth in metres from a single RGB image.

How it works
------------
1. MiDaS runs on the whole image and produces a relative inverse-depth map
   (higher value = closer to camera).  The output is unitless.

2. Scale calibration — two modes controlled by settings:

   Person-anchored (DEPTH_PERSON_ANCHOR=True, recommended):
     If a person bbox is present in the detections list, we know a person
     is roughly PERSON_HEIGHT_M tall.  We measure the median MiDaS value
     inside that bbox and compute:
         scale = (PERSON_HEIGHT_M * median_person_depth) / 1.0
     ... actually we use the geometric relationship:
         apparent_depth_ratio ∝ bbox_height / image_height
     Combined with MiDaS inverse-depth, the per-frame scale is:
         scale = PERSON_HEIGHT_M / (bbox_height_fraction * median_inv_depth)
     This auto-adapts each frame and is more accurate than a fixed factor.

   Fixed-scale fallback (DEPTH_PERSON_ANCHOR=False, or no person visible):
     depth_metres = DEPTH_SCALE_FACTOR / inverse_depth_value
     DEPTH_SCALE_FACTOR defaults to 7.5 which is empirically tuned for
     typical walking scenes at 1–10 m range.

3. Per-object depth sampling:
   For each detected bounding box, we take the median of a DEPTH_PATCH_SIZE²
   patch centred on the bbox centre.  Median is more robust than mean against
   depth discontinuities at object edges.

Important caveats
-----------------
- This produces *approximate* metres.  Do not treat as survey-grade distance.
- Person-anchored mode is only as good as the person detector confidence and
  the assumption that the person is standing upright (not crouching/seated).
- MiDaS DPT_Hybrid (~400 MB) is downloaded from PyTorch Hub on first run and
  cached in ~/.cache/torch/hub/.

Thread safety
-------------
torch.hub models are not thread-safe by default.  FastAPI runs async handlers
on a single event loop; inference calls are synchronous and therefore safe
as long as we do not share the model across threads.  The model is loaded once
in app.state and called sequentially per request.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.utilities.logger import get_logger
from app.vision.detector import Detection

logger = get_logger(__name__)

# Average person height fraction of image height that corresponds to
# "person at 1 metre".  Used as a sanity-check guard.
_MIN_PERSON_BBOX_HEIGHT_FRACTION = 0.02   # person must be at least 2 % of frame


class DepthEstimationError(Exception):
    """Raised when depth inference fails unexpectedly."""
    pass


class DepthEstimator:
    """
    MiDaS-based monocular depth estimator.

    Load once at startup via DepthEstimator.load(), then call
    estimate_depths() per frame.
    """

    def __init__(
        self,
        model_type: str = "DPT_Hybrid",
        scale_factor: float = 7.5,
        person_anchor: bool = True,
        person_height_m: float = 1.70,
        patch_size: int = 11,
    ) -> None:
        self.model_type = model_type
        self.scale_factor = scale_factor
        self.person_anchor = person_anchor
        self.person_height_m = person_height_m
        self.patch_size = patch_size

        self._model = None
        self._transform = None
        self._device = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Downloads (first run) and loads the MiDaS model.
        Moves to CUDA if available, otherwise CPU.
        Called once at application startup.
        """
        import torch

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(
            f"Loading MiDaS {self.model_type} on {self._device} ..."
        )

        try:
            start = time.perf_counter()
            # torch.hub downloads to ~/.cache/torch/hub/ on first call
            self._model = torch.hub.load(
                "intel-isl/MiDaS",
                self.model_type,
                trust_repo=True,
            )
            self._model.to(self._device)
            self._model.eval()

            # Load the matching transform for this model type
            midas_transforms = torch.hub.load(
                "intel-isl/MiDaS",
                "transforms",
                trust_repo=True,
            )
            if self.model_type in ("DPT_Large", "DPT_Hybrid"):
                self._transform = midas_transforms.dpt_transform
            else:
                self._transform = midas_transforms.small_transform

            elapsed = (time.perf_counter() - start) * 1000
            logger.info(
                f"MiDaS {self.model_type} loaded in {elapsed:.0f} ms "
                f"on {self._device}"
            )
            self._loaded = True

            # Warmup pass
            self._warmup()

        except Exception as e:
            logger.error(f"Failed to load MiDaS model: {e}")
            raise RuntimeError(f"Depth model loading failed: {e}") from e

    def _warmup(self) -> None:
        """Single silent inference pass to initialise CUDA kernels."""
        try:
            import torch
            dummy = np.zeros((384, 384, 3), dtype=np.uint8)
            with torch.no_grad():
                inp = self._transform(dummy).to(self._device)
                self._model(inp)
            logger.info("MiDaS warmup complete.")
        except Exception as e:
            logger.warning(f"MiDaS warmup skipped: {e}")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------
    # Public inference API
    # ------------------------------------------------------------------

    def estimate_depths(
        self,
        image_bgr: np.ndarray,
        detections: List[Detection],
    ) -> List[Optional[float]]:
        """
        Runs MiDaS on the full image and returns one depth value per detection.

        Args:
            image_bgr:  BGR numpy array (H × W × 3) from OpenCV.
            detections: YOLO detections for this frame (same order as returned
                        by ObjectDetector.detect()).

        Returns:
            List of float | None, parallel to `detections`.
            Each entry is the approximate distance in metres to that object's
            centre, or None if estimation failed for that detection.
        """
        if not self._loaded:
            logger.warning("DepthEstimator.estimate_depths called before load().")
            return [None] * len(detections)

        if not detections:
            return []

        try:
            depth_map = self._run_midas(image_bgr)
        except Exception as e:
            logger.error(f"MiDaS inference failed: {e}")
            return [None] * len(detections)

        h, w = image_bgr.shape[:2]

        # --- Compute per-frame scale ---
        scale = self._compute_scale(depth_map, detections, image_height=h)

        # --- Sample depth for each detection ---
        results: List[Optional[float]] = []
        for det in detections:
            depth_m = self._sample_object_depth(depth_map, det, scale, h, w)
            results.append(depth_m)

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_midas(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Runs MiDaS inference and returns a normalised inverse-depth map
        resized back to the original image dimensions.

        Returns:
            Float32 numpy array (H × W), values in [0, 1].
            Higher = closer to the camera.
        """
        import torch

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = image_rgb.shape[:2]

        start = time.perf_counter()
        with torch.no_grad():
            inp = self._transform(image_rgb).to(self._device)
            prediction = self._model(inp)
            # Interpolate back to original size
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(h_orig, w_orig),
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_raw = prediction.cpu().numpy().astype(np.float32)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.debug(f"MiDaS inference: {elapsed_ms:.1f} ms")

        # Normalise to [0, 1] — avoids numerical issues in scale computation
        d_min, d_max = depth_raw.min(), depth_raw.max()
        if d_max - d_min < 1e-6:
            return np.zeros_like(depth_raw)
        depth_norm = (depth_raw - d_min) / (d_max - d_min)
        return depth_norm

    def _compute_scale(
        self,
        depth_map: np.ndarray,
        detections: List[Detection],
        image_height: int,
    ) -> float:
        """
        Returns a scale factor: depth_metres = scale / inverse_depth_value.

        Person-anchored mode:
            Find the best person detection (highest confidence), sample its
            median inverse-depth, and compute scale from known height.

        Fixed-scale fallback:
            Return DEPTH_SCALE_FACTOR directly.
        """
        if self.person_anchor:
            person_scale = self._person_anchored_scale(
                depth_map, detections, image_height
            )
            if person_scale is not None:
                logger.debug(f"Person-anchored depth scale: {person_scale:.4f}")
                return person_scale

        logger.debug(f"Using fixed depth scale: {self.scale_factor}")
        return self.scale_factor

    def _person_anchored_scale(
        self,
        depth_map: np.ndarray,
        detections: List[Detection],
        image_height: int,
    ) -> Optional[float]:
        """
        Estimate scale from the most-confident person detection.

        Geometric relationship:
            At distance D metres, a person of height H_real metres projects
            to bbox_height pixels.  Pinhole camera model (approximate):
                D ≈ focal_length * H_real / bbox_height_pixels

            MiDaS gives us inv_depth ∝ 1/D, so:
                scale = D * inv_depth_sample
                      = (H_real / bbox_height_fraction) * inv_depth_sample
            where bbox_height_fraction = bbox_height / image_height.

        This is a linear approximation — accurate enough for 1–10 m range.
        """
        best_person: Optional[Detection] = None
        best_conf = 0.0
        for det in detections:
            if det.label == "person" and det.confidence > best_conf:
                best_conf = det.confidence
                best_person = det

        if best_person is None:
            return None

        bbox_h_fraction = best_person.height / max(image_height, 1)
        if bbox_h_fraction < _MIN_PERSON_BBOX_HEIGHT_FRACTION:
            # Person too small / far — scale estimate unreliable
            return None

        inv_depth = self._median_patch(
            depth_map, best_person.center_x, best_person.center_y
        )
        if inv_depth < 1e-4:
            return None

        # Avoid extreme scale values that indicate bad depth map regions
        scale = (self.person_height_m / bbox_h_fraction) * inv_depth
        if scale < 0.5 or scale > 200.0:
            return None

        return float(scale)

    def _sample_object_depth(
        self,
        depth_map: np.ndarray,
        det: Detection,
        scale: float,
        img_h: int,
        img_w: int,
    ) -> Optional[float]:
        """
        Converts the median inverse-depth at an object's centre to metres.

        depth_metres = scale / inv_depth
        (Because MiDaS outputs relative inverse depth: larger = closer.)
        """
        cx = int(np.clip(det.center_x, 0, img_w - 1))
        cy = int(np.clip(det.center_y, 0, img_h - 1))

        inv_depth = self._median_patch(depth_map, cx, cy)
        if inv_depth < 1e-4:
            return None

        depth_m = scale / inv_depth
        # Sanity clamp: don't report depths > 50 m or < 0.1 m
        depth_m = float(np.clip(depth_m, 0.1, 50.0))
        return round(depth_m, 2)

    def _median_patch(
        self,
        depth_map: np.ndarray,
        cx: int,
        cy: int,
    ) -> float:
        """
        Returns the median depth value in a patch_size × patch_size window
        centred on (cx, cy).  Clamped to image bounds.
        """
        h, w = depth_map.shape
        half = self.patch_size // 2
        y1 = max(0, cy - half)
        y2 = min(h, cy + half + 1)
        x1 = max(0, cx - half)
        x2 = min(w, cx + half + 1)
        patch = depth_map[y1:y2, x1:x2]
        if patch.size == 0:
            return 0.0
        return float(np.median(patch))
