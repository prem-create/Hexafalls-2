"""
Walking Eye - AI Perception Engine
Image Processor.

Handles all OpenCV-based image operations:
- Decoding raw bytes into OpenCV arrays
- Validation (format, size, corruption)
- Aspect-ratio-preserving resize
- Color space conversion

This layer is completely stateless — every function takes inputs
and returns outputs with no side effects. Easy to test in isolation.
"""

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from app.utilities.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ProcessedImage:
    """
    Result of image processing pipeline.
    Carries the image array alongside its metadata.
    """

    array: np.ndarray        # BGR image array (OpenCV native format)
    original_width: int
    original_height: int
    processed_width: int
    processed_height: int
    was_resized: bool


class ImageProcessingError(Exception):
    """Raised when image decoding or validation fails."""
    pass


class ImageProcessor:
    """
    Stateless image processing pipeline.

    All methods are instance methods for consistency and future
    extensibility (e.g. injecting settings), but hold no mutable state.
    """

    def __init__(self, max_dimension: int = 1280) -> None:
        """
        Args:
            max_dimension: Longest side of the image will be resized
                           to this value if exceeded. Aspect ratio preserved.
        """
        self.max_dimension = max_dimension

    def decode(self, raw_bytes: bytes) -> np.ndarray:
        """
        Decodes raw image bytes into an OpenCV BGR numpy array.

        Args:
            raw_bytes: Raw bytes from the uploaded file.

        Returns:
            BGR numpy array.

        Raises:
            ImageProcessingError: If bytes cannot be decoded as an image.
        """
        if not raw_bytes:
            raise ImageProcessingError("Received empty image data.")

        np_array = np.frombuffer(raw_bytes, dtype=np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        if image is None:
            raise ImageProcessingError(
                "Could not decode image. "
                "The file may be corrupted or in an unsupported format."
            )

        logger.debug(f"Decoded image shape: {image.shape}")
        return image

    def validate(self, image: np.ndarray) -> None:
        """
        Validates that the decoded image array is usable.

        Checks:
        - Array has correct number of dimensions
        - Neither dimension is zero
        - Image is not pure black / fully empty (basic corruption check)

        Args:
            image: BGR numpy array.

        Raises:
            ImageProcessingError: If validation fails.
        """
        if image.ndim not in (2, 3):
            raise ImageProcessingError(
                f"Unexpected image dimensions: {image.ndim}. Expected 2 or 3."
            )

        h, w = image.shape[:2]

        if h == 0 or w == 0:
            raise ImageProcessingError(
                f"Image has zero dimension: width={w}, height={h}."
            )

        if h < 16 or w < 16:
            raise ImageProcessingError(
                f"Image is too small to process: {w}x{h} pixels. "
                "Minimum size is 16x16."
            )

    def resize_if_needed(self, image: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Resizes image so its longest side does not exceed max_dimension.
        Aspect ratio is always preserved. No upscaling.

        Args:
            image: BGR numpy array.

        Returns:
            Tuple of (resized_or_original_array, was_resized).
        """
        h, w = image.shape[:2]
        longest = max(h, w)

        if longest <= self.max_dimension:
            return image, False

        scale = self.max_dimension / longest
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        logger.debug(f"Resized image from {w}x{h} to {new_w}x{new_h}")
        return resized, True

    def to_rgb(self, image: np.ndarray) -> np.ndarray:
        """
        Converts BGR (OpenCV default) to RGB.
        YOLO via Ultralytics accepts both, but RGB is safer for
        future integrations (Pillow, matplotlib, LLM vision APIs).

        Args:
            image: BGR numpy array.

        Returns:
            RGB numpy array.
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def process(self, raw_bytes: bytes) -> ProcessedImage:
        """
        Full processing pipeline: decode → validate → resize.

        This is the main entry point called by the service layer.

        Args:
            raw_bytes: Raw bytes from the uploaded image file.

        Returns:
            ProcessedImage dataclass with array and metadata.

        Raises:
            ImageProcessingError: On any processing failure.
        """
        # Step 1: Decode
        image = self.decode(raw_bytes)

        # Step 2: Validate
        self.validate(image)

        original_h, original_w = image.shape[:2]
        logger.info(f"Processing image: {original_w}x{original_h} px")

        # Step 3: Resize if needed (preserves aspect ratio)
        image, was_resized = self.resize_if_needed(image)

        processed_h, processed_w = image.shape[:2]

        return ProcessedImage(
            array=image,
            original_width=original_w,
            original_height=original_h,
            processed_width=processed_w,
            processed_height=processed_h,
            was_resized=was_resized,
        )
