"""
Pipeline sub-package — camera capture, pose estimation, fall detection, and dataset utilities.
"""

from src.pipeline.camera_stream import CameraStream, BaseCameraStream, PICAMERA2_AVAILABLE
from src.pipeline.pose_estimator import PoseEstimator, Landmark
from src.pipeline.fall_detector import FallDetector, FallState, FallMetrics
from src.pipeline.dataset_stream import (
    DatasetStream,
    GroundTruthAnnotation,
    COCO_KEYPOINT_NAMES,
    COCO_TO_MEDIAPIPE,
)
