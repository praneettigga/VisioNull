"""
Dataset Stream Module
Provides a camera-like interface for loading frames from a dataset directory
or video file, enabling offline evaluation of the pose estimation and fall
detection pipeline.

Supports:
- Image directories (jpg/png) with optional YOLO Pose label files
- Video files (mp4/avi/mkv) via OpenCV
- Ground-truth COCO 17-keypoint annotations (YOLO Pose format)
- Configurable FPS pacing or instant-mode for batch evaluation

The YOLO Pose label format per line:
    class_id cx cy w h kp1_x kp1_y kp1_v kp2_x kp2_y kp2_v ... kp17_x kp17_y kp17_v

Where coordinates are normalized [0,1] and visibility: 0=not labeled, 1=labeled but occluded, 2=visible.
"""

import cv2
import numpy as np
import os
import glob
import time
import logging
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass

from src.camera_stream import BaseCameraStream

logger = logging.getLogger(__name__)


# COCO 17-keypoint names (used in YOLO Pose format)
COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Mapping from COCO 17 keypoints to MediaPipe 33 landmark indices
# This allows comparing ground-truth annotations against MediaPipe output
COCO_TO_MEDIAPIPE = {
    0: 0,    # nose → nose
    1: 2,    # left_eye → left_eye
    2: 5,    # right_eye → right_eye
    3: 7,    # left_ear → left_ear
    4: 8,    # right_ear → right_ear
    5: 11,   # left_shoulder → left_shoulder
    6: 12,   # right_shoulder → right_shoulder
    7: 13,   # left_elbow → left_elbow
    8: 14,   # right_elbow → right_elbow
    9: 15,   # left_wrist → left_wrist
    10: 16,  # right_wrist → right_wrist
    11: 23,  # left_hip → left_hip
    12: 24,  # right_hip → right_hip
    13: 25,  # left_knee → left_knee
    14: 26,  # right_knee → right_knee
    15: 27,  # left_ankle → left_ankle
    16: 28,  # right_ankle → right_ankle
}


@dataclass
class GroundTruthAnnotation:
    """
    Ground-truth annotation for a single frame/image.

    Attributes:
        class_id: Class index (dataset-specific, e.g. 0=laying, 1=standing)
        class_name: Human-readable class name
        bbox: Bounding box as (cx, cy, w, h) in pixel coordinates
        keypoints: List of (x, y, visibility) tuples in pixel coordinates
                   for COCO 17-keypoint layout. None if no keypoints annotated.
        keypoint_names: Names of keypoints (COCO order)
    """
    class_id: int
    class_name: str
    bbox: Tuple[float, float, float, float]  # cx, cy, w, h in pixels
    keypoints: Optional[List[Tuple[float, float, float]]]  # (x, y, visibility)
    keypoint_names: List[str]


class DatasetStream(BaseCameraStream):
    """
    Camera-like stream that reads frames from a dataset directory or video file.
    Implements BaseCameraStream so it can plug into the existing pipeline.
    """

    def __init__(
        self,
        source_path: str,
        labels_path: Optional[str] = None,
        class_names: Optional[Dict[int, str]] = None,
        fps: float = 0,
        loop: bool = False,
        frame_width: int = 0,
        frame_height: int = 0
    ):
        """
        Initialize the dataset stream.

        Args:
            source_path: Path to directory of images, or path to a video file
            labels_path: Path to directory of label files (YOLO Pose format).
                        If None and source_path is a directory, looks for a
                        sibling 'labels/' directory.
            class_names: Mapping of class_id to name. Defaults to {0: "laying", 1: "standing"}.
            fps: Playback FPS for pacing. 0 = no delay (instant/batch mode).
            loop: Whether to loop back to the start after reaching the end.
            frame_width: Resize width (0 = keep original)
            frame_height: Resize height (0 = keep original)
        """
        self.source_path = source_path
        self.labels_path = labels_path
        self.class_names = class_names or {0: "laying", 1: "standing"}
        self.target_fps = fps
        self.loop = loop
        self.resize_width = frame_width
        self.resize_height = frame_height

        self._is_video = False
        self._image_paths: List[str] = []
        self._label_paths: Dict[str, str] = {}  # image_stem → label_path
        self._video_cap: Optional[cv2.VideoCapture] = None
        self._current_index = 0
        self._started = False
        self._last_frame_time = 0.0
        self._frame_count = 0
        self._current_ground_truth: Optional[List[GroundTruthAnnotation]] = None

        # Determine source type
        if os.path.isfile(source_path):
            ext = os.path.splitext(source_path)[1].lower()
            if ext in ('.mp4', '.avi', '.mkv', '.mov', '.webm'):
                self._is_video = True
            elif ext in ('.jpg', '.jpeg', '.png', '.bmp'):
                # Single image
                self._image_paths = [source_path]
            else:
                raise ValueError(f"Unsupported file type: {ext}")
        elif os.path.isdir(source_path):
            self._discover_images(source_path)
        else:
            raise FileNotFoundError(f"Source not found: {source_path}")

        # Discover labels
        if not self._is_video:
            self._discover_labels()

    def _discover_images(self, directory: str):
        """Find all image files in a directory, sorted."""
        patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        found = []
        for pat in patterns:
            found.extend(glob.glob(os.path.join(directory, pat)))
            found.extend(glob.glob(os.path.join(directory, pat.upper())))
        self._image_paths = sorted(set(found))
        logger.info(f"Found {len(self._image_paths)} images in {directory}")

    def _discover_labels(self):
        """Find label files corresponding to images."""
        if self.labels_path:
            labels_dir = self.labels_path
        elif len(self._image_paths) > 0:
            # Look for sibling 'labels/' directory
            images_dir = os.path.dirname(self._image_paths[0])
            parent_dir = os.path.dirname(images_dir)
            labels_dir = os.path.join(parent_dir, "labels")
            if not os.path.isdir(labels_dir):
                # Also try same directory
                labels_dir = os.path.join(images_dir, "..", "labels")
        else:
            return

        if not os.path.isdir(labels_dir):
            logger.info(f"No labels directory found at {labels_dir}")
            return

        labels_dir = os.path.abspath(labels_dir)
        for img_path in self._image_paths:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            label_file = os.path.join(labels_dir, stem + ".txt")
            if os.path.isfile(label_file):
                self._label_paths[stem] = label_file

        logger.info(f"Found {len(self._label_paths)} label files in {labels_dir}")

    def _parse_yolo_pose_label(
        self, label_path: str, img_width: int, img_height: int
    ) -> List[GroundTruthAnnotation]:
        """
        Parse a YOLO Pose format label file.

        Format per line:
            class_id cx cy w h kp1_x kp1_y kp1_v ... kp17_x kp17_y kp17_v

        All coordinates are normalized [0, 1].
        """
        annotations = []
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue

                    class_id = int(parts[0])
                    class_name = self.class_names.get(class_id, f"class_{class_id}")

                    # Bounding box (normalized → pixels)
                    if len(parts) >= 5:
                        cx = float(parts[1]) * img_width
                        cy = float(parts[2]) * img_height
                        w = float(parts[3]) * img_width
                        h = float(parts[4]) * img_height
                        bbox = (cx, cy, w, h)
                    else:
                        bbox = (0, 0, 0, 0)

                    # Keypoints (triplets: x, y, visibility)
                    keypoints = None
                    kp_values = parts[5:]
                    if len(kp_values) >= 51:  # 17 keypoints × 3 values
                        keypoints = []
                        for i in range(17):
                            kx = float(kp_values[i * 3]) * img_width
                            ky = float(kp_values[i * 3 + 1]) * img_height
                            kv = float(kp_values[i * 3 + 2])
                            keypoints.append((kx, ky, kv))

                    annotations.append(GroundTruthAnnotation(
                        class_id=class_id,
                        class_name=class_name,
                        bbox=bbox,
                        keypoints=keypoints,
                        keypoint_names=COCO_KEYPOINT_NAMES
                    ))

        except Exception as e:
            logger.warning(f"Error parsing label file {label_path}: {e}")

        return annotations

    def start(self) -> bool:
        """Start the dataset stream."""
        if self._is_video:
            self._video_cap = cv2.VideoCapture(self.source_path)
            if not self._video_cap.isOpened():
                logger.error(f"Cannot open video: {self.source_path}")
                return False
            logger.info(f"Opened video: {self.source_path}")
        else:
            if not self._image_paths:
                logger.error("No images found")
                return False
            logger.info(f"Dataset ready: {len(self._image_paths)} images")

        self._current_index = 0
        self._frame_count = 0
        self._started = True
        self._last_frame_time = time.time()
        return True

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get the next frame from the dataset.

        Returns BGR numpy array, or None if no more frames.
        Also updates self._current_ground_truth with annotations for this frame.
        """
        if not self._started:
            return None

        # FPS pacing
        if self.target_fps > 0:
            elapsed = time.time() - self._last_frame_time
            target_interval = 1.0 / self.target_fps
            if elapsed < target_interval:
                time.sleep(target_interval - elapsed)
            self._last_frame_time = time.time()

        frame = None
        self._current_ground_truth = None

        if self._is_video:
            ret, frame = self._video_cap.read()
            if not ret:
                if self.loop:
                    self._video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self._video_cap.read()
                    if not ret:
                        return None
                else:
                    return None
        else:
            if self._current_index >= len(self._image_paths):
                if self.loop:
                    self._current_index = 0
                else:
                    return None

            img_path = self._image_paths[self._current_index]
            frame = cv2.imread(img_path)

            if frame is None:
                logger.warning(f"Failed to read image: {img_path}")
                self._current_index += 1
                return self.get_frame()  # Skip bad images

            # Load ground truth
            stem = os.path.splitext(os.path.basename(img_path))[0]
            if stem in self._label_paths:
                h, w = frame.shape[:2]
                self._current_ground_truth = self._parse_yolo_pose_label(
                    self._label_paths[stem], w, h
                )

            self._current_index += 1

        # Resize if needed
        if frame is not None and self.resize_width > 0 and self.resize_height > 0:
            frame = cv2.resize(frame, (self.resize_width, self.resize_height))

        self._frame_count += 1
        return frame

    def get_ground_truth(self) -> Optional[List[GroundTruthAnnotation]]:
        """
        Get ground-truth annotations for the most recently returned frame.

        Returns None if no annotations are available.
        """
        return self._current_ground_truth

    def get_current_image_path(self) -> Optional[str]:
        """Get the file path of the current image (for debugging)."""
        if self._is_video or not self._image_paths:
            return None
        idx = max(0, self._current_index - 1)
        if idx < len(self._image_paths):
            return self._image_paths[idx]
        return None

    def stop(self) -> None:
        """Stop the dataset stream."""
        if self._video_cap is not None:
            self._video_cap.release()
            self._video_cap = None
        self._started = False
        logger.info("DatasetStream stopped")

    def is_opened(self) -> bool:
        """Check if the stream is active."""
        if self._is_video:
            return self._video_cap is not None and self._video_cap.isOpened()
        return self._started and self._current_index <= len(self._image_paths)

    def get_stats(self) -> Dict[str, Any]:
        """Get stream statistics."""
        total = len(self._image_paths) if not self._is_video else -1
        return {
            "frame_count": self._frame_count,
            "total_frames": total,
            "current_index": self._current_index,
            "is_video": self._is_video,
            "source": self.source_path,
            "labels_found": len(self._label_paths),
            "camera_type": "Dataset",
            "resolution": f"{self.resize_width}x{self.resize_height}" if self.resize_width > 0 else "original",
            "actual_fps": 0.0,
        }

    def __len__(self):
        """Total number of frames/images."""
        if self._is_video:
            if self._video_cap and self._video_cap.isOpened():
                return int(self._video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            return 0
        return len(self._image_paths)

    def remaining(self) -> int:
        """Number of frames remaining."""
        if self._is_video:
            return max(0, len(self) - self._frame_count)
        return max(0, len(self._image_paths) - self._current_index)
