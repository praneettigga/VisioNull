"""
Pose Estimator Module
Wrapper around MediaPipe Pose for lightweight pose estimation.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, List, Tuple, NamedTuple
from dataclasses import dataclass


@dataclass
class Landmark:
    """
    Represents a single pose landmark.
    
    Attributes:
        x: X coordinate in pixels
        y: Y coordinate in pixels
        z: Z coordinate (depth, relative to hips)
        visibility: Confidence score for this landmark (0-1)
        name: Name of the landmark (e.g., 'nose', 'left_shoulder')
    """
    x: float
    y: float
    z: float
    visibility: float
    name: str


class PoseEstimator:
    """
    Wrapper around MediaPipe Pose for efficient pose estimation.
    Uses the lightweight model suitable for Raspberry Pi.
    """
    
    # MediaPipe Pose landmark names (indices 0-32)
    LANDMARK_NAMES = [
        'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
        'right_eye_inner', 'right_eye', 'right_eye_outer',
        'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
        'left_index', 'right_index', 'left_thumb', 'right_thumb',
        'left_hip', 'right_hip', 'left_knee', 'right_knee',
        'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
        'left_foot_index', 'right_foot_index'
    ]
    
    # Key landmark indices for fall detection
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    
    def __init__(
        self,
        model_complexity: int = 0,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = False
    ):
        """
        Initialize the pose estimator.
        
        Args:
            model_complexity: 0 (lite), 1 (full), or 2 (heavy). Use 0 for Raspberry Pi.
            min_detection_confidence: Minimum confidence for person detection
            min_tracking_confidence: Minimum confidence for landmark tracking
            static_image_mode: If True, treats each frame independently (slower but more accurate)
        """
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        print(f"PoseEstimator initialized (model_complexity={model_complexity})")
    
    def get_landmarks(
        self,
        frame_bgr: np.ndarray
    ) -> Optional[List[Landmark]]:
        """
        Extract pose landmarks from a BGR frame.
        
        Args:
            frame_bgr: Input frame in BGR format (OpenCV default)
            
        Returns:
            List of Landmark objects with pixel coordinates, or None if no person detected
        """
        if frame_bgr is None:
            return None
        
        # Get frame dimensions
        height, width = frame_bgr.shape[:2]
        
        # Convert BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.pose.process(frame_rgb)
        
        # Check if pose was detected
        if results.pose_landmarks is None:
            return None
        
        # Convert normalized landmarks to pixel coordinates
        landmarks = []
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            landmark = Landmark(
                x=lm.x * width,
                y=lm.y * height,
                z=lm.z,  # Keep relative depth as-is
                visibility=lm.visibility,
                name=self.LANDMARK_NAMES[idx]
            )
            landmarks.append(landmark)
        
        return landmarks
    
    def draw_landmarks(
        self,
        frame_bgr: np.ndarray,
        landmarks: Optional[List[Landmark]] = None,
        draw_connections: bool = True,
        landmark_color: Tuple[int, int, int] = (0, 255, 0),
        connection_color: Tuple[int, int, int] = (255, 255, 255)
    ) -> np.ndarray:
        """
        Draw pose landmarks on a frame.
        
        Args:
            frame_bgr: Input frame in BGR format
            landmarks: List of Landmark objects (if None, will run detection)
            draw_connections: Whether to draw skeleton connections
            landmark_color: BGR color for landmark points
            connection_color: BGR color for connections
            
        Returns:
            Frame with landmarks drawn
        """
        output_frame = frame_bgr.copy()
        
        # If no landmarks provided, detect them
        if landmarks is None:
            landmarks = self.get_landmarks(frame_bgr)
        
        if landmarks is None:
            return output_frame
        
        # Draw landmarks
        for lm in landmarks:
            if lm.visibility > 0.5:  # Only draw visible landmarks
                cv2.circle(
                    output_frame,
                    (int(lm.x), int(lm.y)),
                    4,
                    landmark_color,
                    -1
                )
        
        # Draw connections (skeleton)
        if draw_connections:
            connections = [
                # Torso
                (self.LEFT_SHOULDER, self.RIGHT_SHOULDER),
                (self.LEFT_SHOULDER, self.LEFT_HIP),
                (self.RIGHT_SHOULDER, self.RIGHT_HIP),
                (self.LEFT_HIP, self.RIGHT_HIP),
                # Left arm
                (self.LEFT_SHOULDER, 13),  # left_elbow
                (13, 15),  # left_wrist
                # Right arm
                (self.RIGHT_SHOULDER, 14),  # right_elbow
                (14, 16),  # right_wrist
                # Left leg
                (self.LEFT_HIP, self.LEFT_KNEE),
                (self.LEFT_KNEE, self.LEFT_ANKLE),
                # Right leg
                (self.RIGHT_HIP, self.RIGHT_KNEE),
                (self.RIGHT_KNEE, self.RIGHT_ANKLE),
                # Head
                (self.NOSE, self.LEFT_SHOULDER),
                (self.NOSE, self.RIGHT_SHOULDER),
            ]
            
            for start_idx, end_idx in connections:
                if (landmarks[start_idx].visibility > 0.5 and 
                    landmarks[end_idx].visibility > 0.5):
                    start_point = (int(landmarks[start_idx].x), int(landmarks[start_idx].y))
                    end_point = (int(landmarks[end_idx].x), int(landmarks[end_idx].y))
                    cv2.line(output_frame, start_point, end_point, connection_color, 2)
        
        return output_frame
    
    def get_bounding_box(
        self,
        landmarks: List[Landmark],
        padding: float = 0.1
    ) -> Tuple[int, int, int, int]:
        """
        Calculate bounding box around detected pose.
        
        Args:
            landmarks: List of Landmark objects
            padding: Padding as fraction of box size
            
        Returns:
            Tuple of (x1, y1, x2, y2) in pixel coordinates
        """
        visible_landmarks = [lm for lm in landmarks if lm.visibility > 0.5]
        
        if not visible_landmarks:
            return (0, 0, 0, 0)
        
        x_coords = [lm.x for lm in visible_landmarks]
        y_coords = [lm.y for lm in visible_landmarks]
        
        x1, x2 = min(x_coords), max(x_coords)
        y1, y2 = min(y_coords), max(y_coords)
        
        # Add padding
        width = x2 - x1
        height = y2 - y1
        x1 -= width * padding
        x2 += width * padding
        y1 -= height * padding
        y2 += height * padding
        
        return (int(x1), int(y1), int(x2), int(y2))
    
    def close(self) -> None:
        """Release MediaPipe resources."""
        self.pose.close()
        print("PoseEstimator closed")


def main():
    """
    Test the pose estimator by displaying skeleton overlay on camera feed.
    Press 'q' to quit.
    """
    from camera_stream import CameraStream
    
    print("=" * 50)
    print("Pose Estimator Test")
    print("=" * 50)
    print("Press 'q' to quit")
    print()
    
    # Initialize camera and pose estimator
    camera = CameraStream(camera_index=0, frame_width=640, frame_height=480)
    pose_estimator = PoseEstimator(model_complexity=0)
    
    if not camera.start():
        print("Failed to start camera")
        return
    
    try:
        frame_count = 0
        while True:
            frame = camera.get_frame()
            
            if frame is None:
                print("Lost camera connection")
                break
            
            # Get pose landmarks
            landmarks = pose_estimator.get_landmarks(frame)
            
            # Draw skeleton on frame
            output_frame = pose_estimator.draw_landmarks(frame, landmarks)
            
            # Draw bounding box if person detected
            if landmarks:
                bbox = pose_estimator.get_bounding_box(landmarks)
                cv2.rectangle(
                    output_frame,
                    (bbox[0], bbox[1]),
                    (bbox[2], bbox[3]),
                    (255, 0, 255),
                    2
                )
                status = "Person Detected"
                color = (0, 255, 0)
            else:
                status = "No Person Detected"
                color = (0, 0, 255)
            
            # Add status overlay
            frame_count += 1
            cv2.putText(
                output_frame,
                f"Frame: {frame_count} | {status}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )
            
            cv2.putText(
                output_frame,
                "Press 'q' to quit",
                (10, output_frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )
            
            # Display the frame
            cv2.imshow("Pose Estimator Test", output_frame)
            
            # Check for quit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        pose_estimator.close()
        camera.stop()
        cv2.destroyAllWindows()
        print("Test completed")


if __name__ == "__main__":
    main()
