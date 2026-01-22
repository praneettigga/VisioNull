"""
Fall Detector Module
Rule-based fall detection using pose landmarks from MediaPipe.

Fall Detection Logic:
1. Compute body orientation (vertical vs horizontal) using shoulder-hip angle
2. Track head height (y position) relative to frame
3. Detect rapid downward head movement (velocity)
4. Confirm fall when body is horizontal AND head is low for several frames
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
from enum import Enum

# Import Landmark from pose_estimator (for type hints)
try:
    from src.pose_estimator import Landmark, PoseEstimator
except ImportError:
    from pose_estimator import Landmark, PoseEstimator


class FallState(Enum):
    """Possible states for fall detection."""
    UNKNOWN = "unknown"
    STANDING = "standing"
    FALLING = "falling"
    FALLEN = "fallen"


@dataclass
class FallMetrics:
    """
    Debug metrics for fall detection.
    
    Attributes:
        body_angle: Angle of body from vertical (0=upright, 90=horizontal)
        head_height_ratio: Head Y position as ratio of frame height (0=top, 1=bottom)
        head_velocity: Rate of head movement (positive = moving down)
        shoulder_hip_ratio: Ratio of horizontal to vertical distance (>1 = horizontal)
        confidence: Confidence in current state (0-1)
    """
    body_angle: float
    head_height_ratio: float
    head_velocity: float
    shoulder_hip_ratio: float
    confidence: float


class FallDetector:
    """
    Rule-based fall detector using pose landmarks.
    
    The detection logic:
    1. Body Orientation: Compare vertical vs horizontal span of torso
       - Standing: shoulders-to-hips is more vertical than horizontal
       - Fallen: shoulders-to-hips is more horizontal than vertical
    
    2. Head Position: Track where the head (nose) is in the frame
       - Standing: head is in upper portion of frame
       - Fallen: head drops to lower portion of frame
    
    3. Temporal Analysis: Use frame history to detect:
       - Sudden downward movement (falling motion)
       - Sustained horizontal position (confirmed fall)
    """
    
    def __init__(
        self,
        history_size: int = 15,
        fall_head_threshold: float = 0.65,
        horizontal_ratio_threshold: float = 0.8,
        fall_confirm_frames: int = 8,
        head_velocity_threshold: float = 15.0,
        recovery_frames: int = 10
    ):
        """
        Initialize the fall detector.
        
        Args:
            history_size: Number of frames to keep in history
            fall_head_threshold: Head Y ratio (0-1) above which is considered "low"
            horizontal_ratio_threshold: Shoulder-hip width/height ratio above which is "horizontal"
            fall_confirm_frames: Frames the person must be horizontal+low to confirm fall
            head_velocity_threshold: Minimum head velocity (pixels/frame) to detect falling motion
            recovery_frames: Frames standing before exiting fallen state
        """
        self.history_size = history_size
        self.fall_head_threshold = fall_head_threshold
        self.horizontal_ratio_threshold = horizontal_ratio_threshold
        self.fall_confirm_frames = fall_confirm_frames
        self.head_velocity_threshold = head_velocity_threshold
        self.recovery_frames = recovery_frames
        
        # State tracking
        self.current_state = FallState.UNKNOWN
        self.head_history: deque = deque(maxlen=history_size)
        self.state_history: deque = deque(maxlen=history_size)
        self.horizontal_count = 0
        self.standing_count = 0
        self.frame_height = 480  # Will be updated from landmarks
        
        print(f"FallDetector initialized:")
        print(f"  Head threshold: {fall_head_threshold}")
        print(f"  Horizontal ratio: {horizontal_ratio_threshold}")
        print(f"  Confirm frames: {fall_confirm_frames}")
    
    def _get_midpoint(self, lm1: Landmark, lm2: Landmark) -> Tuple[float, float]:
        """Get midpoint between two landmarks."""
        return ((lm1.x + lm2.x) / 2, (lm1.y + lm2.y) / 2)
    
    def _compute_body_metrics(
        self,
        landmarks: List[Landmark],
        frame_height: int
    ) -> Tuple[float, float, float]:
        """
        Compute body orientation metrics.
        
        Returns:
            Tuple of (shoulder_hip_ratio, body_angle, head_height_ratio)
        """
        # Get key landmarks
        nose = landmarks[PoseEstimator.NOSE]
        left_shoulder = landmarks[PoseEstimator.LEFT_SHOULDER]
        right_shoulder = landmarks[PoseEstimator.RIGHT_SHOULDER]
        left_hip = landmarks[PoseEstimator.LEFT_HIP]
        right_hip = landmarks[PoseEstimator.RIGHT_HIP]
        
        # Calculate shoulder and hip midpoints
        shoulder_mid = self._get_midpoint(left_shoulder, right_shoulder)
        hip_mid = self._get_midpoint(left_hip, right_hip)
        
        # Calculate torso dimensions
        torso_vertical = abs(hip_mid[1] - shoulder_mid[1])  # Y distance
        torso_horizontal = abs(hip_mid[0] - shoulder_mid[0])  # X distance
        
        # Also consider shoulder width vs torso height
        shoulder_width = abs(right_shoulder.x - left_shoulder.x)
        
        # Compute shoulder-hip ratio (horizontal / vertical)
        # High ratio = body is horizontal
        if torso_vertical > 10:  # Avoid division by very small numbers
            shoulder_hip_ratio = (torso_horizontal + shoulder_width * 0.5) / torso_vertical
        else:
            shoulder_hip_ratio = 10.0  # Very horizontal
        
        # Compute body angle from vertical (0 = upright, 90 = horizontal)
        if torso_vertical > 1:
            body_angle = np.degrees(np.arctan2(torso_horizontal, torso_vertical))
        else:
            body_angle = 90.0
        
        # Compute head height ratio (0 = top, 1 = bottom)
        head_height_ratio = nose.y / frame_height
        
        return shoulder_hip_ratio, body_angle, head_height_ratio
    
    def _compute_head_velocity(self) -> float:
        """
        Compute head velocity from history (positive = moving down).
        """
        if len(self.head_history) < 3:
            return 0.0
        
        # Use last few frames to compute velocity
        recent = list(self.head_history)[-5:]
        if len(recent) < 2:
            return 0.0
        
        # Simple velocity: change in Y over frames
        velocity = (recent[-1] - recent[0]) / len(recent)
        return velocity
    
    def update(
        self,
        landmarks: Optional[List[Landmark]],
        frame_height: int = 480
    ) -> Tuple[FallState, FallMetrics]:
        """
        Update fall detection state with new landmarks.
        
        Args:
            landmarks: List of pose landmarks, or None if no person detected
            frame_height: Height of the video frame in pixels
            
        Returns:
            Tuple of (current_state, debug_metrics)
        """
        self.frame_height = frame_height
        
        # Handle no detection
        if landmarks is None:
            self.head_history.clear()
            self.horizontal_count = 0
            self.standing_count = 0
            # Keep current state but with low confidence
            return self.current_state, FallMetrics(
                body_angle=0,
                head_height_ratio=0,
                head_velocity=0,
                shoulder_hip_ratio=0,
                confidence=0.0
            )
        
        # Check visibility of key landmarks
        key_indices = [
            PoseEstimator.NOSE,
            PoseEstimator.LEFT_SHOULDER,
            PoseEstimator.RIGHT_SHOULDER,
            PoseEstimator.LEFT_HIP,
            PoseEstimator.RIGHT_HIP
        ]
        
        for idx in key_indices:
            if landmarks[idx].visibility < 0.3:
                # Key landmark not visible enough
                return self.current_state, FallMetrics(
                    body_angle=0,
                    head_height_ratio=0,
                    head_velocity=0,
                    shoulder_hip_ratio=0,
                    confidence=0.2
                )
        
        # Compute metrics
        shoulder_hip_ratio, body_angle, head_height_ratio = self._compute_body_metrics(
            landmarks, frame_height
        )
        
        # Update head history
        self.head_history.append(landmarks[PoseEstimator.NOSE].y)
        
        # Compute head velocity
        head_velocity = self._compute_head_velocity()
        
        # Determine if body is horizontal
        is_horizontal = (
            shoulder_hip_ratio > self.horizontal_ratio_threshold or
            body_angle > 45
        )
        
        # Determine if head is low
        is_head_low = head_height_ratio > self.fall_head_threshold
        
        # Determine if currently falling (rapid downward motion)
        is_falling_motion = head_velocity > self.head_velocity_threshold
        
        # State machine logic
        confidence = 0.5
        
        if is_horizontal and is_head_low:
            # Potential fall condition
            self.horizontal_count += 1
            self.standing_count = 0
            
            if is_falling_motion:
                self.current_state = FallState.FALLING
                confidence = 0.7
            elif self.horizontal_count >= self.fall_confirm_frames:
                self.current_state = FallState.FALLEN
                confidence = 0.9
            elif self.horizontal_count >= 3:
                self.current_state = FallState.FALLING
                confidence = 0.6
        else:
            # Not in fall position
            self.horizontal_count = max(0, self.horizontal_count - 1)
            
            if self.current_state in [FallState.FALLEN, FallState.FALLING]:
                # Recovery from fall
                self.standing_count += 1
                if self.standing_count >= self.recovery_frames:
                    self.current_state = FallState.STANDING
                    confidence = 0.8
            else:
                self.current_state = FallState.STANDING
                self.standing_count += 1
                confidence = 0.8
        
        # Create metrics for debugging
        metrics = FallMetrics(
            body_angle=body_angle,
            head_height_ratio=head_height_ratio,
            head_velocity=head_velocity,
            shoulder_hip_ratio=shoulder_hip_ratio,
            confidence=confidence
        )
        
        return self.current_state, metrics
    
    def reset(self) -> None:
        """Reset all state tracking."""
        self.current_state = FallState.UNKNOWN
        self.head_history.clear()
        self.state_history.clear()
        self.horizontal_count = 0
        self.standing_count = 0


def main():
    """
    Test the fall detector with simulated data.
    """
    print("=" * 50)
    print("Fall Detector Test")
    print("=" * 50)
    
    # Create detector
    detector = FallDetector()
    
    # Simulate some scenarios
    print("\n--- Test 1: No landmarks (no person) ---")
    state, metrics = detector.update(None, 480)
    print(f"State: {state.value}, Confidence: {metrics.confidence}")
    
    print("\n--- Test 2: Simulated standing pose ---")
    # Create mock landmarks for standing position
    mock_landmarks = []
    for i in range(33):
        # Default landmark
        lm = Landmark(x=320, y=240, z=0, visibility=0.9, name=f"landmark_{i}")
        mock_landmarks.append(lm)
    
    # Set key landmarks for standing pose
    # Nose at top
    mock_landmarks[0] = Landmark(x=320, y=100, z=0, visibility=0.9, name="nose")
    # Shoulders below nose
    mock_landmarks[11] = Landmark(x=280, y=180, z=0, visibility=0.9, name="left_shoulder")
    mock_landmarks[12] = Landmark(x=360, y=180, z=0, visibility=0.9, name="right_shoulder")
    # Hips below shoulders
    mock_landmarks[23] = Landmark(x=290, y=320, z=0, visibility=0.9, name="left_hip")
    mock_landmarks[24] = Landmark(x=350, y=320, z=0, visibility=0.9, name="right_hip")
    
    for i in range(5):
        state, metrics = detector.update(mock_landmarks, 480)
    print(f"State: {state.value}")
    print(f"Body angle: {metrics.body_angle:.1f}°")
    print(f"Head height ratio: {metrics.head_height_ratio:.2f}")
    print(f"Shoulder-hip ratio: {metrics.shoulder_hip_ratio:.2f}")
    
    print("\n--- Test 3: Simulated fallen pose ---")
    detector.reset()
    
    # Modify landmarks for fallen position (horizontal, head low)
    # Nose at bottom-left
    mock_landmarks[0] = Landmark(x=150, y=400, z=0, visibility=0.9, name="nose")
    # Shoulders horizontal
    mock_landmarks[11] = Landmark(x=200, y=380, z=0, visibility=0.9, name="left_shoulder")
    mock_landmarks[12] = Landmark(x=350, y=390, z=0, visibility=0.9, name="right_shoulder")
    # Hips horizontal, far from shoulders horizontally
    mock_landmarks[23] = Landmark(x=400, y=370, z=0, visibility=0.9, name="left_hip")
    mock_landmarks[24] = Landmark(x=500, y=380, z=0, visibility=0.9, name="right_hip")
    
    # Simulate multiple frames to confirm fall
    for i in range(15):
        state, metrics = detector.update(mock_landmarks, 480)
        if i % 3 == 0:
            print(f"  Frame {i+1}: {state.value} (confidence: {metrics.confidence:.2f})")
    
    print(f"\nFinal State: {state.value}")
    print(f"Body angle: {metrics.body_angle:.1f}°")
    print(f"Head height ratio: {metrics.head_height_ratio:.2f}")
    print(f"Shoulder-hip ratio: {metrics.shoulder_hip_ratio:.2f}")
    
    print("\n" + "=" * 50)
    print("Test completed!")


if __name__ == "__main__":
    main()
