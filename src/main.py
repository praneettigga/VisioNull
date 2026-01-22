"""
Fall Detection System - Main Application
Real-time fall detection using Raspberry Pi camera, MediaPipe Pose, and OpenCV.

Usage:
    python -m src.main

Press 'q' to quit, 'd' to toggle debug info, 'r' to reset detector.
"""

import cv2
import time
from typing import Optional

# Import project modules
try:
    from src.camera_stream import CameraStream
    from src.pose_estimator import PoseEstimator
    from src.fall_detector import FallDetector, FallState
except ImportError:
    from camera_stream import CameraStream
    from pose_estimator import PoseEstimator
    from fall_detector import FallDetector, FallState


class FallDetectionApp:
    """
    Main application class that integrates camera, pose estimation, and fall detection.
    """
    
    def __init__(
        self,
        camera_index: int = 0,
        frame_width: int = 640,
        frame_height: int = 480,
        show_debug: bool = True
    ):
        """
        Initialize the fall detection application.
        
        Args:
            camera_index: Camera device index
            frame_width: Desired frame width
            frame_height: Desired frame height
            show_debug: Whether to show debug metrics overlay
        """
        self.show_debug = show_debug
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Initialize components
        print("=" * 60)
        print("  FALL DETECTION SYSTEM")
        print("  Real-time fall detection using MediaPipe Pose")
        print("=" * 60)
        print()
        
        print("Initializing components...")
        
        # Camera
        self.camera = CameraStream(
            camera_index=camera_index,
            frame_width=frame_width,
            frame_height=frame_height
        )
        
        # Pose estimator (lightweight model for Raspberry Pi)
        self.pose_estimator = PoseEstimator(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Fall detector
        self.fall_detector = FallDetector(
            history_size=15,
            fall_head_threshold=0.65,
            horizontal_ratio_threshold=0.8,
            fall_confirm_frames=8
        )
        
        # FPS tracking
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0.0
        
    def _update_fps(self) -> None:
        """Update FPS calculation."""
        self.fps_frame_count += 1
        elapsed = time.time() - self.fps_start_time
        
        if elapsed >= 1.0:  # Update every second
            self.current_fps = self.fps_frame_count / elapsed
            self.fps_start_time = time.time()
            self.fps_frame_count = 0
    
    def _draw_status_overlay(
        self,
        frame,
        state: FallState,
        confidence: float
    ):
        """
        Draw the main status overlay on the frame.
        """
        height, width = frame.shape[:2]
        
        # Determine colors and text based on state
        if state == FallState.FALLEN:
            status_text = "FALL DETECTED"
            bg_color = (0, 0, 180)  # Red
            text_color = (255, 255, 255)
            # Draw red border around frame
            cv2.rectangle(frame, (0, 0), (width-1, height-1), (0, 0, 255), 8)
        elif state == FallState.FALLING:
            status_text = "FALLING..."
            bg_color = (0, 140, 255)  # Orange
            text_color = (255, 255, 255)
            cv2.rectangle(frame, (0, 0), (width-1, height-1), (0, 165, 255), 4)
        elif state == FallState.STANDING:
            status_text = "STANDING"
            bg_color = (0, 120, 0)  # Green
            text_color = (255, 255, 255)
        else:
            status_text = "DETECTING..."
            bg_color = (100, 100, 100)  # Gray
            text_color = (255, 255, 255)
        
        # Draw status banner at top
        banner_height = 50
        cv2.rectangle(frame, (0, 0), (width, banner_height), bg_color, -1)
        
        # Status text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 3
        
        text_size = cv2.getTextSize(status_text, font, font_scale, thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (banner_height + text_size[1]) // 2
        
        cv2.putText(
            frame, status_text,
            (text_x, text_y),
            font, font_scale, text_color, thickness
        )
        
        # Confidence indicator
        conf_text = f"Conf: {confidence:.0%}"
        cv2.putText(
            frame, conf_text,
            (width - 120, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1
        )
        
        # FPS indicator
        fps_text = f"FPS: {self.current_fps:.1f}"
        cv2.putText(
            frame, fps_text,
            (10, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1
        )
    
    def _draw_debug_overlay(self, frame, metrics):
        """
        Draw debug metrics on the frame.
        """
        height = frame.shape[0]
        
        # Debug info box at bottom
        debug_y_start = height - 80
        cv2.rectangle(frame, (0, debug_y_start), (300, height), (0, 0, 0), -1)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        color = (255, 255, 255)
        
        lines = [
            f"Body Angle: {metrics.body_angle:.1f} deg",
            f"Head Height: {metrics.head_height_ratio:.2f}",
            f"Head Velocity: {metrics.head_velocity:.1f} px/f",
            f"SH Ratio: {metrics.shoulder_hip_ratio:.2f}"
        ]
        
        for i, line in enumerate(lines):
            y = debug_y_start + 18 + i * 16
            cv2.putText(frame, line, (10, y), font, font_scale, color, 1)
    
    def _draw_controls_overlay(self, frame):
        """Draw control instructions."""
        height = frame.shape[0]
        
        controls = "Press: [Q]uit | [D]ebug | [R]eset"
        cv2.putText(
            frame, controls,
            (10, height - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1
        )
    
    def run(self) -> None:
        """
        Main application loop.
        """
        print()
        print("Starting camera...")
        
        if not self.camera.start():
            print("ERROR: Failed to start camera!")
            print("Please check camera connection and settings.")
            return
        
        print()
        print("Controls:")
        print("  [Q] - Quit")
        print("  [D] - Toggle debug info")
        print("  [R] - Reset fall detector")
        print()
        print("Running fall detection... Press 'q' to quit.")
        print()
        
        window_name = "Fall Detection System"
        
        try:
            while True:
                # Capture frame
                frame = self.camera.get_frame()
                
                if frame is None:
                    print("Warning: Lost camera feed")
                    break
                
                frame_height = frame.shape[0]
                
                # Get pose landmarks
                landmarks = self.pose_estimator.get_landmarks(frame)
                
                # Update fall detector
                state, metrics = self.fall_detector.update(landmarks, frame_height)
                
                # Draw pose skeleton
                if landmarks:
                    frame = self.pose_estimator.draw_landmarks(frame, landmarks)
                    
                    # Draw bounding box
                    bbox = self.pose_estimator.get_bounding_box(landmarks)
                    
                    # Color based on state
                    if state == FallState.FALLEN:
                        box_color = (0, 0, 255)  # Red
                    elif state == FallState.FALLING:
                        box_color = (0, 165, 255)  # Orange
                    else:
                        box_color = (0, 255, 0)  # Green
                    
                    cv2.rectangle(
                        frame,
                        (bbox[0], bbox[1]),
                        (bbox[2], bbox[3]),
                        box_color, 2
                    )
                
                # Draw overlays
                self._draw_status_overlay(frame, state, metrics.confidence)
                
                if self.show_debug:
                    self._draw_debug_overlay(frame, metrics)
                
                self._draw_controls_overlay(frame)
                
                # Update FPS
                self._update_fps()
                
                # Display frame
                cv2.imshow(window_name, frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord('d'):
                    self.show_debug = not self.show_debug
                    print(f"Debug overlay: {'ON' if self.show_debug else 'OFF'}")
                elif key == ord('r'):
                    self.fall_detector.reset()
                    print("Fall detector reset")
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        """Clean up resources."""
        print("Cleaning up...")
        self.pose_estimator.close()
        self.camera.stop()
        cv2.destroyAllWindows()
        print("Application closed")


def main():
    """Entry point for the fall detection application."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Real-time fall detection using MediaPipe Pose"
    )
    parser.add_argument(
        "--camera", "-c",
        type=int,
        default=0,
        help="Camera index (default: 0)"
    )
    parser.add_argument(
        "--width", "-W",
        type=int,
        default=640,
        help="Frame width (default: 640)"
    )
    parser.add_argument(
        "--height", "-H",
        type=int,
        default=480,
        help="Frame height (default: 480)"
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Hide debug metrics overlay"
    )
    
    args = parser.parse_args()
    
    # Create and run application
    app = FallDetectionApp(
        camera_index=args.camera,
        frame_width=args.width,
        frame_height=args.height,
        show_debug=not args.no_debug
    )
    
    app.run()


if __name__ == "__main__":
    main()
