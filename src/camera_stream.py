"""
Camera Stream Module
Handles camera capture for Raspberry Pi using OpenCV.
"""

import cv2
import numpy as np
from typing import Generator, Optional, Tuple


class CameraStream:
    """
    A class to handle camera capture from Raspberry Pi camera or USB webcam.
    
    Attributes:
        camera_index: The camera device index (default 0 for primary camera)
        frame_width: Desired frame width
        frame_height: Desired frame height
        fps: Desired frames per second
    """
    
    def __init__(
        self,
        camera_index: int = 0,
        frame_width: int = 640,
        frame_height: int = 480,
        fps: int = 30
    ):
        """
        Initialize the camera stream.
        
        Args:
            camera_index: Camera device index (0 for default camera)
            frame_width: Width of captured frames
            frame_height: Height of captured frames
            fps: Target frames per second
        """
        self.camera_index = camera_index
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        self.cap: Optional[cv2.VideoCapture] = None
        
    def start(self) -> bool:
        """
        Start the camera capture.
        
        Returns:
            True if camera opened successfully, False otherwise
        """
        # Try to open the camera
        # On Raspberry Pi, index 0 usually works for the Pi Camera
        # For libcamera-based systems, you might need to use a different backend
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open camera at index {self.camera_index}")
            return False
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Verify actual settings (camera may not support requested values)
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Camera opened successfully!")
        print(f"  Resolution: {int(actual_width)}x{int(actual_height)}")
        print(f"  FPS: {actual_fps}")
        
        return True
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame from the camera.
        
        Returns:
            Frame as numpy array in BGR format, or None if capture failed
        """
        if self.cap is None or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        
        if not ret:
            print("Warning: Failed to capture frame")
            return None
        
        return frame
    
    def frames(self) -> Generator[np.ndarray, None, None]:
        """
        Generator that yields frames continuously.
        
        Yields:
            Frames as numpy arrays in BGR format
        """
        while True:
            frame = self.get_frame()
            if frame is None:
                break
            yield frame
    
    def get_frame_dimensions(self) -> Tuple[int, int]:
        """
        Get the actual frame dimensions.
        
        Returns:
            Tuple of (width, height)
        """
        if self.cap is None:
            return (self.frame_width, self.frame_height)
        
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)
    
    def stop(self) -> None:
        """Release the camera resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            print("Camera released")
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False


def main():
    """
    Test the camera stream by displaying raw video feed.
    Press 'q' to quit.
    """
    print("=" * 50)
    print("Camera Stream Test")
    print("=" * 50)
    print("Press 'q' to quit")
    print()
    
    # Create and start camera
    camera = CameraStream(camera_index=0, frame_width=640, frame_height=480)
    
    if not camera.start():
        print("Failed to start camera. Please check:")
        print("  1. Camera is properly connected")
        print("  2. Camera interface is enabled (raspi-config)")
        print("  3. No other application is using the camera")
        return
    
    try:
        frame_count = 0
        while True:
            frame = camera.get_frame()
            
            if frame is None:
                print("Lost camera connection")
                break
            
            # Add frame info overlay
            frame_count += 1
            info_text = f"Frame: {frame_count} | Size: {frame.shape[1]}x{frame.shape[0]}"
            cv2.putText(
                frame,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Add instructions
            cv2.putText(
                frame,
                "Press 'q' to quit",
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )
            
            # Display the frame
            cv2.imshow("Camera Stream Test", frame)
            
            # Check for quit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        camera.stop()
        cv2.destroyAllWindows()
        print("Test completed")


if __name__ == "__main__":
    main()
