"""
Camera Stream Module
Handles camera capture for Raspberry Pi Camera Module using picamera2.
Optimized for Raspberry Pi 4 with Pi Camera Module v2/v3.

Supports:
- Raspberry Pi Camera Module (v1, v2, v3, HQ)
- picamera2 library (modern, recommended for Pi OS Bullseye/Bookworm)
- Fallback to OpenCV for USB webcams or testing on non-Pi systems
"""

import cv2
import numpy as np
import time
import logging
from typing import Generator, Optional, Tuple
from abc import ABC, abstractmethod

# Setup logging
logger = logging.getLogger(__name__)

# Try to import picamera2 (only available on Raspberry Pi)
PICAMERA2_AVAILABLE = False
try:
    from picamera2 import Picamera2
    from libcamera import controls
    PICAMERA2_AVAILABLE = True
    logger.info("picamera2 library available")
except ImportError:
    logger.warning("picamera2 not available, will use OpenCV fallback")


class BaseCameraStream(ABC):
    """Abstract base class for camera streams."""
    
    @abstractmethod
    def start(self) -> bool:
        """Start the camera capture."""
        pass
    
    @abstractmethod
    def get_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame."""
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """Stop the camera and release resources."""
        pass
    
    @abstractmethod
    def is_opened(self) -> bool:
        """Check if camera is currently open."""
        pass


class PiCameraStream(BaseCameraStream):
    """
    Camera stream using picamera2 for Raspberry Pi Camera Module.
    
    This is the recommended approach for Pi Camera on Raspberry Pi OS
    Bullseye/Bookworm which use libcamera.
    """
    
    def __init__(
        self,
        frame_width: int = 1280,
        frame_height: int = 720,
        fps: int = 15,
        auto_reconnect: bool = True
    ):
        """
        Initialize the Pi Camera stream.
        
        Args:
            frame_width: Width of captured frames (default 1280 for better accuracy)
            frame_height: Height of captured frames (default 720 for better accuracy)
            fps: Target frames per second (default 15, suitable for Pi 4 with ML)
            auto_reconnect: Whether to auto-reconnect on camera failure
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        self.auto_reconnect = auto_reconnect
        
        self.camera: Optional[Picamera2] = None
        self._is_started = False
        
        # Statistics
        self._frame_count = 0
        self._start_time = 0
        self._last_frame_time = 0
        self._actual_fps = 0.0
        self._consecutive_failures = 0
        self._max_consecutive_failures = 10
    
    def start(self) -> bool:
        """
        Start the Pi Camera capture.
        
        Returns:
            True if camera started successfully, False otherwise
        """
        try:
            logger.info("Initializing Pi Camera with picamera2...")
            
            # Create Picamera2 instance
            self.camera = Picamera2()
            
            # Configure camera for video capture
            # Use RGB888 format for direct use with OpenCV/MediaPipe
            config = self.camera.create_video_configuration(
                main={
                    "size": (self.frame_width, self.frame_height),
                    "format": "RGB888"
                },
                controls={
                    "FrameDurationLimits": (int(1000000 / self.fps), int(1000000 / self.fps))
                },
                buffer_count=2  # Minimal buffer for low latency
            )
            
            self.camera.configure(config)
            
            # Set additional controls for better image quality
            self.camera.set_controls({
                "AeEnable": True,  # Auto exposure
                "AwbEnable": True,  # Auto white balance
            })
            
            # Start the camera
            self.camera.start()
            self._is_started = True
            
            # Warm up camera (first few frames may be dark)
            logger.info("Warming up Pi Camera...")
            time.sleep(1.0)  # Give camera time to adjust exposure
            for _ in range(5):
                self.camera.capture_array()
                time.sleep(0.1)
            
            # Get actual configuration
            actual_config = self.camera.camera_configuration()
            actual_size = actual_config['main']['size']
            
            print(f"Pi Camera opened successfully!")
            print(f"  Resolution: {actual_size[0]}x{actual_size[1]}")
            print(f"  Target FPS: {self.fps}")
            logger.info(f"Pi Camera ready: {actual_size[0]}x{actual_size[1]} @ {self.fps} FPS")
            
            # Reset statistics
            self._frame_count = 0
            self._start_time = time.time()
            self._consecutive_failures = 0
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start Pi Camera: {e}")
            print(f"Error: Could not start Pi Camera: {e}")
            self._is_started = False
            return False
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame from the Pi Camera.
        
        Returns:
            Frame as numpy array in BGR format (OpenCV compatible), or None if capture failed
        """
        if not self._is_started or self.camera is None:
            if self.auto_reconnect:
                logger.warning("Camera not started, attempting restart...")
                if self.start():
                    return self.get_frame()
            return None
        
        try:
            # Capture frame (returns RGB format)
            frame_rgb = self.camera.capture_array()
            
            if frame_rgb is None:
                raise RuntimeError("Captured frame is None")
            
            # Convert RGB to BGR for OpenCV compatibility
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            # Reset failure counter on success
            self._consecutive_failures = 0
            
            # Update statistics
            self._frame_count += 1
            current_time = time.time()
            
            if self._last_frame_time > 0:
                frame_interval = current_time - self._last_frame_time
                instant_fps = 1.0 / frame_interval if frame_interval > 0 else 0
                self._actual_fps = 0.9 * self._actual_fps + 0.1 * instant_fps
            
            self._last_frame_time = current_time
            
            return frame_bgr
            
        except Exception as e:
            self._consecutive_failures += 1
            logger.warning(f"Failed to capture frame: {e} ({self._consecutive_failures}/{self._max_consecutive_failures})")
            
            if self._consecutive_failures >= self._max_consecutive_failures:
                if self.auto_reconnect:
                    logger.warning("Too many failures, attempting camera restart...")
                    self.stop()
                    time.sleep(1.0)
                    if self.start():
                        return self.get_frame()
            return None
    
    def stop(self) -> None:
        """Stop the Pi Camera and release resources."""
        if self.camera is not None:
            try:
                self.camera.stop()
                self.camera.close()
            except Exception as e:
                logger.warning(f"Error stopping camera: {e}")
            finally:
                self.camera = None
                self._is_started = False
                logger.info("Pi Camera released")
                print("Pi Camera released")
    
    def is_opened(self) -> bool:
        """Check if camera is currently open."""
        return self._is_started and self.camera is not None
    
    def get_fps(self) -> float:
        """Get the actual measured FPS."""
        return self._actual_fps
    
    def get_frame_count(self) -> int:
        """Get total number of frames captured."""
        return self._frame_count
    
    def get_uptime(self) -> float:
        """Get camera uptime in seconds."""
        if self._start_time == 0:
            return 0
        return time.time() - self._start_time
    
    def get_stats(self) -> dict:
        """Get camera statistics."""
        return {
            'frame_count': self._frame_count,
            'uptime': self.get_uptime(),
            'actual_fps': round(self._actual_fps, 1),
            'target_fps': self.fps,
            'resolution': f"{self.frame_width}x{self.frame_height}",
            'camera_type': 'PiCamera (picamera2)'
        }
    
    def get_frame_dimensions(self) -> Tuple[int, int]:
        """Get the actual frame dimensions."""
        return (self.frame_width, self.frame_height)
    
    def frames(self) -> Generator[np.ndarray, None, None]:
        """Generator that yields frames continuously."""
        while True:
            frame = self.get_frame()
            if frame is None:
                if self.auto_reconnect:
                    time.sleep(0.5)
                    continue
                break
            yield frame
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False


class OpenCVCameraStream(BaseCameraStream):
    """
    Fallback camera stream using OpenCV.
    Used for USB webcams or testing on non-Pi systems.
    """
    
    def __init__(
        self,
        camera_index: int = 0,
        frame_width: int = 1280,
        frame_height: int = 720,
        fps: int = 15,
        auto_reconnect: bool = True
    ):
        """
        Initialize the OpenCV camera stream.
        
        Args:
            camera_index: Camera device index (0 for default camera)
            frame_width: Width of captured frames
            frame_height: Height of captured frames
            fps: Target frames per second
            auto_reconnect: Whether to auto-reconnect on camera failure
        """
        self.camera_index = camera_index
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        self.auto_reconnect = auto_reconnect
        
        self.cap: Optional[cv2.VideoCapture] = None
        
        # Statistics
        self._frame_count = 0
        self._start_time = 0
        self._last_frame_time = 0
        self._actual_fps = 0.0
        self._consecutive_failures = 0
        self._max_consecutive_failures = 10
    
    def start(self) -> bool:
        """Start the camera capture."""
        try:
            # Close existing camera if any
            if self.cap is not None:
                self.cap.release()
            
            logger.info(f"Opening camera with OpenCV (index {self.camera_index})...")
            
            # Try to open the camera
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                logger.error(f"Could not open camera at index {self.camera_index}")
                print(f"Error: Could not open camera at index {self.camera_index}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Verify actual settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            # Warm up camera
            logger.info("Warming up camera...")
            for _ in range(5):
                self.cap.read()
                time.sleep(0.1)
            
            print(f"Camera opened successfully (OpenCV fallback)!")
            print(f"  Resolution: {actual_width}x{actual_height}")
            print(f"  Target FPS: {actual_fps}")
            logger.info(f"Camera ready: {actual_width}x{actual_height} @ {actual_fps} FPS")
            
            # Reset statistics
            self._frame_count = 0
            self._start_time = time.time()
            self._consecutive_failures = 0
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start camera: {e}")
            print(f"Error: Could not start camera: {e}")
            return False
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame from the camera."""
        if self.cap is None or not self.cap.isOpened():
            if self.auto_reconnect:
                logger.warning("Camera not open, attempting reconnect...")
                if self.start():
                    return self.get_frame()
            return None
        
        ret, frame = self.cap.read()
        
        if not ret or frame is None:
            self._consecutive_failures += 1
            logger.warning(f"Failed to capture frame ({self._consecutive_failures}/{self._max_consecutive_failures})")
            
            if self._consecutive_failures >= self._max_consecutive_failures:
                if self.auto_reconnect:
                    logger.warning("Too many failures, attempting camera reconnect...")
                    time.sleep(1.0)
                    if self.start():
                        return self.get_frame()
            return None
        
        # Reset failure counter on success
        self._consecutive_failures = 0
        
        # Update statistics
        self._frame_count += 1
        current_time = time.time()
        
        if self._last_frame_time > 0:
            frame_interval = current_time - self._last_frame_time
            instant_fps = 1.0 / frame_interval if frame_interval > 0 else 0
            self._actual_fps = 0.9 * self._actual_fps + 0.1 * instant_fps
        
        self._last_frame_time = current_time
        
        return frame
    
    def stop(self) -> None:
        """Release the camera resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("Camera released")
            print("Camera released")
    
    def is_opened(self) -> bool:
        """Check if camera is currently open."""
        return self.cap is not None and self.cap.isOpened()
    
    def get_fps(self) -> float:
        """Get the actual measured FPS."""
        return self._actual_fps
    
    def get_frame_count(self) -> int:
        """Get total number of frames captured."""
        return self._frame_count
    
    def get_uptime(self) -> float:
        """Get camera uptime in seconds."""
        if self._start_time == 0:
            return 0
        return time.time() - self._start_time
    
    def get_stats(self) -> dict:
        """Get camera statistics."""
        return {
            'frame_count': self._frame_count,
            'uptime': self.get_uptime(),
            'actual_fps': round(self._actual_fps, 1),
            'target_fps': self.fps,
            'resolution': f"{self.frame_width}x{self.frame_height}",
            'camera_type': 'OpenCV',
            'camera_index': self.camera_index
        }
    
    def get_frame_dimensions(self) -> Tuple[int, int]:
        """Get the actual frame dimensions."""
        if self.cap is None:
            return (self.frame_width, self.frame_height)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)
    
    def frames(self) -> Generator[np.ndarray, None, None]:
        """Generator that yields frames continuously."""
        while True:
            frame = self.get_frame()
            if frame is None:
                if self.auto_reconnect:
                    time.sleep(0.5)
                    continue
                break
            yield frame
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False


def CameraStream(
    camera_index: int = 0,
    frame_width: int = 1280,
    frame_height: int = 720,
    fps: int = 15,
    auto_reconnect: bool = True,
    force_opencv: bool = False
) -> BaseCameraStream:
    """
    Factory function to create the appropriate camera stream.
    
    Automatically selects:
    - PiCameraStream if picamera2 is available (Raspberry Pi with Pi Camera)
    - OpenCVCameraStream as fallback (USB webcam or non-Pi systems)
    
    Args:
        camera_index: Camera device index (only used for OpenCV fallback)
        frame_width: Width of captured frames
        frame_height: Height of captured frames
        fps: Target frames per second
        auto_reconnect: Whether to auto-reconnect on camera failure
        force_opencv: Force using OpenCV even if picamera2 is available
        
    Returns:
        Appropriate camera stream instance
    """
    if PICAMERA2_AVAILABLE and not force_opencv:
        logger.info("Using Pi Camera with picamera2")
        print("Detected Pi Camera - using picamera2")
        return PiCameraStream(
            frame_width=frame_width,
            frame_height=frame_height,
            fps=fps,
            auto_reconnect=auto_reconnect
        )
    else:
        logger.info("Using OpenCV camera (fallback)")
        print("Using OpenCV camera (Pi Camera not detected or force_opencv=True)")
        return OpenCVCameraStream(
            camera_index=camera_index,
            frame_width=frame_width,
            frame_height=frame_height,
            fps=fps,
            auto_reconnect=auto_reconnect
        )


def main():
    """
    Test the camera stream by displaying raw video feed.
    Press 'q' to quit.
    """
    print("=" * 50)
    print("Camera Stream Test")
    print("=" * 50)
    print(f"picamera2 available: {PICAMERA2_AVAILABLE}")
    print("Press 'q' to quit")
    print()
    
    # Create camera (auto-selects Pi Camera or OpenCV)
    camera = CameraStream(
        frame_width=1280,
        frame_height=720,
        fps=15
    )
    
    if not camera.start():
        print("Failed to start camera. Please check:")
        print("  1. Camera ribbon cable is properly connected")
        print("  2. Camera is detected: rpicam-hello --list-cameras")
        print("  3. No other application is using the camera")
        print("  4. For Pi Camera: sudo apt install python3-picamera2")
        print("  5. If camera not detected, try adding dtoverlay=imx219")
        print("     to /boot/firmware/config.txt and reboot")
        return
    
    try:
        while True:
            frame = camera.get_frame()
            
            if frame is None:
                print("Lost camera connection")
                break
            
            # Add frame info overlay
            stats = camera.get_stats()
            info_text = f"FPS: {stats['actual_fps']:.1f} | Frames: {stats['frame_count']} | {stats['resolution']}"
            cv2.putText(
                frame,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            cv2.putText(
                frame,
                f"Camera: {stats.get('camera_type', 'Unknown')}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )
            
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
