"""
Camera Stream Module
Handles camera capture for Raspberry Pi Camera Module using picamera2.
Optimized for Raspberry Pi 4 with Pi Camera Module v2/v3.

Supports:
- Raspberry Pi Camera Module (v1, v2, v3, HQ)
- picamera2 library (modern, recommended for Pi OS Bullseye/Bookworm)
- rpicam-vid subprocess MJPEG pipe (fallback when picamera2 is unavailable)
- OpenCV V4L2 as final fallback for USB webcams
"""

import cv2
import fcntl
import numpy as np
import os
import select
import time
import logging
import shutil
import subprocess
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

# Detect rpicam-vid (libcamera CLI tool, available on Pi OS Bookworm/Trixie)
RPICAM_AVAILABLE = shutil.which("rpicam-vid") is not None
if not RPICAM_AVAILABLE:
    logger.warning("rpicam-vid not found; Pi Camera via subprocess unavailable")


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
            
            # Try to open the camera; on Pi, prefer V4L2 backend explicitly
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
            
            if not self.cap.isOpened():
                logger.error(f"Could not open camera at index {self.camera_index}")
                print(f"Error: Could not open camera at index {self.camera_index}")
                return False
            
            # Set camera properties
            # Request BGR3 (24-bit BGR) so libcamera V4L2 bridge returns proper BGR data
            # Request YUYV (native Pi Camera format); OpenCV auto-converts it to BGR
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Verify actual settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            # Warm up camera (libcamera V4L2 bridge needs more frames to stabilise)
            logger.info("Warming up camera...")
            for _ in range(15):
                self.cap.read()
                time.sleep(0.05)
            
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
        
        # Reshape flat buffer returned by libcamera V4L2 bridge.
        # The bridge may return shape (1, W*H*3) instead of (H, W, 3).
        if frame is not None and frame.ndim == 2 and frame.shape[0] == 1:
            n = frame.shape[1]
            if n % 3 == 0:
                n_pixels = n // 3
                # Try cap-reported dims first, then common resolutions
                rw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                rh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if rw > 0 and rh > 0 and rw * rh == n_pixels:
                    frame = frame.reshape(rh, rw, 3)
                else:
                    for w, h in [(640, 480), (1280, 720), (320, 240), (800, 600), (1920, 1080), (480, 640), (720, 1280)]:
                        if w * h == n_pixels:
                            frame = frame.reshape(h, w, 3)
                            break
                    else:
                        logger.warning(f"Cannot reshape flat buffer of {n} bytes; dropping frame")
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


class RpicamStream(BaseCameraStream):
    """
    Camera stream using rpicam-vid subprocess with MJPEG output.

    Launches rpicam-vid as a child process, pipes MJPEG frames to stdout,
    and decodes each JPEG frame with OpenCV.  This uses the full libcamera
    ISP pipeline, so white balance and colour correction work correctly —
    unlike accessing /dev/video0 (unicam) directly via V4L2.

    Preferred over OpenCVCameraStream when picamera2 is unavailable.
    """

    _JPEG_SOI = b'\xff\xd8'
    _JPEG_EOI = b'\xff\xd9'
    _CHUNK = 16384  # bytes per os.read(); smaller = less blocking latency

    def __init__(
        self,
        frame_width: int = 1280,
        frame_height: int = 720,
        fps: int = 15,
        auto_reconnect: bool = True,
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        self.auto_reconnect = auto_reconnect

        self._proc: Optional[subprocess.Popen] = None
        self._buf = b''

        self._frame_count = 0
        self._start_time: float = 0
        self._last_frame_time: float = 0
        self._actual_fps: float = 0.0
        self._consecutive_failures: int = 0
        self._max_consecutive_failures: int = 10

    def start(self) -> bool:
        """Launch rpicam-vid subprocess."""
        self.stop()
        cmd = [
            'rpicam-vid',
            '--nopreview',
            '--codec', 'mjpeg',
            '-t', '0',                       # run indefinitely
            '--width', str(self.frame_width),
            '--height', str(self.frame_height),
            '--framerate', str(self.fps),
            '-o', '-',                        # output to stdout
        ]
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
            self._buf = b''
            self._frame_count = 0
            self._start_time = time.time()
            self._last_frame_time = 0
            self._actual_fps = 0.0
            self._consecutive_failures = 0

            # Warmup: read until we get the first real frame
            logger.info('Waiting for first rpicam-vid frame...')
            for _ in range(200):          # up to ~200 chunks ≈ a couple of seconds
                frame = self._read_frame()
                if frame is not None:
                    # Switch pipe to non-blocking so get_frame() can drain
                    # stale buffered frames without hanging.
                    fd = self._proc.stdout.fileno()
                    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
                    fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

                    logger.info(f'rpicam-vid ready: {frame.shape}')
                    print(f'Camera opened successfully (rpicam-vid MJPEG)!')
                    print(f'  Resolution: {self.frame_width}x{self.frame_height}')
                    print(f'  Target FPS: {self.fps}')
                    return True

            logger.error('rpicam-vid failed to produce a frame during warmup')
            self.stop()
            return False
        except Exception as e:
            logger.error(f'Failed to start rpicam-vid: {e}')
            print(f'Error: could not start rpicam-vid: {e}')
            return False

    def _try_read(self) -> bool:
        """Read available data from the pipe (non-blocking).

        Returns True if data was read, False on EOF/error/would-block.
        """
        try:
            chunk = os.read(self._proc.stdout.fileno(), self._CHUNK)
        except BlockingIOError:
            return False   # nothing available right now
        except Exception:
            return False
        if not chunk:
            return False
        self._buf += chunk
        return True

    def _extract_frame(self) -> Optional[np.ndarray]:
        """Extract one complete JPEG from the buffer, or None."""
        soi = self._buf.find(self._JPEG_SOI)
        if soi < 0:
            self._buf = b''  # no start marker — discard prefix
            return None
        eoi = self._buf.find(self._JPEG_EOI, soi + 2)
        if eoi < 0:
            return None  # incomplete frame, need more data
        eoi += 2
        jpeg = self._buf[soi:eoi]
        self._buf = self._buf[eoi:]
        return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)

    def _read_frame(self) -> Optional[np.ndarray]:
        """Wait for one complete JPEG frame from the pipe and return it.

        Uses select() to block-wait for data on the non-blocking fd.
        """
        if self._proc is None or self._proc.poll() is not None:
            return None
        fd = self._proc.stdout.fileno()
        while True:
            frame = self._extract_frame()
            if frame is not None:
                return frame
            # Wait up to 2s for data to arrive
            readable, _, _ = select.select([fd], [], [], 2.0)
            if not readable:
                return None  # timeout
            if not self._try_read():
                return None

    def get_frame(self) -> Optional[np.ndarray]:
        """Return the *latest* available frame, draining any stale ones.

        When ML inference takes longer than the camera interval, multiple
        JPEG frames can accumulate in the pipe. We drain all of them and
        return only the most recent one so the display stays current.
        """
        if self._proc is None or self._proc.poll() is not None:
            if self.auto_reconnect:
                logger.warning('rpicam-vid process ended, restarting...')
                if self.start():
                    return self.get_frame()
            return None

        # 1. Read at least one frame (blocking)
        latest = self._read_frame()
        if latest is None:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._max_consecutive_failures and self.auto_reconnect:
                logger.warning('Too many rpicam-vid failures, restarting...')
                self._consecutive_failures = 0
                self.start()
            return None

        # 2. Drain any additional buffered frames to stay current
        while True:
            # Non-blocking: read whatever data is already available
            try:
                chunk = os.read(self._proc.stdout.fileno(), self._CHUNK)
            except (BlockingIOError, OSError):
                break
            if not chunk:
                break
            self._buf += chunk
            # Extract as many frames as possible from the buffer
            while True:
                frame = self._extract_frame()
                if frame is None:
                    break
                latest = frame  # keep overwriting with newer frame

        self._consecutive_failures = 0
        self._frame_count += 1
        now = time.time()
        if self._last_frame_time > 0:
            fi = now - self._last_frame_time
            self._actual_fps = 0.9 * self._actual_fps + 0.1 * (1.0 / fi if fi > 0 else 0)
        self._last_frame_time = now
        return latest

    def stop(self) -> None:
        """Terminate the rpicam-vid subprocess."""
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=3)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None
            self._buf = b''
            logger.info('rpicam-vid process stopped')
            print('Camera released')

    def is_opened(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def get_fps(self) -> float:
        return self._actual_fps

    def get_stats(self) -> dict:
        elapsed = time.time() - self._start_time if self._start_time > 0 else 0
        return {
            'frame_count': self._frame_count,
            'elapsed_time': elapsed,
            'actual_fps': self._actual_fps,
            'resolution': f'{self.frame_width}x{self.frame_height}',
            'camera_type': 'RpicamMJPEG',
        }

    def get_frame_dimensions(self) -> Tuple[int, int]:
        return self.frame_width, self.frame_height

    def frames(self):
        """Generator yielding frames continuously."""
        while self.is_opened():
            frame = self.get_frame()
            if frame is not None:
                yield frame

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
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

    Priority:
    1. PiCameraStream (picamera2) — best quality, needs system picamera2 package
    2. RpicamStream (rpicam-vid subprocess MJPEG) — full ISP pipeline, proper color
    3. OpenCVCameraStream — last resort for USB webcams / non-Pi systems

    Args:
        camera_index: Camera device index (only used for OpenCV fallback)
        frame_width: Width of captured frames
        frame_height: Height of captured frames
        fps: Target frames per second
        auto_reconnect: Whether to auto-reconnect on camera failure
        force_opencv: Force using OpenCV even if picamera2/rpicam-vid is available

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
    elif RPICAM_AVAILABLE and not force_opencv:
        logger.info("Using Pi Camera via rpicam-vid subprocess (ISP pipeline)")
        print("Detected rpicam-vid - using Pi Camera with full ISP colour processing")
        return RpicamStream(
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
