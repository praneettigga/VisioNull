#!/usr/bin/env python3
"""
Fall Detection System - Raspberry Pi Headless Main Application
Real-time fall detection using Raspberry Pi Camera, MediaPipe Pose, and HTTP webhook notifications.

This is the production entry point for running on Raspberry Pi without a display.

Usage:
    python -m src.main_pi
    
    Or run directly:
    python src/main_pi.py

Environment:
    Set VISIONULL_WEBHOOK_URL to override the webhook URL from config.
    Set VISIONULL_DEVICE_NAME to override the device name from config.
"""

import os
import sys
import time
import signal
import logging
from datetime import datetime
from typing import Optional

# Setup logging before imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

# Import project modules
try:
    from src.camera_stream import CameraStream
    from src.pose_estimator import PoseEstimator
    from src.fall_detector import FallDetector, FallState
    from src.notifier import FallNotifier, get_notifier
    from src import config
except ImportError:
    from camera_stream import CameraStream
    from pose_estimator import PoseEstimator
    from fall_detector import FallDetector, FallState
    from notifier import FallNotifier, get_notifier
    import config


class FallDetectionPi:
    """
    Headless fall detection application for Raspberry Pi.
    
    Features:
    - No display required (headless mode)
    - HTTP webhook notifications on fall detection
    - Post-fall validation to reduce false positives
    - Auto-recovery on camera failure
    - Graceful shutdown handling
    - System monitoring and logging
    """
    
    def __init__(self):
        """Initialize the fall detection system."""
        self.running = False
        self.camera = None
        self.pose_estimator = None
        self.fall_detector = None
        self.notifier = None
        
        # Statistics
        self.start_time = None
        self.frames_processed = 0
        self.falls_detected = 0
        self.notifications_sent = 0
        
        # Track if we've already notified for current fall
        self._fall_notified = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.running = False
    
    def initialize(self) -> bool:
        """
        Initialize all components.
        
        Returns:
            True if all components initialized successfully
        """
        print()
        print("=" * 60)
        print("  VISIONULL - FALL DETECTION SYSTEM")
        print("  Raspberry Pi Headless Mode")
        print("=" * 60)
        print()
        print(config.get_config_summary())
        print()
        
        logger.info("Initializing fall detection system...")
        
        try:
            # Initialize camera
            logger.info("Initializing camera...")
            self.camera = CameraStream(
                frame_width=config.FRAME_WIDTH,
                frame_height=config.FRAME_HEIGHT,
                fps=config.TARGET_FPS,
                auto_reconnect=True
            )
            
            if not self.camera.start():
                logger.error("Failed to start camera")
                return False
            
            # Initialize pose estimator
            logger.info("Initializing pose estimator...")
            self.pose_estimator = PoseEstimator(
                model_complexity=0,  # Lite model for Pi
                min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
            )
            
            # Initialize fall detector with post-validation
            logger.info("Initializing fall detector...")
            self.fall_detector = FallDetector(
                history_size=15,
                fall_head_threshold=config.FALL_HEAD_THRESHOLD,
                horizontal_ratio_threshold=config.HORIZONTAL_RATIO_THRESHOLD,
                fall_confirm_frames=config.FALL_CONFIRM_FRAMES,
                head_velocity_threshold=config.HEAD_VELOCITY_THRESHOLD,
                recovery_frames=config.RECOVERY_FRAMES,
                post_fall_validation_seconds=config.POST_FALL_VALIDATION_SECONDS,
                confidence_threshold=config.FALL_CONFIDENCE_THRESHOLD
            )
            
            # Initialize notifier
            logger.info("Initializing notification system...")
            webhook_url = os.environ.get('VISIONULL_WEBHOOK_URL', config.WEBHOOK_URL)
            device_name = os.environ.get('VISIONULL_DEVICE_NAME', config.DEVICE_NAME)
            
            self.notifier = FallNotifier(
                webhook_url=webhook_url,
                device_name=device_name,
                device_location=config.DEVICE_LOCATION,
                cooldown_seconds=config.NOTIFICATION_COOLDOWN_SECONDS,
                timeout=config.WEBHOOK_TIMEOUT,
                enable_queue=config.ENABLE_OFFLINE_QUEUE
            )
            
            logger.info("All components initialized successfully!")
            print()
            print("System ready. Monitoring for falls...")
            print("Press Ctrl+C to stop.")
            print()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize: {e}", exc_info=True)
            return False
    
    def run(self) -> None:
        """Main detection loop."""
        if not self.initialize():
            logger.error("Initialization failed, exiting")
            return
        
        self.running = True
        self.start_time = time.time()
        last_status_time = time.time()
        status_interval = 60  # Print status every 60 seconds
        
        logger.info("Starting detection loop...")
        
        try:
            while self.running:
                # Capture frame
                frame = self.camera.get_frame()
                
                if frame is None:
                    logger.warning("Failed to capture frame, retrying...")
                    time.sleep(0.5)
                    continue
                
                frame_height = frame.shape[0]
                
                # Process pose estimation
                landmarks = self.pose_estimator.process_frame(frame)
                
                # Update fall detection
                state, metrics = self.fall_detector.update(landmarks, frame_height)
                
                self.frames_processed += 1
                
                # Handle fall detection
                self._handle_fall_state(state, metrics)
                
                # Periodic status logging
                current_time = time.time()
                if current_time - last_status_time >= status_interval:
                    self._log_status()
                    last_status_time = current_time
                
        except Exception as e:
            logger.error(f"Error in detection loop: {e}", exc_info=True)
        finally:
            self.shutdown()
    
    def _handle_fall_state(self, state: FallState, metrics) -> None:
        """Handle fall detection state changes."""
        
        if state == FallState.VALIDATING:
            # Log validation progress
            if metrics.validation_time > 0:
                logger.debug(f"Validating fall... {metrics.validation_time:.1f}s remaining")
        
        elif state == FallState.FALLEN:
            # Check if fall is validated and we haven't notified yet
            if self.fall_detector.is_fall_validated() and not self._fall_notified:
                # Check confidence threshold
                if metrics.confidence >= self.fall_detector.get_confidence_threshold():
                    self._send_fall_notification(metrics)
                    self._fall_notified = True
                    self.falls_detected += 1
                    logger.warning(f"FALL DETECTED AND VALIDATED! Confidence: {metrics.confidence:.0%}")
        
        elif state == FallState.STANDING:
            # Reset notification flag when person recovers
            if self._fall_notified:
                logger.info("Person recovered, resetting fall notification flag")
                self._fall_notified = False
    
    def _send_fall_notification(self, metrics) -> None:
        """Send fall notification via webhook."""
        try:
            success = self.notifier.notify_fall(confidence=metrics.confidence)
            
            if success:
                self.notifications_sent += 1
                logger.info(f"Fall notification sent successfully (#{self.notifications_sent})")
            else:
                logger.warning("Failed to send notification (may be in cooldown or queued)")
                
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
    
    def _log_status(self) -> None:
        """Log periodic status update."""
        uptime = time.time() - self.start_time
        uptime_str = time.strftime("%H:%M:%S", time.gmtime(uptime))
        
        camera_stats = self.camera.get_stats()
        queue_size = self.notifier.get_queue_size() if self.notifier else 0
        
        status = (
            f"Status: Uptime={uptime_str} | "
            f"Frames={self.frames_processed} | "
            f"FPS={camera_stats['actual_fps']:.1f} | "
            f"Falls={self.falls_detected} | "
            f"Notifications={self.notifications_sent} | "
            f"Queued={queue_size}"
        )
        logger.info(status)
        
        # Check system health
        self._check_system_health()
    
    def _check_system_health(self) -> None:
        """Check system health (CPU temp, etc.)."""
        if not config.ENABLE_SYSTEM_MONITORING:
            return
        
        try:
            # Read CPU temperature on Raspberry Pi
            temp_file = "/sys/class/thermal/thermal_zone0/temp"
            if os.path.exists(temp_file):
                with open(temp_file, 'r') as f:
                    temp = int(f.read().strip()) / 1000
                
                if temp >= config.CPU_TEMP_CRITICAL:
                    logger.error(f"CPU temperature critical: {temp}°C - consider reducing load")
                elif temp >= config.CPU_TEMP_WARNING:
                    logger.warning(f"CPU temperature high: {temp}°C")
                else:
                    logger.debug(f"CPU temperature: {temp}°C")
                    
        except Exception as e:
            logger.debug(f"Could not read CPU temperature: {e}")
    
    def shutdown(self) -> None:
        """Gracefully shutdown all components."""
        logger.info("Shutting down fall detection system...")
        
        self.running = False
        
        # Stop camera
        if self.camera:
            try:
                self.camera.stop()
            except Exception as e:
                logger.warning(f"Error stopping camera: {e}")
        
        # Shutdown notifier
        if self.notifier:
            try:
                self.notifier.shutdown()
            except Exception as e:
                logger.warning(f"Error shutting down notifier: {e}")
        
        # Final statistics
        if self.start_time:
            uptime = time.time() - self.start_time
            uptime_str = time.strftime("%H:%M:%S", time.gmtime(uptime))
            
            print()
            print("=" * 60)
            print("  SHUTDOWN COMPLETE")
            print("=" * 60)
            print(f"  Uptime: {uptime_str}")
            print(f"  Frames processed: {self.frames_processed}")
            print(f"  Falls detected: {self.falls_detected}")
            print(f"  Notifications sent: {self.notifications_sent}")
            print("=" * 60)
            print()
        
        logger.info("Shutdown complete")


def main():
    """Entry point for the fall detection system."""
    # Ensure required directories exist
    config.ensure_directories()
    
    # Add file logging
    if config.ENABLE_LOGGING:
        file_handler = logging.FileHandler(config.SYSTEM_LOG_FILE)
        file_handler.setLevel(getattr(logging, config.LOG_LEVEL))
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logging.getLogger().addHandler(file_handler)
    
    # Run the application
    app = FallDetectionPi()
    app.run()


if __name__ == "__main__":
    main()
