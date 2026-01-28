"""
Fall Detection System - Configuration
All settings for the Raspberry Pi fall detection prototype.

Edit this file to customize the system behavior.
"""

import os
from pathlib import Path

# =============================================================================
# DEVICE SETTINGS
# =============================================================================

# Unique name for this device (used in notifications)
DEVICE_NAME = "living-room-pi"

# Device location (optional, for notifications)
DEVICE_LOCATION = "Living Room"

# =============================================================================
# CAMERA SETTINGS
# =============================================================================

# Camera device index (usually 0 for USB webcam)
CAMERA_INDEX = 0

# Resolution for better accuracy (1280x720 recommended for Pi 4)
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Target FPS (Pi 4 can handle ~15-20 FPS at 720p with pose estimation)
TARGET_FPS = 15

# =============================================================================
# FALL DETECTION SETTINGS
# =============================================================================

# Pose estimation confidence (0.0 - 1.0)
# Higher = more accurate but may miss some detections
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.6

# Fall detection thresholds (medium sensitivity)
FALL_HEAD_THRESHOLD = 0.55          # Head Y ratio to consider "low" (0.55 = 55% down)
HORIZONTAL_RATIO_THRESHOLD = 0.6    # Body width/height ratio for "horizontal"
FALL_CONFIRM_FRAMES = 6             # Frames to initially detect fall

# Confidence threshold for sending notifications
# Only send notification if fall confidence >= this value
FALL_CONFIDENCE_THRESHOLD = 0.7     # 70% confidence required

# Post-detection validation: person must stay down for X seconds
# This prevents false positives from bending down
POST_FALL_VALIDATION_SECONDS = 2.0

# Head velocity threshold for detecting falling motion (pixels/frame)
HEAD_VELOCITY_THRESHOLD = 10.0

# Frames of standing required before exiting "fallen" state
RECOVERY_FRAMES = 10

# =============================================================================
# NOTIFICATION SETTINGS
# =============================================================================

# Webhook URL - REPLACE THIS with your actual webhook endpoint
# Examples:
#   - https://your-server.com/fall-alert
#   - https://webhook.site/your-unique-id (for testing)
#   - https://maker.ifttt.com/trigger/fall_detected/with/key/YOUR_KEY
WEBHOOK_URL = "https://webhook.site/your-unique-id"

# Webhook request timeout (seconds)
WEBHOOK_TIMEOUT = 10

# Cooldown between notifications (seconds)
# Prevents notification spam for the same fall event
NOTIFICATION_COOLDOWN_SECONDS = 30

# Retry settings for failed notifications
MAX_RETRY_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 5

# =============================================================================
# QUEUE SETTINGS (for offline notifications)
# =============================================================================

# Enable offline queue (saves notifications when internet is down)
ENABLE_OFFLINE_QUEUE = True

# Queue file location
PROJECT_ROOT = Path(__file__).parent.parent
QUEUE_FILE = PROJECT_ROOT / "data" / "notification_queue.json"

# Maximum queued notifications (oldest are dropped when exceeded)
MAX_QUEUE_SIZE = 100

# How often to retry sending queued notifications (seconds)
QUEUE_RETRY_INTERVAL = 60

# =============================================================================
# LOGGING SETTINGS
# =============================================================================

# Enable logging
ENABLE_LOGGING = True

# Log directory
LOG_DIR = PROJECT_ROOT / "logs"

# Log file for fall events
FALL_LOG_FILE = LOG_DIR / "falls.log"

# Log file for system events (startup, errors, etc.)
SYSTEM_LOG_FILE = LOG_DIR / "system.log"

# Log level: DEBUG, INFO, WARNING, ERROR
LOG_LEVEL = "INFO"

# Maximum log file size (bytes) before rotation
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10 MB

# Number of backup log files to keep
LOG_BACKUP_COUNT = 5

# =============================================================================
# SYSTEM SETTINGS
# =============================================================================

# Enable headless mode (no display window)
# Set to True for production on Pi without monitor
HEADLESS_MODE = True

# Processing interval (process every N frames to reduce CPU load)
# 1 = process every frame, 2 = every other frame, etc.
PROCESS_EVERY_N_FRAMES = 1

# Enable system monitoring (CPU temp, memory usage)
ENABLE_SYSTEM_MONITORING = True

# CPU temperature warning threshold (Celsius)
CPU_TEMP_WARNING = 70

# CPU temperature critical threshold (will reduce FPS)
CPU_TEMP_CRITICAL = 80

# =============================================================================
# DEBUG SETTINGS
# =============================================================================

# Enable debug mode (more verbose output)
DEBUG_MODE = False

# Save snapshot on fall detection (for debugging)
SAVE_FALL_SNAPSHOTS = False
SNAPSHOT_DIR = PROJECT_ROOT / "snapshots"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def ensure_directories():
    """Create required directories if they don't exist."""
    directories = [
        LOG_DIR,
        QUEUE_FILE.parent,
    ]
    
    if SAVE_FALL_SNAPSHOTS:
        directories.append(SNAPSHOT_DIR)
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def get_config_summary() -> str:
    """Return a summary of current configuration."""
    return f"""
╔══════════════════════════════════════════════════════════════╗
║          FALL DETECTION SYSTEM - CONFIGURATION               ║
╠══════════════════════════════════════════════════════════════╣
║  Device: {DEVICE_NAME:<50} ║
║  Location: {DEVICE_LOCATION:<48} ║
╠══════════════════════════════════════════════════════════════╣
║  Camera: Index {CAMERA_INDEX}, {FRAME_WIDTH}x{FRAME_HEIGHT} @ {TARGET_FPS} FPS{' ':<18} ║
║  Headless Mode: {str(HEADLESS_MODE):<44} ║
╠══════════════════════════════════════════════════════════════╣
║  Fall Confidence Threshold: {FALL_CONFIDENCE_THRESHOLD*100:.0f}%{' ':<28} ║
║  Post-Fall Validation: {POST_FALL_VALIDATION_SECONDS}s{' ':<33} ║
║  Notification Cooldown: {NOTIFICATION_COOLDOWN_SECONDS}s{' ':<32} ║
╠══════════════════════════════════════════════════════════════╣
║  Webhook: {WEBHOOK_URL[:48]:<50} ║
║  Offline Queue: {str(ENABLE_OFFLINE_QUEUE):<44} ║
╚══════════════════════════════════════════════════════════════╝
"""


# Ensure directories exist when config is imported
ensure_directories()
