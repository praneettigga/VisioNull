# VisioNull - Real-Time Fall Detection System

A real-time fall detection system designed for **Raspberry Pi** using a camera module, **MediaPipe Pose**, and **OpenCV**. The system detects when a person falls and displays "FALL DETECTED" on the video feed.

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange.svg)

## Features

- **Real-time pose estimation** using MediaPipe's lightweight model
- **Rule-based fall detection** - no ML training required
- **Visual feedback** with status overlays and skeleton visualization
- **Debug mode** showing detection metrics
- **Optimized for Raspberry Pi** (Pi 3/4/5)

---

## How Fall Detection Works

The system uses a **rule-based approach** analyzing body pose landmarks from MediaPipe:

### Detection Logic

1. **Body Orientation Analysis**
   - Measures the angle between shoulders and hips
   - Standing: shoulders are vertically above hips
   - Fallen: shoulders and hips are roughly horizontal (side by side)

2. **Head Position Tracking**
   - Monitors where the head (nose landmark) is in the frame
   - Standing: head is in the upper portion of the frame
   - Fallen: head drops to the lower portion of the frame (>65% down)

3. **Velocity Detection**
   - Tracks rapid downward movement of the head
   - Sudden drops indicate falling motion

4. **Temporal Confirmation**
   - Requires the "fallen" position to persist for 8+ frames
   - Prevents false positives from quick movements like bending down
   - Uses a state machine: `STANDING` → `FALLING` → `FALLEN`

### Key Thresholds (Tunable)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fall_head_threshold` | 0.55 | Head Y position ratio to consider "low" |
| `horizontal_ratio_threshold` | 0.6 | Body width/height ratio for "horizontal" |
| `fall_confirm_frames` | 6 | Frames to confirm a fall |
| `head_velocity_threshold` | 10.0 | Pixels/frame for "falling motion" |
| `post_fall_validation_seconds` | 2.0 | Seconds person must stay down after detection |

---

## Raspberry Pi Setup Instructions

### Quick Setup (Recommended)

For automated installation, use the provided setup script:

```bash
cd ~/VisioNull
chmod +x setup_pi.sh
./setup_pi.sh
```

The script will:
- Update system packages
- Install dependencies (OpenCV, MediaPipe, picamera2)
- Enable camera interface
- Download pose estimation model
- Create required directories
- Install systemd service for auto-start
- Run system tests

### Manual Setup

#### Prerequisites

- Raspberry Pi 3, 4, or 5 (Pi 4 with 4GB+ RAM recommended)
- Raspberry Pi Camera Module (v1, v2, or v3) or USB webcam
- Raspberry Pi OS (Bullseye or Bookworm, 64-bit recommended)
- Internet connection for notifications
- Monitor, keyboard, and mouse for initial setup (optional for headless)

### Step 1: Update Your System

Open a terminal and run:

```bash
sudo apt update && sudo apt upgrade -y
```

### Step 2: Enable the Camera Interface

#### For Pi Camera Module:

**Option A: Using raspi-config (Recommended)**
```bash
sudo raspi-config
```
Navigate to: `Interface Options` → `Camera` → `Enable` → `Finish` → Reboot

**Option B: Using the desktop**
Go to: `Preferences` → `Raspberry Pi Configuration` → `Interfaces` → Enable `Camera`

Then reboot:
```bash
sudo reboot
```

#### Verify Camera is Detected:

For legacy camera stack:
```bash
vcgencmd get_camera
# Should show: supported=1 detected=1
```

For libcamera (newer systems):
```bash
libcamera-hello --list-cameras
# Should list your camera
```

### Step 3: Install System Dependencies

```bash
# Install Python development tools
sudo apt install -y python3-pip python3-venv

# Install OpenCV system dependencies
sudo apt install -y libopencv-dev python3-opencv

# Install Pi Camera support (for Pi Camera Module)
sudo apt install -y python3-picamera2 libcamera-apps

# Install additional libraries for MediaPipe
sudo apt install -y libatlas-base-dev libhdf5-dev libharfbuzz-dev
```

### Step 4: Create Project Directory and Virtual Environment

```bash
# Navigate to your projects folder
cd ~

# Clone or copy the project (if using git)
# git clone <your-repo-url> VisioNull
# OR create the directory manually if you have the files

cd VisioNull

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

> **Note:** Always activate the virtual environment before running the project:
> ```bash
> source ~/VisioNull/venv/bin/activate
> ```

### Step 5: Install Python Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

#### MediaPipe on Raspberry Pi

MediaPipe provides pre-built wheels for Raspberry Pi. If the standard `mediapipe` package doesn't work, try:

**For Raspberry Pi 4 (64-bit OS):**
```bash
pip install mediapipe
```

**For Raspberry Pi 3 or 32-bit OS:**
```bash
# Try the community builds
pip install mediapipe-rpi4
# OR
pip install mediapipe-rpi3
```

If you encounter issues, you may need to build from source or use an older version:
```bash
pip install mediapipe==0.10.0
```

### Step 6: Test Your Setup

Run these tests in order to verify each component works:

#### Test 1: Camera Stream

```bash
python -m src.camera_stream
```

**Expected Result:**
- A window opens showing live camera feed
- Frame counter visible in top-left
- Press `q` to quit

**If it fails:**
- Check camera connection
- Try `--camera 1` if using USB webcam
- Verify camera is enabled in raspi-config

#### Test 2: Pose Estimation

```bash
python -m src.pose_estimator
```

**Expected Result:**
- Camera feed with pose skeleton overlay
- Green dots on body joints
- White lines connecting joints
- Purple bounding box around detected person
- Press `q` to quit

**If it fails:**
- Ensure MediaPipe installed correctly
- Stand in view of camera (full body if possible)

#### Test 3: Fall Detection (Simulated)

```bash
python -m src.fall_detector
```

**Expected Result:**
- Text output showing simulated standing and fallen poses
- No camera required for this test

#### Test 4: Full Application

```bash
python -m src.main
```

**Expected Result:**
- Full fall detection interface
- Status banner at top: "STANDING" (green) or "FALL DETECTED" (red)
- Debug metrics in bottom-left (toggle with `d`)
- Skeleton overlay on detected person

---

## Production Deployment (Raspberry Pi)

### Configuration

Before running in production, configure the system by editing [src/config.py](src/config.py):

```python
# Device identification
DEVICE_NAME = "living-room-pi"  # Unique name for this device
DEVICE_LOCATION = "Living Room"  # Human-readable location

# Webhook URL for fall notifications (REQUIRED)
WEBHOOK_URL = "https://your-server.com/fall-alert"
# Test URL: https://webhook.site/your-unique-id

# Camera settings
FRAME_WIDTH = 1280   # Higher resolution = better accuracy
FRAME_HEIGHT = 720
TARGET_FPS = 15      # Pi 4 can handle 15-20 FPS

# Fall detection sensitivity
FALL_HEAD_THRESHOLD = 0.55           # Lower = more sensitive
HORIZONTAL_RATIO_THRESHOLD = 0.6    # Lower = easier to detect horizontal
FALL_CONFIRM_FRAMES = 6              # Fewer = faster detection
POST_FALL_VALIDATION_SECONDS = 2.0   # Person must stay down this long
FALL_CONFIDENCE_THRESHOLD = 0.7      # Minimum confidence to notify

# Notification settings
NOTIFICATION_COOLDOWN_SECONDS = 30   # Prevent spam
ENABLE_OFFLINE_QUEUE = True          # Queue notifications when offline
```

#### Webhook Setup

The system sends HTTP POST requests with JSON payload:

```json
{
  "timestamp": "2026-02-09T14:30:45",
  "device_name": "living-room-pi",
  "device_location": "Living Room",
  "message": "Fall detected!",
  "fall_confidence": 0.85,
  "event_id": "living-room-pi-20260209143045-0001"
}
```

**Webhook examples:**
- **Testing**: [webhook.site](https://webhook.site) - Get a free test URL
- **IFTTT**: `https://maker.ifttt.com/trigger/fall_detected/with/key/YOUR_KEY`
- **Home Assistant**: `https://your-ha.com/api/webhook/fall_alert`
- **Custom server**: Your own API endpoint

### Running in Headless Mode

For production deployment without a display:

```bash
# Activate virtual environment
source ~/VisioNull/venv/bin/activate

# Run headless application
python -m src.main_pi
```

The headless application:
- Runs without GUI/display
- Sends webhook notifications on fall detection
- Logs to `logs/system.log` and `logs/falls.log`
- Auto-recovers from camera failures
- Handles graceful shutdown (Ctrl+C)

### Running as a System Service

For auto-start on boot:

```bash
# Enable and start the service
sudo systemctl enable visionull
sudo systemctl start visionull

# Check status
sudo systemctl status visionull

# View logs
sudo journalctl -u visionull -f
# Or
tail -f ~/VisioNull/logs/system.log

# Stop service
sudo systemctl stop visionull

# Disable auto-start
sudo systemctl disable visionull
```

**Service features:**
- Starts automatically on boot
- Restarts on failure
- Runs as your user (access to camera)
- Logs to `logs/system.log` and `logs/error.log`

---

## Usage

### Running the Desktop Application (with Display)

```bash
# Activate virtual environment (if not already active)
source ~/VisioNull/venv/bin/activate

# Run with defaults
python -m src.main

# Run with custom camera index (e.g., USB webcam)
python -m src.main --camera 1

# Run with custom resolution
python -m src.main --width 1280 --height 720

# Run without debug overlay
python -m src.main --no-debug
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| `Q` | Quit the application |
| `D` | Toggle debug metrics overlay |
| `R` | Reset fall detector state |

### Command Line Options

**Desktop application (main.py):**
```
usage: main.py [-h] [--camera CAMERA] [--width WIDTH] [--height HEIGHT] [--no-debug]

optional arguments:
  -h, --help            show this help message and exit
  --camera CAMERA, -c CAMERA
                        Camera index (default: 0)
  --width WIDTH, -W WIDTH
                        Frame width (default: 640)
  --height HEIGHT, -H HEIGHT
                        Frame height (default: 480)
  --no-debug            Hide debug metrics overlay
```

**Headless application (main_pi.py):**
```
# Configuration via src/config.py or environment variables:
export VISIONULL_WEBHOOK_URL="https://your-webhook.com/endpoint"
export VISIONULL_DEVICE_NAME="my-device"
python -m src.main_pi
```

---

## Project Structure

```
VisioNull/
├── src/
│   ├── __init__.py          # Package initialization
│   ├── camera_stream.py     # Camera capture module (picamera2 + OpenCV)
│   ├── pose_estimator.py    # MediaPipe Pose wrapper
│   ├── fall_detector.py     # Rule-based fall detection
│   ├── config.py            # Configuration settings
│   ├── notifier.py          # Webhook notification system
│   ├── main.py              # Main application (with display)
│   └── main_pi.py           # Headless Pi application (production)
├── requirements.txt          # Python dependencies
├── setup_pi.sh               # Automated setup script
├── visionull.service         # Systemd service file
├── pose_landmarker_lite.task # MediaPipe pose model
└── README.md                 # This file
```

### Module Descriptions

| Module | Purpose |
|--------|---------|
| `camera_stream.py` | Camera capture with picamera2 (Pi Camera) or OpenCV fallback (USB webcam). Auto-reconnect support. |
| `pose_estimator.py` | Wraps MediaPipe Pose Landmarker. Extracts 33 body landmarks in pixel coordinates. |
| `fall_detector.py` | Rule-based state machine with post-fall validation. Analyzes pose landmarks to detect falls. |
| `config.py` | Central configuration file. Customize thresholds, camera settings, webhook URL, and more. |
| `notifier.py` | HTTP webhook notification system with offline queue, retry logic, and cooldown. |
| `main.py` | Desktop application with GUI. Displays video feed with overlays and debug metrics. |
| `main_pi.py` | **Production entry point for Raspberry Pi.** Headless mode with webhook notifications. |

---

## Troubleshooting

### Camera Not Working

1. **Check if camera is detected:**
   ```bash
   ls /dev/video*
   # Should show /dev/video0 or similar
   ```

2. **For Pi Camera with libcamera:**
   ```bash
   # Test with libcamera
   libcamera-hello
   ```

3. **Try different camera index:**
   ```bash
   python -m src.camera_stream --camera 1
   ```

### MediaPipe Installation Issues

1. **Memory errors during install:**
   ```bash
   # Use swap space
   sudo dphys-swapfile swapoff
   sudo nano /etc/dphys-swapfile
   # Set CONF_SWAPSIZE=2048
   sudo dphys-swapfile setup
   sudo dphys-swapfile swapon
   ```

2. **Missing shared libraries:**
   ```bash
   sudo apt install -y libgl1-mesa-glx libglib2.0-0
   ```

### Low FPS

- Use lower resolution: `--width 320 --height 240`
- Raspberry Pi 3 may run at 5-10 FPS; Pi 4/5 should achieve 15-25 FPS
- Close other applications to free resources

### False Positives/Negatives

Tune the detection thresholds in [src/config.py](src/config.py):

```python
# More sensitive (detects falls easier, more false positives)
FALL_HEAD_THRESHOLD = 0.50
HORIZONTAL_RATIO_THRESHOLD = 0.5
FALL_CONFIRM_FRAMES = 4
POST_FALL_VALIDATION_SECONDS = 1.0

# Less sensitive (fewer false positives, may miss some falls)
FALL_HEAD_THRESHOLD = 0.65
HORIZONTAL_RATIO_THRESHOLD = 0.8
FALL_CONFIRM_FRAMES = 10
POST_FALL_VALIDATION_SECONDS = 3.0

# Balanced (default)
FALL_HEAD_THRESHOLD = 0.55
HORIZONTAL_RATIO_THRESHOLD = 0.6
FALL_CONFIRM_FRAMES = 6
POST_FALL_VALIDATION_SECONDS = 2.0
```

---

## Extending the Project

### Custom Notifications

The webhook system ([src/notifier.py](src/notifier.py)) makes it easy to integrate with any service:

**1. SMS via Twilio:**
```python
# Add to notifier.py
import twilio
# Send SMS when webhook is called
```

**2. Email via SMTP:**
```python
import smtplib
# Send email notification
```

**3. Smart Home Integration:**
- **Home Assistant**: Use webhook automation
- **IFTTT**: Use Webhooks service
- **Pushover/Pushbullet**: For mobile notifications

**4. Local Alerts:**

To add audio/visual alerts, modify [src/main_pi.py](src/main_pi.py):

```python
# In the main loop, after fall detection:
if state == FallState.FALLEN:
    # Play sound
    os.system('aplay /usr/share/sounds/alert.wav &')
    # Flash LED on GPIO
    # GPIO.output(LED_PIN, GPIO.HIGH)
```

### Network Streaming

To stream the video over the network, consider integrating with Flask:

```python
# Example: Add a /video_feed endpoint
from flask import Flask, Response
```

### Logging Falls

Add timestamped logging:

```python
import logging
logging.basicConfig(filename='falls.log', level=logging.INFO)

if state == FallState.FALLEN:
    logging.info(f"Fall detected at {datetime.now()}")
```

---

## License

This project is open source and available under the MIT License.

---

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) by Google for pose estimation
- [OpenCV](https://opencv.org/) for computer vision
- Raspberry Pi Foundation for the hardware platform
