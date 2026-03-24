# VisioNull - Real-Time Fall Detection System

A real-time fall detection system designed for **Raspberry Pi** using a camera module, **MediaPipe Pose**, and **OpenCV**. The system detects when a person falls and sends HTTP webhook notifications to a laptop dashboard.

This directory (`rpi/`) contains all Raspberry Pi code. See [../README.md](../README.md) for the full system overview including the laptop dashboard.

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.18-orange.svg)

## Features

- **Real-time pose estimation** using MediaPipe's lightweight model
- **Rule-based fall detection** - no ML training required
- **Visual feedback** with status overlays and skeleton visualization
- **Debug mode** showing detection metrics
- **Optimized for Raspberry Pi** (Pi 3/4/5)
- **Dataset evaluation** — test with pre-recorded images for offline validation
- **Threshold tuning** — sweep detection parameters on labeled data

> **Note:** This project is compatible with **Raspberry Pi OS based on Debian Trixie/Bookworm**.
> Uses `rpicam-*` camera tools and auto-detected camera modules (no `raspi-config` camera enable needed).

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
| `fall_head_threshold` | 0.35 | Head Y position ratio to consider "low" |
| `horizontal_ratio_threshold` | 0.3 | Body width/height ratio for "horizontal" |
| `fall_confirm_frames` | 6 | Frames to confirm a fall |
| `head_velocity_threshold` | 10.0 | Pixels/frame for "falling motion" |
| `post_fall_validation_seconds` | 2.0 | Seconds person must stay down after detection |

---

## Raspberry Pi Setup Instructions

### Quick Setup (Recommended)

For automated installation, use the provided setup script:

```bash
cd ~/VisioNull/rpi
chmod +x setup.sh
./setup.sh
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
- Raspberry Pi OS (Trixie or Bookworm, 64-bit recommended)
- **Python 3.12** (required — see Step 4; Debian Trixie ships Python 3.13 which is not supported by MediaPipe on aarch64)
- Internet connection for notifications
- Monitor, keyboard, and mouse for initial setup (optional for headless)

### Step 1: Update Your System

Open a terminal and run:

```bash
sudo apt update && sudo apt upgrade -y
```

### Step 2: Verify Camera is Detected

On **Trixie/Bookworm**, the camera is auto-detected — no enable step needed.

```bash
# List available cameras
rpicam-hello --list-cameras
# Should list your camera (e.g. imx219 for Camera Module v2)
```

> **Note:** On older Raspberry Pi OS (Bullseye), use `libcamera-hello --list-cameras` instead.
> If your camera isn't detected, check the ribbon cable connection and try adding
> `dtoverlay=imx219` to `/boot/firmware/config.txt`, then reboot.

### Step 3: Install System Dependencies

```bash
# Install Python development tools
sudo apt install -y python3-pip python3-venv

# Install OpenCV system dependencies
sudo apt install -y libopencv-dev python3-opencv

# Install Pi Camera support (for Pi Camera Module)
sudo apt install -y python3-picamera2

# Install additional libraries for MediaPipe
sudo apt install -y libopenblas-dev libhdf5-dev libharfbuzz-dev

# Install pyenv build dependencies (needed for Step 4)
sudo apt install -y build-essential libssl-dev zlib1g-dev libbz2-dev \
  libreadline-dev libsqlite3-dev curl libncursesw5-dev xz-utils \
  libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
```

> If you're on an older Raspberry Pi OS image where `libopenblas-dev` isn't available,
> use `libatlas-base-dev` instead.

### Step 4: Install Python 3.12 and Create Virtual Environment

> **Why Python 3.12?** Debian Trixie ships Python 3.13, but MediaPipe's pre-built
> aarch64 wheels only support Python 3.8–3.12. Use [pyenv](https://github.com/pyenv/pyenv)
> to install Python 3.12 alongside the system Python.

```bash
# Install pyenv
curl https://pyenv.run | bash

# Add pyenv to your shell (for bash)
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init - bash)"' >> ~/.bashrc
source ~/.bashrc

# Install Python 3.12
pyenv install 3.12
```

```bash
# Navigate to your projects folder
cd ~

# Clone or copy the project (if using git)
# git clone <your-repo-url> VisioNull
# OR create the directory manually if you have the files

cd VisioNull/rpi
pyenv local 3.12

# Create a virtual environment using Python 3.12
# --system-site-packages is required so picamera2 (system package) is accessible
python -m venv venv --system-site-packages

# Activate the virtual environment
source venv/bin/activate
```

> **Note:** `--system-site-packages` is required because `picamera2` is only available
> as a system package (`sudo apt install python3-picamera2`) and cannot be installed
> via pip. Without it, the camera module silently falls back to OpenCV, which cannot
> access the Pi Camera.

> **Note:** Always activate the virtual environment before running the project:
> ```bash
> source ~/VisioNull/rpi/venv/bin/activate
> ```

### Step 5: Install Python Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

#### MediaPipe on Raspberry Pi (aarch64)

MediaPipe's aarch64 Linux wheels **only support Python 3.8–3.12**. Versions 0.10.20 and later
dropped aarch64 Linux support entirely. The last compatible release is `0.10.18`.

> This is why Python 3.12 via pyenv is required (see Step 4). The system Python 3.13
> on Debian Trixie has **no available mediapipe wheel**.

```bash
# Install the last aarch64-compatible mediapipe release explicitly
pip install mediapipe==0.10.18

# Then install the rest of the dependencies
pip install -r requirements.txt
```

### Step 6: Test Your Setup

Run these tests in order to verify each component works:

#### Test 1: Camera Stream

```bash
python -m src.pipeline.camera_stream
```

**Expected Result:**
- A window opens showing live camera feed
- Frame counter visible in top-left
- Press `q` to quit

**If it fails:**
- Check camera connection
- Try `--camera 1` if using USB webcam
- Run `rpicam-hello --list-cameras` to verify camera is detected

#### Test 2: Pose Estimation

```bash
python -m src.pipeline.pose_estimator
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
python -m src.pipeline.fall_detector
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

## Staged Testing (Recommended)

The project includes stage-by-stage test scripts that validate each layer independently.
Run them **in order** — each builds on the previous:

```bash
# Activate the virtual environment first
source ~/VisioNull/rpi/venv/bin/activate

# Stage 0: Verify environment (tools, libraries, model file)
python tests/stage0_env_check.py

# Stage 1: Test camera capture (saves a test frame)
python tests/stage1_camera.py

# Stage 2: Load & verify dataset (download first)
bash tests/download_dataset.sh
python tests/stage2_dataset.py
python tests/stage2_dataset.py --browse   # Interactive viewer

# Stage 3: Test pose estimation
python tests/stage3_pose.py --live        # On camera
python tests/stage3_pose.py --dataset     # On dataset (with accuracy report)

# Stage 4: Test fall detection
python tests/stage4_fall_detection.py --live      # Act out falls
python tests/stage4_fall_detection.py --dataset   # Precision/recall on dataset

# Stage 5: Optimize thresholds
python tests/stage5_tune.py               # Sweep thresholds
python tests/stage5_tune.py --ml          # Also compare ML classifiers

# Stage 6: Full pipeline integration
python tests/stage6_full_pipeline.py --live   # With local test webhook
```

### Dataset Evaluation

The system supports offline evaluation using labeled datasets. We use the
**CCTV Incident Dataset** (111 synthetic images with COCO 17-keypoint skeleton
annotations, CC BY-NC-SA 4.0).

**Download:**
```bash
bash tests/download_dataset.sh
```

**Evaluate:**
```bash
# Run fall detection on all images, compare against ground truth
python3 tests/stage4_fall_detection.py --dataset --visualize

# Sweep thresholds to find optimal configuration
python3 tests/stage5_tune.py

# Compare an ML classifier against rule-based detection
python3 tests/stage5_tune.py --ml
```

**Custom datasets:** Any directory of images (with optional YOLO Pose label files)
or video file can be used:
```bash
python3 tests/stage3_pose.py --dataset --dataset-path /path/to/images
python3 tests/stage4_fall_detection.py --dataset --dataset-path /path/to/video.mp4
```

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
FALL_HEAD_THRESHOLD = 0.35           # Lower = more sensitive
HORIZONTAL_RATIO_THRESHOLD = 0.3    # Lower = easier to detect horizontal
FALL_CONFIRM_FRAMES = 6              # Fewer = faster detection
POST_FALL_VALIDATION_SECONDS = 2.0   # Person must stay down this long
FALL_CONFIDENCE_THRESHOLD = 0.7      # Minimum confidence to notify

# Notification settings
NOTIFICATION_COOLDOWN_SECONDS = 30   # Prevent spam
ENABLE_OFFLINE_QUEUE = True          # Queue notifications when offline

# Pre-fall clip settings (RAM-only buffer)
PRE_FALL_BUFFER_SECONDS = 15          # Keep last 15s of frames in memory
PRE_FALL_CLIP_FPS = 12                # Encoded clip FPS
CLIP_UPLOAD_TIMEOUT = 45              # Timeout for clip upload request only
CLIP_UPLOAD_MAX_RETRIES = 2           # In-memory retry attempts
MAX_CLIP_UPLOAD_BYTES = 20971520      # 20 MB max clip payload
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

When a fall is validated, the RPi now uses a two-step transfer:

1. POST metadata to `/webhook` (existing behavior)
2. Upload pre-fall clip to `/api/events/<id>/clip` using multipart form-data (`clip` field)

The clip is extracted from a rolling 15-second in-memory frame buffer anchored to the first `VALIDATING` transition.
No clip files are written to disk on the RPi.

> Note: MP4 encoding uses `ffmpeg` via stdin/stdout pipes. Install ffmpeg on the RPi for clip upload support.

**Webhook examples:**
- **Testing**: [webhook.site](https://webhook.site) - Get a free test URL
- **IFTTT**: `https://maker.ifttt.com/trigger/fall_detected/with/key/YOUR_KEY`
- **Home Assistant**: `https://your-ha.com/api/webhook/fall_alert`
- **Custom server**: Your own API endpoint

### Running in Headless Mode

For production deployment without a display:

```bash
# Activate virtual environment
source ~/VisioNull/rpi/venv/bin/activate

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
tail -f ~/VisioNull/rpi/logs/system.log

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
source ~/VisioNull/rpi/venv/bin/activate

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
rpi/
├── src/
│   ├── __init__.py              # Package initialization
│   ├── config.py                # Configuration settings
│   ├── main.py                  # Main application (with display)
│   ├── main_pi.py               # Headless Pi application (production)
│   ├── pipeline/
│   │   ├── __init__.py          # Re-exports pipeline modules
│   │   ├── camera_stream.py     # Camera capture (picamera2 + OpenCV)
│   │   ├── dataset_stream.py    # Dataset/video frame source for offline eval
│   │   ├── pose_estimator.py    # MediaPipe Pose wrapper
│   │   └── fall_detector.py     # Rule-based fall detection
│   └── notification/
│       ├── __init__.py          # Re-exports notification modules
│       └── notifier.py          # Webhook notification system
├── tests/
│   ├── stage0_env_check.py      # Environment verification
│   ├── stage1_camera.py         # Camera capture test
│   ├── stage2_dataset.py        # Dataset loading test
│   ├── stage3_pose.py           # Pose estimation test
│   ├── stage4_fall_detection.py # Fall detection test
│   ├── stage5_tune.py           # Threshold tuning
│   ├── stage6_full_pipeline.py  # Full integration test
│   └── download_dataset.sh     # Dataset download script
├── model/
│   └── pose_landmarker_lite.task # MediaPipe pose model
├── data/                         # Datasets (gitignored)
├── logs/                         # Runtime logs
├── requirements.txt              # Python dependencies
├── setup.sh                      # Automated setup script
├── visionull.service             # Systemd service file
└── README.md                     # This file
```

### Module Descriptions

| Module | Purpose |
|--------|---------|
| `pipeline/camera_stream.py` | Camera capture with picamera2 (Pi Camera) or OpenCV fallback (USB webcam). Auto-reconnect support. |
| `pipeline/dataset_stream.py` | Loads frames from image directories or video files for offline evaluation. Parses YOLO Pose annotations. |
| `pipeline/pose_estimator.py` | Wraps MediaPipe Pose Landmarker. Extracts 33 body landmarks in pixel coordinates. |
| `pipeline/fall_detector.py` | Rule-based state machine with post-fall validation. Analyzes pose landmarks to detect falls. |
| `config.py` | Central configuration file. Customize thresholds, camera settings, webhook URL, and more. |
| `notification/notifier.py` | HTTP webhook notification system with offline queue, retry logic, and cooldown. |
| `main.py` | Desktop application with GUI. Displays video feed with overlays and debug metrics. |
| `main_pi.py` | **Production entry point for Raspberry Pi.** Headless mode with webhook notifications. |

---

## Troubleshooting

### Camera Not Working

1. **Check if camera is detected:**
   ```bash
   rpicam-hello --list-cameras
   # Should list your camera model (e.g. imx219)
   ```

2. **Check device nodes:**
   ```bash
   ls /dev/video*
   # Should show /dev/video0 or similar
   ```

3. **Quick preview test:**
   ```bash
   rpicam-hello -t 5000
   # Shows a 5-second preview window
   ```

4. **If camera not detected:**
   - Check ribbon cable is firmly seated in the CSI connector
   - Try adding `dtoverlay=imx219` to `/boot/firmware/config.txt` and reboot
   - Ensure no other process is using the camera

5. **Try different camera index (USB webcam):**
   ```bash
   python -m src.pipeline.camera_stream --camera 1
   ```

### MediaPipe Installation Issues

1. **`No matching distribution found for mediapipe` (Python 3.13):**
   MediaPipe has no aarch64 wheel for Python 3.13. You must use Python 3.12 via pyenv.
   Follow Step 4 above, then reinstall:
   ```bash
   # Confirm you're on Python 3.12
   python --version  # Should print Python 3.12.x

   # Install correct mediapipe version
   pip install mediapipe==0.10.18
   ```

2. **Memory errors during install:**
   ```bash
   # Use swap space
   sudo dphys-swapfile swapoff
   sudo nano /etc/dphys-swapfile
   # Set CONF_SWAPSIZE=2048
   sudo dphys-swapfile setup
   sudo dphys-swapfile swapon
   ```

3. **Missing shared libraries:**
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

# Balanced (tuned default — F1=0.957)
FALL_HEAD_THRESHOLD = 0.35
HORIZONTAL_RATIO_THRESHOLD = 0.3
FALL_CONFIRM_FRAMES = 6
POST_FALL_VALIDATION_SECONDS = 2.0
```

---

## Extending the Project

### Custom Notifications

The webhook system ([src/notification/notifier.py](src/notification/notifier.py)) makes it easy to integrate with any service:

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
- **/Pushbullet**: For mobile notifications

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



Remove after:
# On the Pi
curl -X POST http://<LAPTOP_IP>:5000/webhook \
  -H "Content-Type: application/json" \
  -d '{"device_name":"pi-test","message":"connectivity test","fall_confidence":0.5,"event_id":"ping-001"}'