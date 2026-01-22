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
| `fall_head_threshold` | 0.65 | Head Y position ratio to consider "low" |
| `horizontal_ratio_threshold` | 0.8 | Body width/height ratio for "horizontal" |
| `fall_confirm_frames` | 8 | Frames to confirm a fall |
| `head_velocity_threshold` | 15.0 | Pixels/frame for "falling motion" |

---

## Raspberry Pi Setup Instructions

### Prerequisites

- Raspberry Pi 3, 4, or 5
- Raspberry Pi Camera Module (v1, v2, or v3) or USB webcam
- Raspberry Pi OS (Bullseye or newer recommended)
- Monitor, keyboard, and mouse for initial setup

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

## Usage

### Running the Application

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

---

## Project Structure

```
VisioNull/
├── src/
│   ├── __init__.py          # Package initialization
│   ├── camera_stream.py     # Camera capture module
│   ├── pose_estimator.py    # MediaPipe Pose wrapper
│   ├── fall_detector.py     # Rule-based fall detection
│   └── main.py              # Main application
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

### Module Descriptions

| Module | Purpose |
|--------|---------|
| `camera_stream.py` | Handles camera capture with OpenCV. Provides frame generator and clean shutdown. |
| `pose_estimator.py` | Wraps MediaPipe Pose. Extracts 33 body landmarks in pixel coordinates. |
| `fall_detector.py` | Rule-based state machine analyzing pose landmarks to detect falls. |
| `main.py` | Integrates all modules. Handles display, overlays, and user input. |

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

Tune the detection thresholds in `fall_detector.py`:

```python
self.fall_detector = FallDetector(
    fall_head_threshold=0.70,      # Increase to require head lower
    horizontal_ratio_threshold=1.0, # Increase for stricter horizontal detection
    fall_confirm_frames=10          # Increase for fewer false positives
)
```

---

## Extending the Project

### Adding Alerts

To add audio/visual alerts when a fall is detected, modify `main.py`:

```python
# In the main loop, after state detection:
if state == FallState.FALLEN:
    # Play sound
    os.system('aplay alert.wav &')
    # Or send notification
    # send_sms_alert()
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
