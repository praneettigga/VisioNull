# Laptop Setup Guide (Linux Development)

Complete setup guide for developing and improving the fall detection system on a Linux laptop.
This guide covers installation, dataset setup, testing, and model tuning.

---

## Prerequisites

- **Linux OS** (Ubuntu/Debian-based preferred)
- **Python 3.9+** installed
- **Webcam** (built-in or USB) for live testing
- **~500 MB disk space** for dataset
- **Internet connection** for downloading dependencies and dataset

---

## Initial Setup

### 1. Install System Dependencies

First, install the required system packages:

```bash
# Update package list
sudo apt update

# Install Python virtual environment support
sudo apt install python3-venv

# Install Python development tools
sudo apt install python3-pip python3-dev

# Install OpenCV system dependencies (optional but recommended for better performance)
sudo apt install -y libopencv-dev

# Install additional libraries for MediaPipe
sudo apt install -y libgl1-mesa-glx libglib2.0-0
```

> **Note:** If you encountered the `ensurepip is not available` error, the `python3-venv` package fixes it.
> For specific Python versions (e.g., Python 3.13), use: `sudo apt install python3.13-venv`

### 2. Navigate to Project

```bash
cd ~/MyStuff/MyProjects/VisioNull
```

### 3. Create Virtual Environment

```bash
# Create virtual environment (first time only)
python3 -m venv venv
```

### 4. Activate Virtual Environment

**Run this every time you open a new terminal:**

```bash
source venv/bin/activate
```

You should see `(venv)` at the start of your prompt:
```
(venv) praneet@Legion:~/MyStuff/MyProjects/VisioNull$
```

### 5. Install Python Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

**Expected packages:**
- `opencv-python` - Image processing
- `mediapipe` - Pose estimation
- `numpy` - Numerical operations

---

## Dataset Setup

The project uses the **CCTV Incident Dataset** for offline testing and model tuning.

### Option 1: Download via Kaggle CLI (Recommended)

```bash
# Install Kaggle CLI
pip install kaggle

# Set up Kaggle credentials
# 1. Go to https://www.kaggle.com/settings
# 2. Click "Create New Token" under API section
# 3. Save kaggle.json to ~/.kaggle/

mkdir -p ~/.kaggle
# Copy your kaggle.json to ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Download dataset
bash tests/download_dataset.sh
```

### Option 2: Manual Download

If Kaggle CLI doesn't work:

```bash
# The script will provide a download link
bash tests/download_dataset.sh

# Or manually:
# 1. Visit: https://www.kaggle.com/datasets/simuletic/cctv-incident-dataset-fall-and-lying-down-detection
# 2. Download and extract to: data/cctv-incident/
```

**Dataset structure after download:**
```
data/cctv-incident/
├── images/          # 111 synthetic images
├── labels/          # YOLO Pose format annotations
└── data.yaml        # Dataset metadata
```

---

## Staged Testing (Development Workflow)

Run these tests **in order** to verify each component works correctly:

### Stage 0: Environment Check

Verify all dependencies and model file are present:

```bash
python3 tests/stage0_env_check.py
```

**Expected output:**
- ✓ Python version
- ✓ All libraries installed
- ✓ Pose model file exists

### Stage 1: Camera Test

Test webcam capture:

```bash
python3 tests/stage1_camera.py
```

**Expected result:**
- Saves `test_frame.jpg` in project root
- Displays camera info and resolution

**If it fails:**
```bash
# Try different camera index
python3 tests/stage1_camera.py --camera 1
```

### Stage 2: Dataset Verification

Verify dataset is downloaded and accessible:

```bash
# Check dataset structure
python3 tests/stage2_dataset.py

# Browse dataset interactively (with visualization)
python3 tests/stage2_dataset.py --browse
```

**Controls in browse mode:**
- `n` - Next image
- `p` - Previous image  
- `q` - Quit

### Stage 3: Pose Estimation

Test MediaPipe pose detection:

```bash
# Test on live webcam
python3 tests/stage3_pose.py --live

# Test on dataset (with accuracy metrics)
python3 tests/stage3_pose.py --dataset
```

**Expected output (dataset mode):**
- Pose detection accuracy
- Visualization of pose landmarks
- Per-class accuracy (standing vs laying)

### Stage 4: Fall Detection

Test the fall detection algorithm:

```bash
# Test on live webcam (act out falls)
python3 tests/stage4_fall_detection.py --live

# Test on dataset (with precision/recall metrics)
python3 tests/stage4_fall_detection.py --dataset

# With visualization (saves annotated images)
python3 tests/stage4_fall_detection.py --dataset --visualize
```

**Expected output (dataset mode):**
- Precision, Recall, F1-Score
- Confusion matrix
- False positive/negative analysis

### Stage 5: Threshold Tuning (Model Improvement)

**This is where you improve the model!**

Sweep detection thresholds to find optimal configuration:

```bash
# Grid search over threshold combinations
python3 tests/stage5_tune.py

# Also compare ML classifiers (Random Forest, SVM, etc.)
python3 tests/stage5_tune.py --ml

# Use custom dataset path
python3 tests/stage5_tune.py --dataset-path data/custom-dataset/images
```

**What it does:**
- Tests combinations of `fall_head_threshold`, `horizontal_ratio_threshold`, `fall_confirm_frames`
- Computes precision/recall for each combination
- Reports best configuration
- (With `--ml` flag) Trains ML models on extracted pose features

**Expected output:**
```
Best Configuration:
  fall_head_threshold: 0.55
  horizontal_ratio_threshold: 0.6
  fall_confirm_frames: 6
  Precision: 0.92
  Recall: 0.88
  F1-Score: 0.90
```

**To apply the best thresholds:**
1. Note the best configuration from the output
2. Edit [src/config.py](src/config.py):
   ```python
   FALL_HEAD_THRESHOLD = 0.55
   HORIZONTAL_RATIO_THRESHOLD = 0.6
   FALL_CONFIRM_FRAMES = 6
   ```
3. Re-run Stage 4 to verify improvement

### Stage 6: Full Pipeline

End-to-end integration test with webhook simulation:

```bash
# Runs local webhook server + full detection pipeline
python3 tests/stage6_full_pipeline.py --live
```

---

## Running the Application

### Desktop Mode (with GUI)

```bash
# Activate virtual environment first
source venv/bin/activate

# Run with defaults (640x480)
python -m src.main

# Run with higher resolution (better accuracy)
python -m src.main --width 1280 --height 720

# Use specific camera
python -m src.main --camera 1

# Hide debug overlay
python -m src.main --no-debug
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `D` | Toggle debug metrics |
| `R` | Reset fall detector state |

---

## Model Tuning & Improvement

### Understanding Detection Thresholds

The fall detector uses these key parameters (in [src/config.py](src/config.py)):

| Parameter | Default | Effect | Tuning Tips |
|-----------|---------|--------|-------------|
| `FALL_HEAD_THRESHOLD` | 0.55 | Head Y position ratio (0-1) to consider "low" | **Lower** = more sensitive (detects falls earlier) |
| `HORIZONTAL_RATIO_THRESHOLD` | 0.6 | Body width/height ratio for "horizontal" | **Lower** = easier to trigger horizontal state |
| `FALL_CONFIRM_FRAMES` | 6 | Frames needed to confirm fall | **Fewer** = faster detection, more false positives |
| `POST_FALL_VALIDATION_SECONDS` | 2.0 | Time person must stay down after fall | **Shorter** = faster alert, more false positives |
| `HEAD_VELOCITY_THRESHOLD` | 10.0 | Pixels/frame for falling motion | **Lower** = detects slower falls |

### Tuning Workflow

1. **Baseline evaluation:**
   ```bash
   python3 tests/stage4_fall_detection.py --dataset
   ```
   Note current precision/recall.

2. **Run grid search:**
   ```bash
   python3 tests/stage5_tune.py
   ```
   Let it test combinations (may take 5-10 minutes).

3. **Analyze results:**
   - Check which thresholds give best F1-score
   - Consider your use case:
     - **High precision** needed? (Avoid false alarms) → Higher thresholds
     - **High recall** needed? (Don't miss real falls) → Lower thresholds

4. **Update config:**
   Edit `src/config.py` with the best values.

5. **Verify improvement:**
   ```bash
   python3 tests/stage4_fall_detection.py --dataset
   ```

6. **Test live:**
   ```bash
   python3 tests/stage4_fall_detection.py --live
   ```
   Act out falls and normal movements to verify in real conditions.

### Training ML Classifiers (Advanced)

Instead of rule-based detection, train a machine learning model:

```bash
# Train and compare multiple ML models
python3 tests/stage5_tune.py --ml
```

**What it does:**
- Extracts features from pose landmarks (angles, ratios, velocities)
- Trains Random Forest, SVM, Logistic Regression, etc.
- Compares accuracy vs rule-based detector
- If ML model is better, you can integrate it into `fall_detector.py`

### Using Custom Datasets

To test on your own video or images:

```bash
# From video file
python3 tests/stage3_pose.py --dataset --dataset-path ~/Videos/test_falls.mp4

# From image directory
python3 tests/stage4_fall_detection.py --dataset --dataset-path ~/Pictures/fall_test/

# Labels expected in YOLO Pose format (optional)
# Place .txt files with same name as images in labels/ subdirectory
```

---

## Development Tips

### Quick Test Loop

```bash
# 1. Make changes to src/fall_detector.py
# 2. Test on dataset
python3 tests/stage4_fall_detection.py --dataset
# 3. If good, test live
python3 tests/stage4_fall_detection.py --live
```

### Debugging

Enable verbose output:

```python
# In src/fall_detector.py, add print statements:
def update(self, landmarks, frame_height):
    # ... existing code ...
    print(f"Debug: head_y={head_y:.2f}, body_angle={body_angle:.1f}°")
```

Run with debug overlay:

```bash
python -m src.main  # Press 'D' to toggle debug info
```

### Performance Profiling

Check FPS and timing:

```bash
# The application shows FPS in the window title
# Or add timing in code:
import time
start = time.time()
# ... your code ...
print(f"Took {time.time() - start:.3f}s")
```

---

## Troubleshooting

### "No module named 'cv2'" Error

You forgot to activate the virtual environment:

```bash
source venv/bin/activate
```

### Camera Not Found

```bash
# List available cameras
ls /dev/video*

# Try different index
python -m src.main --camera 1
```

### Dataset Not Found

```bash
# Re-download
bash tests/download_dataset.sh

# Or check if path exists
ls -la data/cctv-incident/images/
```

### Low FPS

```bash
# Use lower resolution
python -m src.main --width 320 --height 240

# Close other applications
# Check CPU usage: htop
```

### Import Errors

```bash
# Verify installation
pip list | grep -E 'opencv|mediapipe|numpy'

# Reinstall if needed
pip install --force-reinstall opencv-python mediapipe numpy
```

---

## Configuration for Production

Before deploying to Raspberry Pi, configure [src/config.py](src/config.py):

```python
# Device identification
DEVICE_NAME = "your-device-name"
DEVICE_LOCATION = "Room Name"

# Webhook for notifications
WEBHOOK_URL = "https://your-webhook-url.com/fall-alert"

# Optimized settings
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
TARGET_FPS = 15
```

Test webhook locally:

```bash
# Get free test webhook
# Visit: https://webhook.site
# Copy the URL to config.py
# Run the app and trigger a fall
```

---

## Deactivating Virtual Environment

When done working:

```bash
deactivate
```

---

## Next Steps

1. ✅ **Complete all staged tests** (Stage 0-6)
2. ✅ **Tune detection thresholds** using Stage 5
3. ✅ **Test with live camera** acting out falls
4. 📝 **Document your optimal thresholds** in config.py
5. 🚀 **Deploy to Raspberry Pi** (see main README.md)

---

## Quick Reference

```bash
# Daily workflow
cd ~/MyStuff/MyProjects/VisioNull
source venv/bin/activate

# Test changes
python3 tests/stage4_fall_detection.py --dataset

# Run application
python -m src.main

# When done
deactivate
```
