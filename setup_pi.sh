#!/bin/bash
# VisioNull Fall Detection System - Raspberry Pi Setup Script
# Compatible with Raspberry Pi OS based on Debian Trixie / Bookworm
# Run this script on your Raspberry Pi to set up the fall detection system

set -e  # Exit on error

echo "=============================================="
echo "  VisioNull - Fall Detection Setup"
echo "  Raspberry Pi Installation Script"
echo "=============================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Detect OS version and set appropriate commands
detect_os() {
    echo "Detecting OS version..."
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS_VERSION_CODENAME="${VERSION_CODENAME:-unknown}"
        OS_PRETTY_NAME="${PRETTY_NAME:-unknown}"
    else
        OS_VERSION_CODENAME="unknown"
        OS_PRETTY_NAME="unknown"
    fi

    echo -e "${CYAN}OS: $OS_PRETTY_NAME ($OS_VERSION_CODENAME)${NC}"

    # Determine camera CLI tool prefix
    # Trixie and Bookworm use rpicam-*, older use libcamera-*
    case "$OS_VERSION_CODENAME" in
        trixie|bookworm)
            CAM_PREFIX="rpicam"
            CAM_APPS_PKG="rpicam-apps"
            echo -e "${GREEN}Using rpicam-* camera tools (modern)${NC}"
            ;;
        bullseye)
            CAM_PREFIX="libcamera"
            CAM_APPS_PKG="libcamera-apps"
            echo -e "${YELLOW}Using libcamera-* camera tools (Bullseye)${NC}"
            ;;
        *)
            CAM_PREFIX="rpicam"
            CAM_APPS_PKG="rpicam-apps"
            echo -e "${YELLOW}Unknown OS version, assuming rpicam-* tools${NC}"
            ;;
    esac
}

# Check if running on Raspberry Pi
check_pi() {
    if [ ! -f /proc/device-tree/model ]; then
        echo -e "${YELLOW}Warning: This doesn't appear to be a Raspberry Pi${NC}"
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        PI_MODEL=$(cat /proc/device-tree/model)
        echo -e "${GREEN}Detected: $PI_MODEL${NC}"
    fi
}

# Update system
update_system() {
    echo ""
    echo "Step 1: Updating system packages..."
    sudo apt update
    sudo apt upgrade -y
    echo -e "${GREEN}✓ System updated${NC}"
}

# Install system dependencies
install_dependencies() {
    echo ""
    echo "Step 2: Installing system dependencies..."

    # BLAS package name differs across Raspberry Pi OS releases
    if apt-cache show libopenblas-dev &>/dev/null; then
        BLAS_PKG="libopenblas-dev"
    else
        BLAS_PKG="libatlas-base-dev"
    fi

    # Core packages (available on all versions)
    PACKAGES=(
        python3-pip
        python3-opencv
        python3-numpy
        "$BLAS_PKG"
    )

    # Camera packages (from Raspberry Pi apt repos)
    PACKAGES+=(python3-picamera2)

    # rpicam-apps or libcamera-apps (may already be pre-installed)
    if apt-cache show "$CAM_APPS_PKG" &>/dev/null; then
        PACKAGES+=("$CAM_APPS_PKG")
    else
        echo -e "${YELLOW}Note: $CAM_APPS_PKG not found in apt cache (may be pre-installed)${NC}"
    fi

    # Build deps for pyenv / Python 3.12
    PACKAGES+=(
        build-essential libssl-dev zlib1g-dev libbz2-dev
        libreadline-dev libsqlite3-dev curl libncursesw5-dev xz-utils
        libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
        libharfbuzz-dev libhdf5-dev
    )

    sudo apt install -y "${PACKAGES[@]}" || {
        echo -e "${YELLOW}Some packages may have failed. Retrying core packages...${NC}"
        sudo apt install -y python3-pip python3-opencv python3-numpy "$BLAS_PKG"
        sudo apt install -y python3-picamera2 || echo -e "${YELLOW}python3-picamera2 not available via apt, will try pip later${NC}"
    }

    echo -e "${GREEN}✓ System dependencies installed${NC}"
}

# Check camera
check_camera() {
    echo ""
    echo "Step 3: Checking camera..."

    CAM_HELLO="${CAM_PREFIX}-hello"

    # Check if camera tool exists
    if ! command -v "$CAM_HELLO" &>/dev/null; then
        echo -e "${YELLOW}Warning: $CAM_HELLO not found in PATH${NC}"
        echo "Camera CLI tools may not be installed."
        echo "Try: sudo apt install $CAM_APPS_PKG"
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
        return
    fi

    # Check if camera is detected
    if $CAM_HELLO --list-cameras 2>/dev/null | grep -q -i "available\|imx\|ov5647\|camera"; then
        echo -e "${GREEN}✓ Camera detected${NC}"
        $CAM_HELLO --list-cameras
    else
        echo -e "${YELLOW}Warning: No camera detected${NC}"
        echo "Please ensure:"
        echo "  1. Camera ribbon cable is properly connected"
        echo "  2. Camera is seated correctly in the CSI connector"
        echo ""
        echo "On Trixie/Bookworm, the camera is auto-detected (no raspi-config step needed)."
        echo ""
        echo "To verify manually:"
        echo "  $CAM_HELLO --list-cameras"
        echo ""
        echo "If using a non-standard camera, add a dtoverlay to /boot/firmware/config.txt:"
        echo "  dtoverlay=imx219"
        echo ""
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Install Python 3.12 via pyenv
# Required because Debian Trixie ships Python 3.13, and mediapipe has no
# aarch64 wheel for Python 3.13 (last supported version: 0.10.18 on Python 3.12).
install_python312() {
    echo ""
    echo "Step 4a: Installing Python 3.12 via pyenv..."

    # Install pyenv if not already present
    if ! command -v pyenv &>/dev/null; then
        echo "Installing pyenv..."
        curl https://pyenv.run | bash

        # Add pyenv to current shell session
        export PYENV_ROOT="$HOME/.pyenv"
        export PATH="$PYENV_ROOT/bin:$PATH"
        eval "$(pyenv init - bash)"

        # Persist to .bashrc
        if ! grep -q 'PYENV_ROOT' ~/.bashrc; then
            echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
            echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
            echo 'eval "$(pyenv init - bash)"' >> ~/.bashrc
        fi
    else
        export PYENV_ROOT="$HOME/.pyenv"
        export PATH="$PYENV_ROOT/bin:$PATH"
        eval "$(pyenv init - bash)"
        echo "pyenv already installed"
    fi

    # Install Python 3.12 if not already present
    if pyenv versions | grep -q '3.12'; then
        echo "Python 3.12 already installed via pyenv"
    else
        echo "Building Python 3.12 (this takes 5-10 minutes)..."
        pyenv install 3.12
    fi

    echo -e "${GREEN}✓ Python 3.12 available${NC}"
}

# Create virtual environment with system-site-packages
# --system-site-packages is required so picamera2 (apt package) is accessible inside the venv.
setup_venv() {
    echo ""
    echo "Step 4b: Creating virtual environment..."

    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "$SCRIPT_DIR"

    # Set project Python to 3.12
    pyenv local 3.12

    # Remove stale venv if it exists
    if [ -d "venv" ]; then
        echo "Removing existing venv..."
        rm -rf venv
    fi

    # Create venv with system site-packages (needed for picamera2)
    python -m venv venv --system-site-packages

    echo -e "${GREEN}✓ Virtual environment created (venv/)${NC}"
    echo "  Activate with: source venv/bin/activate"
}

# Install Python dependencies into the venv
install_python_deps() {
    echo ""
    echo "Step 4c: Installing Python dependencies..."

    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PIP="$SCRIPT_DIR/venv/bin/pip"

    "$PIP" install --upgrade pip

    # mediapipe 0.10.18 is the last version with aarch64 Linux wheels.
    # Versions 0.10.20+ dropped aarch64 support; Python 3.13 has no wheel at all.
    echo "Installing mediapipe==0.10.18 (last aarch64-compatible release)..."
    "$PIP" install mediapipe==0.10.18 || {
        echo -e "${RED}Failed to install MediaPipe${NC}"
        echo "Ensure you are using Python 3.12 (pyenv local 3.12) and retry:"
        echo "  pip install mediapipe==0.10.18"
    }

    # Install remaining dependencies from requirements.txt
    "$PIP" install -r "$SCRIPT_DIR/requirements.txt"

    # Optional: scikit-learn for threshold tuning
    "$PIP" install scikit-learn 2>/dev/null || echo -e "${YELLOW}scikit-learn not installed (optional)${NC}"

    echo -e "${GREEN}✓ Python dependencies installed${NC}"
}

# Create directories
create_directories() {
    echo ""
    echo "Step 5: Creating directories..."

    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    mkdir -p "$SCRIPT_DIR/logs"
    mkdir -p "$SCRIPT_DIR/data"
    mkdir -p "$SCRIPT_DIR/tests"

    echo -e "${GREEN}✓ Directories created${NC}"
}

# Download pose model
download_model() {
    echo ""
    echo "Step 6: Downloading pose estimation model..."

    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    MODEL_PATH="$SCRIPT_DIR/pose_landmarker_lite.task"
    MODEL_URL="https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"

    if [ -f "$MODEL_PATH" ]; then
        echo "Model already exists, skipping download"
    else
        wget -O "$MODEL_PATH" "$MODEL_URL" || {
            echo -e "${YELLOW}Warning: Could not download model${NC}"
            echo "The model will be downloaded automatically on first run"
        }
    fi

    echo -e "${GREEN}✓ Model ready${NC}"
}

# Setup systemd service
setup_service() {
    echo ""
    echo "Step 7: Setting up systemd service..."

    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SERVICE_FILE="$SCRIPT_DIR/visionull.service"

    CURRENT_USER=$(whoami)
    sed -i "s|User=pi|User=$CURRENT_USER|g" "$SERVICE_FILE"
    sed -i "s|/home/pi/VisioNull|$SCRIPT_DIR|g" "$SERVICE_FILE"

    sudo cp "$SERVICE_FILE" /etc/systemd/system/
    sudo systemctl daemon-reload

    echo -e "${GREEN}✓ Service installed${NC}"
    echo ""
    echo "To enable auto-start on boot:"
    echo "  sudo systemctl enable visionull"
    echo ""
    echo "To start the service:"
    echo "  sudo systemctl start visionull"
}

# Test the system
test_system() {
    echo ""
    echo "Step 8: Testing the system..."

    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PYTHON="$SCRIPT_DIR/venv/bin/python"
    cd "$SCRIPT_DIR"

    echo "Running quick camera test..."
    "$PYTHON" -c "
from src.camera_stream import CameraStream, PICAMERA2_AVAILABLE
print(f'picamera2 available: {PICAMERA2_AVAILABLE}')
camera = CameraStream(frame_width=640, frame_height=480, fps=10)
if camera.start():
    print('Camera test: SUCCESS')
    frame = camera.get_frame()
    if frame is not None:
        print(f'Frame captured: {frame.shape}')
    camera.stop()
else:
    print('Camera test: FAILED')
" && echo -e "${GREEN}✓ Camera test passed${NC}" || echo -e "${YELLOW}Camera test failed${NC}"

    echo ""
    echo "Running quick pose estimation test..."
    "$PYTHON" -c "
from src.pose_estimator import PoseEstimator
import numpy as np
pose = PoseEstimator()
frame = np.zeros((480, 640, 3), dtype=np.uint8)
print('Pose estimator initialized: SUCCESS')
" && echo -e "${GREEN}✓ Pose estimation test passed${NC}" || echo -e "${YELLOW}Pose estimation test failed${NC}"
}

# Print final instructions
print_instructions() {
    echo ""
    echo "=============================================="
    echo "  Setup Complete!"
    echo "=============================================="
    echo ""
    echo "Before running, configure your settings:"
    echo "  1. Edit src/config.py"
    echo "  2. Set WEBHOOK_URL to your notification endpoint"
    echo "  3. Set DEVICE_NAME to identify this device"
    echo ""
    echo "Stage-by-stage validation:"
    echo "  python3 tests/stage0_env_check.py"
    echo "  python3 tests/stage1_camera.py"
    echo "  python3 tests/stage2_dataset.py"
    echo "  python3 tests/stage3_pose.py --live"
    echo "  python3 tests/stage4_fall_detection.py --live"
    echo ""
    echo "Activate the virtual environment first:"
    echo "  source ~/VisioNull/venv/bin/activate"
    echo ""
    echo "To run manually:"
    echo "  source venv/bin/activate && python -m src.main_pi"
    echo ""
    echo "To run as a service:"
    echo "  sudo systemctl start visionull"
    echo "  sudo systemctl enable visionull"
    echo ""
    echo "=============================================="
}

# Main installation flow
main() {
    detect_os
    check_pi
    update_system
    install_dependencies
    check_camera
    install_python312
    setup_venv
    install_python_deps
    create_directories
    download_model
    setup_service
    test_system
    print_instructions
}

main
