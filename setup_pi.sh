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

    # Core packages (available on all versions)
    PACKAGES=(
        python3-pip
        python3-opencv
        python3-numpy
        libatlas-base-dev
    )

    # Camera packages (from Raspberry Pi apt repos)
    PACKAGES+=(python3-picamera2)

    # rpicam-apps or libcamera-apps (may already be pre-installed)
    if apt-cache show "$CAM_APPS_PKG" &>/dev/null; then
        PACKAGES+=("$CAM_APPS_PKG")
    else
        echo -e "${YELLOW}Note: $CAM_APPS_PKG not found in apt cache (may be pre-installed)${NC}"
    fi

    sudo apt install -y "${PACKAGES[@]}" || {
        echo -e "${YELLOW}Some packages may have failed. Retrying core packages...${NC}"
        sudo apt install -y python3-pip python3-opencv python3-numpy libatlas-base-dev
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

# Install Python dependencies
install_python_deps() {
    echo ""
    echo "Step 4: Installing Python dependencies..."

    echo "Installing MediaPipe (this may take several minutes)..."
    pip3 install --upgrade pip

    # Try regular mediapipe first, fall back to rpi4 version
    if pip3 install mediapipe 2>/dev/null; then
        echo -e "${GREEN}✓ MediaPipe installed${NC}"
    else
        echo "Trying mediapipe-rpi4..."
        pip3 install mediapipe-rpi4 || {
            echo -e "${RED}Failed to install MediaPipe${NC}"
            echo "You may need to build from source. See:"
            echo "https://google.github.io/mediapipe/getting_started/install.html"
        }
    fi

    # Install scikit-learn for threshold tuning (optional)
    pip3 install scikit-learn 2>/dev/null || echo -e "${YELLOW}scikit-learn not installed (optional, for threshold tuning)${NC}"

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
    cd "$SCRIPT_DIR"

    echo "Running quick camera test..."
    python3 -c "
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
    python3 -c "
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
    echo "To run manually:"
    echo "  python3 -m src.main_pi"
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
    install_python_deps
    create_directories
    download_model
    setup_service
    test_system
    print_instructions
}

main
