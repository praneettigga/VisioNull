#!/bin/bash
# VisioNull Fall Detection System - Raspberry Pi Setup Script
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
NC='\033[0m' # No Color

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
    sudo apt install -y \
        python3-pip \
        python3-opencv \
        python3-picamera2 \
        python3-numpy \
        libcamera-apps \
        libatlas-base-dev \
        libjasper-dev \
        libqtgui4 \
        libqt4-test
    echo -e "${GREEN}✓ System dependencies installed${NC}"
}

# Check camera
check_camera() {
    echo ""
    echo "Step 3: Checking camera..."
    
    # Check if camera is detected
    if libcamera-hello --list-cameras 2>/dev/null | grep -q "Available cameras"; then
        echo -e "${GREEN}✓ Camera detected${NC}"
        libcamera-hello --list-cameras
    else
        echo -e "${YELLOW}Warning: No camera detected${NC}"
        echo "Please ensure:"
        echo "  1. Camera ribbon cable is properly connected"
        echo "  2. Camera is enabled in raspi-config"
        echo ""
        echo "To enable camera:"
        echo "  sudo raspi-config -> Interface Options -> Camera -> Enable"
        echo "  sudo reboot"
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
    
    # Install mediapipe (may take a while on Pi)
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
    
    echo -e "${GREEN}✓ Python dependencies installed${NC}"
}

# Create directories
create_directories() {
    echo ""
    echo "Step 5: Creating directories..."
    
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    
    mkdir -p "$SCRIPT_DIR/logs"
    mkdir -p "$SCRIPT_DIR/data"
    
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
    
    # Update service file with correct paths
    CURRENT_USER=$(whoami)
    sed -i "s|User=pi|User=$CURRENT_USER|g" "$SERVICE_FILE"
    sed -i "s|/home/pi/VisioNull|$SCRIPT_DIR|g" "$SERVICE_FILE"
    
    # Install service
    sudo cp "$SERVICE_FILE" /etc/systemd/system/
    sudo systemctl daemon-reload
    
    echo -e "${GREEN}✓ Service installed${NC}"
    echo ""
    echo "To enable auto-start on boot:"
    echo "  sudo systemctl enable visionull"
    echo ""
    echo "To start the service:"
    echo "  sudo systemctl start visionull"
    echo ""
    echo "To check status:"
    echo "  sudo systemctl status visionull"
}

# Test the system
test_system() {
    echo ""
    echo "Step 8: Testing the system..."
    
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    
    echo "Running quick camera test..."
    cd "$SCRIPT_DIR"
    
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
# Create a dummy frame
frame = np.zeros((480, 640, 3), dtype=np.uint8)
landmarks = pose.process_frame(frame)
print(f'Pose estimator initialized: SUCCESS')
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
    echo "To test the webhook, you can use webhook.site:"
    echo "  1. Go to https://webhook.site"
    echo "  2. Copy your unique URL"
    echo "  3. Paste it in src/config.py as WEBHOOK_URL"
    echo ""
    echo "To run manually:"
    echo "  python3 -m src.main_pi"
    echo ""
    echo "To run as a service:"
    echo "  sudo systemctl start visionull"
    echo "  sudo systemctl enable visionull  # auto-start on boot"
    echo ""
    echo "To view logs:"
    echo "  tail -f logs/system.log"
    echo "  tail -f logs/falls.log"
    echo ""
    echo "=============================================="
}

# Main installation flow
main() {
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

# Run main function
main
