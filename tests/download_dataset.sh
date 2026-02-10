#!/bin/bash
# Download the CCTV Incident Dataset (Fall & Lying Down Detection)
# Source: https://www.kaggle.com/datasets/simuletic/cctv-incident-dataset-fall-and-lying-down-detection
# Size: ~172 MB — 111 synthetic images with COCO 17-keypoint skeleton annotations
# License: CC BY-NC-SA 4.0
#
# This dataset contains:
#   - Synthetic images of people standing and laying down
#   - YOLO Pose format labels with bounding boxes + 17-keypoint skeletons
#   - Classes: 0 = laying (fallen), 1 = standing
#
# Prerequisites:
#   pip install kaggle
#   Set up ~/.kaggle/kaggle.json with your API key
#   (Get it from: https://www.kaggle.com/settings → API → Create New Token)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/data"
DATASET_DIR="$DATA_DIR/cctv-incident"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "=============================================="
echo "  VisioNull — Dataset Download"
echo "=============================================="
echo ""

# Check if dataset already exists
if [ -d "$DATASET_DIR" ] && [ "$(ls -A "$DATASET_DIR" 2>/dev/null)" ]; then
    img_count=$(find "$DATASET_DIR" -name "*.jpg" -o -name "*.png" 2>/dev/null | wc -l)
    echo -e "${GREEN}Dataset already exists at $DATASET_DIR ($img_count images)${NC}"
    read -p "Redownload? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping download."
        exit 0
    fi
fi

mkdir -p "$DATA_DIR"

# Try Kaggle CLI first
if command -v kaggle &>/dev/null; then
    echo "Using Kaggle CLI to download dataset..."
    echo ""

    if [ ! -f ~/.kaggle/kaggle.json ]; then
        echo -e "${YELLOW}Kaggle API key not found at ~/.kaggle/kaggle.json${NC}"
        echo ""
        echo "To set up Kaggle CLI:"
        echo "  1. Go to https://www.kaggle.com/settings"
        echo "  2. Scroll to 'API' section"
        echo "  3. Click 'Create New Token'"
        echo "  4. Move the downloaded file: mv ~/Downloads/kaggle.json ~/.kaggle/"
        echo "  5. chmod 600 ~/.kaggle/kaggle.json"
        echo "  6. Re-run this script"
        echo ""
        echo -e "${YELLOW}Alternatively, download manually:${NC}"
        echo "  1. Go to: https://www.kaggle.com/datasets/simuletic/cctv-incident-dataset-fall-and-lying-down-detection"
        echo "  2. Click 'Download' button"
        echo "  3. Extract to: $DATASET_DIR"
        exit 1
    fi

    kaggle datasets download \
        -d simuletic/cctv-incident-dataset-fall-and-lying-down-detection \
        -p "$DATA_DIR" \
        --unzip

    # The dataset may extract into a subdirectory — normalize the structure
    # Expected structure after download: images/ and labels/ directories
    EXTRACTED=$(find "$DATA_DIR" -maxdepth 3 -type d -name "images" | head -1)
    if [ -n "$EXTRACTED" ]; then
        EXTRACTED_PARENT="$(dirname "$EXTRACTED")"
        if [ "$EXTRACTED_PARENT" != "$DATASET_DIR" ]; then
            # Move to expected location
            rm -rf "$DATASET_DIR"
            mv "$EXTRACTED_PARENT" "$DATASET_DIR"
        fi
    fi

    # Clean up the zip if it remains
    rm -f "$DATA_DIR"/*.zip

    echo ""
    echo -e "${GREEN}Dataset downloaded successfully!${NC}"
else
    echo -e "${YELLOW}Kaggle CLI not found.${NC}"
    echo ""
    echo "Option 1: Install Kaggle CLI"
    echo "  pip install kaggle"
    echo "  # Then set up API key and re-run this script"
    echo ""
    echo "Option 2: Manual download"
    echo "  1. Go to: https://www.kaggle.com/datasets/simuletic/cctv-incident-dataset-fall-and-lying-down-detection"
    echo "  2. Click 'Download' button"
    echo "  3. Extract the zip to: $DATASET_DIR"
    echo "  4. Ensure this structure exists:"
    echo "     $DATASET_DIR/"
    echo "       images/"
    echo "         *.jpg or *.png"
    echo "       labels/"
    echo "         *.txt (YOLO Pose format)"
    exit 1
fi

# Verify
echo ""
echo "── Verification ──"
if [ -d "$DATASET_DIR" ]; then
    img_count=$(find "$DATASET_DIR" -name "*.jpg" -o -name "*.png" 2>/dev/null | wc -l)
    lbl_count=$(find "$DATASET_DIR" -name "*.txt" 2>/dev/null | wc -l)
    echo "  Images: $img_count"
    echo "  Labels: $lbl_count"
    echo "  Location: $DATASET_DIR"

    if [ "$img_count" -gt 0 ]; then
        echo ""
        echo -e "${GREEN}✓ Dataset ready! Run: python3 tests/stage2_dataset.py${NC}"
    else
        echo ""
        echo -e "${RED}✗ No images found. Check the directory structure.${NC}"
    fi
else
    echo -e "${RED}✗ Dataset directory not found.${NC}"
fi
