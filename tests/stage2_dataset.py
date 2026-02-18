#!/usr/bin/env python3
"""
Stage 2: Dataset Loading & Annotation Test
Verifies that the CCTV Incident Dataset loads correctly and that
ground-truth annotations (bounding boxes + COCO 17-keypoint skeletons) parse properly.

Usage:
    python3 tests/stage2_dataset.py
    python3 tests/stage2_dataset.py --path data/cctv-incident/images
    python3 tests/stage2_dataset.py --browse   # Interactive viewer
"""

import sys
import os
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
from src.dataset_stream import DatasetStream, COCO_KEYPOINT_NAMES, GroundTruthAnnotation
from typing import List, Optional


# COCO skeleton connections for visualization
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),        # Head
    (5, 6),                                   # Shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),          # Arms
    (5, 11), (6, 12),                          # Torso
    (11, 12),                                  # Hips
    (11, 13), (13, 15), (12, 14), (14, 16),    # Legs
]


def draw_ground_truth(frame: np.ndarray, annotations: List[GroundTruthAnnotation]) -> np.ndarray:
    """Draw ground-truth annotations on a frame."""
    output = frame.copy()

    for ann in annotations:
        # Color by class
        color = (0, 0, 255) if ann.class_name == "laying" else (0, 255, 0)
        label = f"{ann.class_name} (id={ann.class_id})"

        # Draw bounding box
        cx, cy, w, h = ann.bbox
        x1, y1 = int(cx - w / 2), int(cy - h / 2)
        x2, y2 = int(cx + w / 2), int(cy + h / 2)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.putText(output, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw keypoints
        if ann.keypoints:
            for i, (kx, ky, kv) in enumerate(ann.keypoints):
                if kv > 0:  # visible or occluded
                    pt_color = (0, 255, 255) if kv == 2 else (0, 165, 255)  # yellow=visible, orange=occluded
                    cv2.circle(output, (int(kx), int(ky)), 4, pt_color, -1)
                    # Optional: draw keypoint name
                    # cv2.putText(output, COCO_KEYPOINT_NAMES[i][:3], (int(kx)+5, int(ky)-5),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.3, pt_color, 1)

            # Draw skeleton connections
            for (i, j) in COCO_SKELETON:
                if i < len(ann.keypoints) and j < len(ann.keypoints):
                    kx1, ky1, kv1 = ann.keypoints[i]
                    kx2, ky2, kv2 = ann.keypoints[j]
                    if kv1 > 0 and kv2 > 0:
                        cv2.line(output, (int(kx1), int(ky1)), (int(kx2), int(ky2)), (255, 255, 255), 1)

    return output


def run_verification(source_path: str, labels_path: Optional[str] = None):
    """Run non-interactive verification of dataset loading."""
    print("\n── Dataset Verification ──")

    ds = DatasetStream(source_path, labels_path=labels_path)

    if not ds.start():
        print("  Failed to start DatasetStream")
        return False

    stats = ds.get_stats()
    print(f"  Source: {stats['source']}")
    print(f"  Total images: {stats['total_frames']}")
    print(f"  Labels found: {stats['labels_found']}")

    # Iterate through all frames
    class_counts = {}
    keypoint_stats = {"with_kp": 0, "without_kp": 0}
    frame_count = 0
    annotation_count = 0

    while True:
        frame = ds.get_frame()
        if frame is None:
            break

        frame_count += 1
        gt = ds.get_ground_truth()

        if gt:
            for ann in gt:
                annotation_count += 1
                class_counts[ann.class_name] = class_counts.get(ann.class_name, 0) + 1
                if ann.keypoints:
                    keypoint_stats["with_kp"] += 1
                    # Verify keypoint count
                    if len(ann.keypoints) != 17:
                        print(f"  WARNING: Expected 17 keypoints, got {len(ann.keypoints)} in frame {frame_count}")
                else:
                    keypoint_stats["without_kp"] += 1

        # Verify frame properties
        if frame_count == 1:
            print(f"  First frame — shape: {frame.shape}, dtype: {frame.dtype}")

    ds.stop()

    print(f"\n  Results:")
    print(f"    Frames loaded: {frame_count}")
    print(f"    Annotations: {annotation_count}")
    print(f"    Class distribution: {class_counts}")
    print(f"    Annotations with keypoints: {keypoint_stats['with_kp']}")
    print(f"    Annotations without keypoints: {keypoint_stats['without_kp']}")

    if frame_count > 0 and annotation_count > 0:
        print(f"\n  ✓ Dataset verification PASSED")
        return True
    elif frame_count > 0:
        print(f"\n  ⚠ Images loaded but no annotations found")
        print(f"    Check that label .txt files exist alongside images")
        return True  # Partial success
    else:
        print(f"\n  ✗ Dataset verification FAILED — no frames loaded")
        return False


def run_interactive_browser(source_path: str, labels_path: Optional[str] = None):
    """Interactive visual browser for the dataset."""
    print("\n── Interactive Dataset Browser ──")
    print("  Controls: [→/d] next | [←/a] prev | [q] quit")

    ds = DatasetStream(source_path, labels_path=labels_path)
    if not ds.start():
        print("  Failed to start DatasetStream")
        return

    # Load all frames into memory for browsing
    frames = []
    ground_truths = []
    paths = []

    while True:
        frame = ds.get_frame()
        if frame is None:
            break
        frames.append(frame)
        ground_truths.append(ds.get_ground_truth())
        paths.append(ds.get_current_image_path())

    ds.stop()

    if not frames:
        print("  No frames loaded")
        return

    print(f"  Loaded {len(frames)} frames")

    idx = 0
    while True:
        frame = frames[idx]
        gt = ground_truths[idx]
        path = paths[idx]

        # Draw annotations
        if gt:
            display = draw_ground_truth(frame, gt)
            gt_text = ", ".join(f"{a.class_name}" for a in gt)
        else:
            display = frame.copy()
            gt_text = "no annotations"

        # Info overlay
        filename = os.path.basename(path) if path else "?"
        info = f"[{idx+1}/{len(frames)}] {filename} | GT: {gt_text}"
        cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display, "[a/d] prev/next | [q] quit", (10, display.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Stage 2: Dataset Browser", display)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key in (ord('d'), 83, 3):  # d, right arrow
            idx = min(idx + 1, len(frames) - 1)
        elif key in (ord('a'), 81, 2):  # a, left arrow
            idx = max(idx - 1, 0)

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Dataset Loading Test")
    parser.add_argument("--path", default=None, help="Path to dataset images directory or video file")
    parser.add_argument("--labels", default=None, help="Path to labels directory (auto-detected if not set)")
    parser.add_argument("--browse", action="store_true", help="Open interactive browser")
    args = parser.parse_args()

    print("=" * 50)
    print("  VisioNull — Stage 2: Dataset")
    print("=" * 50)

    # Default path
    source_path = args.path
    if source_path is None:
        # Try to find the dataset in the standard location
        candidates = [
            os.path.join(PROJECT_ROOT, "data", "cctv-incident", "images"),
            os.path.join(PROJECT_ROOT, "data", "cctv-incident", "train", "images"),
            os.path.join(PROJECT_ROOT, "data", "cctv-incident"),
        ]
        for c in candidates:
            if os.path.isdir(c):
                source_path = c
                break

        if source_path is None:
            print("\n  Dataset not found. Download it first:")
            print("    bash tests/download_dataset.sh")
            print("\n  Or specify path manually:")
            print("    python3 tests/stage2_dataset.py --path /path/to/images")
            return 1

    print(f"  Source: {source_path}")

    if args.browse:
        run_interactive_browser(source_path, labels_path=args.labels)
    else:
        success = run_verification(source_path, labels_path=args.labels)
        if success:
            print(f"\n{'=' * 50}")
            print("  Stage 2 PASSED — Dataset loads correctly")
            print("  Try: python3 tests/stage2_dataset.py --browse")
            print("  Proceed to Stage 3 (pose estimation)")
            print(f"{'=' * 50}\n")
        else:
            print(f"\n{'=' * 50}")
            print("  Stage 2 FAILED")
            print(f"{'=' * 50}\n")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
