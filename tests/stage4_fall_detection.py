#!/usr/bin/env python3
"""
Stage 4: Fall Detection Test
Runs camera/dataset → pose estimation → fall detection pipeline.

In dataset mode, compares predicted fall state against ground-truth labels
and reports precision, recall, F1, and confusion matrix.

In live mode, displays real-time state transitions for manual testing.

Usage:
    python3 tests/stage4_fall_detection.py --live
    python3 tests/stage4_fall_detection.py --dataset
    python3 tests/stage4_fall_detection.py --dataset --dataset-path data/cctv-incident/images
"""

import sys
import os
import argparse
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
from src.pose_estimator import PoseEstimator
from src.fall_detector import FallDetector, FallState, FallMetrics
from src.dataset_stream import DatasetStream, GroundTruthAnnotation
from typing import List, Optional, Dict


# Mapping from dataset class names to expected fall states
# "laying" = person is on the ground → should be detected as FALLEN or FALLING
# "standing" = person is upright → should be detected as STANDING
GROUND_TRUTH_MAP = {
    "laying": True,     # Is a fall
    "standing": False,  # Not a fall
}


def compute_metrics(y_true: list, y_pred: list) -> dict:
    """Compute classification metrics."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall,
        "f1": f1, "accuracy": accuracy,
    }


def test_live(num_frames: int = 300):
    """Live camera fall detection test."""
    print("\n── Live Fall Detection ──")
    print("  Stand in view of the camera and act out falls to test detection.")
    print("  Press 'q' to quit, 'r' to reset detector state.\n")

    from src.camera_stream import CameraStream

    camera = CameraStream(camera_index=0, frame_width=640, frame_height=480, fps=15)
    pose = PoseEstimator(model_complexity=0)
    detector = FallDetector()

    if not camera.start():
        print("  Failed to start camera")
        return False

    prev_state = None

    for i in range(num_frames):
        frame = camera.get_frame()
        if frame is None:
            break

        h, w = frame.shape[:2]

        # Pose estimation
        landmarks = pose.get_landmarks(frame)

        # Fall detection
        state, metrics = detector.update(landmarks, h)

        # Log state transitions
        if state != prev_state:
            print(f"  [{i:4d}] State: {prev_state} → {state.value}"
                  f" (conf={metrics.confidence:.2f}, angle={metrics.body_angle:.0f}°,"
                  f" head={metrics.head_height_ratio:.2f})")
            prev_state = state

        if detector.is_fall_validated():
            print(f"  [{i:4d}] ** FALL VALIDATED ** confidence={metrics.confidence:.2f}")

        # Draw
        output = pose.draw_landmarks(frame, landmarks)

        # State banner
        state_colors = {
            FallState.STANDING: (0, 180, 0),
            FallState.FALLING: (0, 165, 255),
            FallState.VALIDATING: (0, 255, 255),
            FallState.FALLEN: (0, 0, 255),
            FallState.UNKNOWN: (128, 128, 128),
        }
        color = state_colors.get(state, (128, 128, 128))
        cv2.rectangle(output, (0, 0), (w, 40), color, -1)
        cv2.putText(output, f"{state.value.upper()} ({metrics.confidence:.0%})",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Metrics
        y = 70
        for label, val in [
            ("Angle", f"{metrics.body_angle:.0f}"),
            ("Head Y", f"{metrics.head_height_ratio:.2f}"),
            ("Head V", f"{metrics.head_velocity:.1f}"),
            ("SH Ratio", f"{metrics.shoulder_hip_ratio:.2f}"),
            ("Hip Y", f"{metrics.hip_height_ratio:.2f}"),
            ("Leg Comp", f"{metrics.leg_compression:.2f}"),
        ]:
            cv2.putText(output, f"{label}: {val}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
            y += 18

        if metrics.validation_time > 0:
            cv2.putText(output, f"Validating: {metrics.validation_time:.1f}s remaining",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        cv2.putText(output, "[q] quit | [r] reset", (10, output.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        cv2.imshow("Stage 4: Fall Detection", output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            detector.reset()
            print("  [RESET] Detector state cleared")

    camera.stop()
    pose.close()
    cv2.destroyAllWindows()

    print("  ✓ Live fall detection test completed")
    return True


def test_dataset(source_path: str, labels_path: Optional[str] = None, visualize: bool = False):
    """
    Run fall detection on dataset and evaluate against ground truth.

    For a dataset of static images (not video sequences), the FallDetector
    state machine won't work naturally since it requires temporal progression.
    So we evaluate using the per-frame body metrics (pose features) and compare
    the is_fall_position condition against ground-truth class labels.
    """
    print(f"\n── Dataset Fall Detection Evaluation ──")
    print(f"  Source: {source_path}")

    ds = DatasetStream(source_path, labels_path=labels_path)
    pose = PoseEstimator(model_complexity=0)

    if not ds.start():
        print("  Failed to start DatasetStream")
        return False

    y_true = []   # Ground truth: True = fall, False = not fall
    y_pred = []   # Prediction: True = fall position detected, False = not
    details = []  # Per-image details for analysis

    frame_count = 0
    no_person_count = 0

    # Since the dataset is static images, we create a fresh detector for each
    # image to avoid state machine carry-over. We also evaluate the raw
    # body metrics to determine if the pose would be classified as a fall position.
    from src import config

    while True:
        frame = ds.get_frame()
        if frame is None:
            break

        frame_count += 1
        gt = ds.get_ground_truth()
        h, w = frame.shape[:2]

        # Pose estimation
        landmarks = pose.get_landmarks(frame)

        if landmarks is None:
            no_person_count += 1
            if gt:
                for ann in gt:
                    is_fall_gt = GROUND_TRUTH_MAP.get(ann.class_name)
                    if is_fall_gt is not None:
                        y_true.append(is_fall_gt)
                        y_pred.append(False)  # No detection = predict standing
            continue

        # Create a fresh detector and feed enough frames to get a reading
        detector = FallDetector()
        # Feed the same landmarks multiple times to get past the confirm threshold
        # This simulates a static person holding this position
        for _ in range(detector.fall_confirm_frames + 2):
            state, metrics = detector.update(landmarks, h)

        # The state after repeated frames tells us if this static pose
        # would eventually be classified as a fall
        is_fall_pred = state in (FallState.FALLING, FallState.FALLEN, FallState.VALIDATING)

        if gt:
            for ann in gt:
                is_fall_gt = GROUND_TRUTH_MAP.get(ann.class_name)
                if is_fall_gt is not None:
                    y_true.append(is_fall_gt)
                    y_pred.append(is_fall_pred)
                    details.append({
                        "img": ds.get_current_image_path(),
                        "gt": ann.class_name,
                        "pred_state": state.value,
                        "angle": metrics.body_angle,
                        "head_y": metrics.head_height_ratio,
                        "sh_ratio": metrics.shoulder_hip_ratio,
                    })

        # Visual
        if visualize:
            output = pose.draw_landmarks(frame, landmarks)
            gt_text = gt[0].class_name if gt else "?"
            pred_text = state.value
            color = (0, 0, 255) if is_fall_pred else (0, 255, 0)
            cv2.putText(output, f"GT: {gt_text} | Pred: {pred_text}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.imshow("Stage 4: Dataset Eval", output)
            key = cv2.waitKey(200) & 0xFF
            if key == ord('q'):
                break

        if frame_count % 20 == 0:
            print(f"    Processed {frame_count} frames...")

    ds.stop()
    pose.close()
    if visualize:
        cv2.destroyAllWindows()

    # Results
    print(f"\n  Results:")
    print(f"    Images: {frame_count}")
    print(f"    No person detected: {no_person_count}")
    print(f"    Evaluated: {len(y_true)}")

    if y_true:
        m = compute_metrics(y_true, y_pred)
        print(f"\n  Confusion Matrix:")
        print(f"    {'':>12s}  {'Pred Fall':>10s}  {'Pred Stand':>10s}")
        print(f"    {'GT Fall':>12s}  {m['tp']:>10d}  {m['fn']:>10d}")
        print(f"    {'GT Stand':>12s}  {m['fp']:>10d}  {m['tn']:>10d}")
        print(f"\n  Metrics:")
        print(f"    Accuracy:  {m['accuracy']:.3f}")
        print(f"    Precision: {m['precision']:.3f}")
        print(f"    Recall:    {m['recall']:.3f}")
        print(f"    F1 Score:  {m['f1']:.3f}")

        # Show misclassified examples
        misclassified = [d for d, t, p in zip(details, y_true, y_pred) if t != p]
        if misclassified:
            print(f"\n  Misclassified ({len(misclassified)}):")
            for d in misclassified[:10]:
                print(f"    {os.path.basename(d['img'])}: GT={d['gt']}, Pred={d['pred_state']},"
                      f" angle={d['angle']:.0f}, head_y={d['head_y']:.2f}, sh_ratio={d['sh_ratio']:.2f}")
    else:
        print("  No ground-truth labels available for evaluation")

    print(f"\n  ✓ Dataset fall detection evaluation completed")
    return True


def main():
    parser = argparse.ArgumentParser(description="Stage 4: Fall Detection Test")
    parser.add_argument("--live", action="store_true", help="Live camera test")
    parser.add_argument("--dataset", action="store_true", help="Dataset evaluation")
    parser.add_argument("--dataset-path", type=str, help="Path to dataset images")
    parser.add_argument("--labels", type=str, help="Path to labels")
    parser.add_argument("--frames", type=int, default=300, help="Live frames (default: 300)")
    parser.add_argument("--visualize", action="store_true", help="Show images during dataset eval")
    args = parser.parse_args()

    if not args.live and not args.dataset:
        args.live = True

    print("=" * 50)
    print("  VisioNull — Stage 4: Fall Detection")
    print("=" * 50)

    if args.dataset:
        source = args.dataset_path
        if source is None:
            candidates = [
                os.path.join(PROJECT_ROOT, "data", "cctv-incident", "images"),
                os.path.join(PROJECT_ROOT, "data", "cctv-incident", "train", "images"),
                os.path.join(PROJECT_ROOT, "data", "cctv-incident"),
            ]
            for c in candidates:
                if os.path.isdir(c):
                    source = c
                    break
            if source is None:
                print("\n  Dataset not found. Run: bash tests/download_dataset.sh")
                return 1
        test_dataset(source, labels_path=args.labels, visualize=args.visualize)

    if args.live:
        test_live(num_frames=args.frames)

    print(f"\n{'=' * 50}")
    print("  Stage 4 complete")
    print("  Proceed to Stage 5 (threshold tuning)")
    print(f"{'=' * 50}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
