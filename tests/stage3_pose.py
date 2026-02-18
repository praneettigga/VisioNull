#!/usr/bin/env python3
"""
Stage 3: Pose Estimation Test
Tests MediaPipe Pose Landmarker on live camera frames and/or dataset images.
Optionally compares MediaPipe output against COCO 17-keypoint ground truth.

Usage:
    python3 tests/stage3_pose.py --live              # Live camera
    python3 tests/stage3_pose.py --dataset            # Run on dataset
    python3 tests/stage3_pose.py --image tests/test_frame.jpg  # Single image
"""

import sys
import os
import argparse
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
from src.pose_estimator import PoseEstimator, Landmark
from src.dataset_stream import DatasetStream, COCO_TO_MEDIAPIPE, GroundTruthAnnotation
from typing import List, Optional, Tuple


def compute_keypoint_error(
    mp_landmarks: List[Landmark],
    gt_annotations: List[GroundTruthAnnotation],
    img_width: int,
    img_height: int
) -> Optional[dict]:
    """
    Compare MediaPipe landmarks against COCO ground-truth keypoints.

    Returns dict with per-keypoint errors in pixels, or None if comparison
    is not possible.
    """
    if not gt_annotations:
        return None

    # Use the first annotation with keypoints
    gt = None
    for ann in gt_annotations:
        if ann.keypoints:
            gt = ann
            break
    if gt is None:
        return None

    errors = {}
    for coco_idx, mp_idx in COCO_TO_MEDIAPIPE.items():
        gt_x, gt_y, gt_v = gt.keypoints[coco_idx]

        if gt_v <= 0:  # Not labeled
            continue

        if mp_idx >= len(mp_landmarks):
            continue

        mp_lm = mp_landmarks[mp_idx]
        if mp_lm.visibility < 0.3:
            continue

        dx = mp_lm.x - gt_x
        dy = mp_lm.y - gt_y
        dist = np.sqrt(dx * dx + dy * dy)

        kp_name = gt.keypoint_names[coco_idx]
        errors[kp_name] = dist

    return errors if errors else None


def test_on_image(pose: PoseEstimator, image_path: str):
    """Test pose estimation on a single image."""
    print(f"\n── Single Image: {os.path.basename(image_path)} ──")

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"  Failed to read: {image_path}")
        return False

    print(f"  Shape: {frame.shape}")

    t0 = time.time()
    landmarks = pose.get_landmarks(frame)
    t1 = time.time()

    if landmarks is None:
        print(f"  No person detected ({(t1-t0)*1000:.0f} ms)")
        return True  # Not a failure

    print(f"  Landmarks: {len(landmarks)} ({(t1-t0)*1000:.0f} ms)")
    print(f"  Key points:")
    for idx in [0, 11, 12, 23, 24, 27, 28]:
        lm = landmarks[idx]
        print(f"    {lm.name:>18s}: ({lm.x:.0f}, {lm.y:.0f}) vis={lm.visibility:.2f}")

    # Display
    output = pose.draw_landmarks(frame, landmarks)
    cv2.imshow("Stage 3: Pose Estimation", output)
    print("  Press any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return True


def test_on_live(pose: PoseEstimator, num_frames: int = 100):
    """Test pose estimation with live camera."""
    print("\n── Live Camera Pose Estimation ──")

    from src.camera_stream import CameraStream

    camera = CameraStream(camera_index=0, frame_width=640, frame_height=480, fps=15)
    if not camera.start():
        print("  Failed to start camera")
        return False

    print("  Press 'q' to quit")

    frame_times = []
    detection_count = 0

    for i in range(num_frames):
        frame = camera.get_frame()
        if frame is None:
            break

        t0 = time.time()
        landmarks = pose.get_landmarks(frame)
        t1 = time.time()
        frame_times.append(t1 - t0)

        if landmarks:
            detection_count += 1

        # Draw
        output = pose.draw_landmarks(frame, landmarks)

        # Overlay
        fps_text = f"Pose: {1/(t1-t0):.0f} FPS" if (t1-t0) > 0 else "Pose: -- FPS"
        status = "DETECTED" if landmarks else "NO PERSON"
        color = (0, 255, 0) if landmarks else (0, 0, 255)
        cv2.putText(output, f"{fps_text} | {status} | {i+1}/{num_frames}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(output, "Press 'q' to quit", (10, output.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Stage 3: Live Pose", output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.stop()
    cv2.destroyAllWindows()

    # Stats
    if frame_times:
        avg_ms = sum(frame_times) / len(frame_times) * 1000
        print(f"\n  Results:")
        print(f"    Frames processed: {len(frame_times)}")
        print(f"    Detections: {detection_count}/{len(frame_times)} ({100*detection_count/len(frame_times):.0f}%)")
        print(f"    Avg pose time: {avg_ms:.0f} ms ({1000/avg_ms:.0f} FPS)")

    print("  ✓ Live pose test completed")
    return True


def test_on_dataset(pose: PoseEstimator, source_path: str, labels_path: Optional[str] = None):
    """Run pose estimation on all dataset images and compare against ground truth."""
    print(f"\n── Dataset Pose Estimation ──")
    print(f"  Source: {source_path}")

    ds = DatasetStream(source_path, labels_path=labels_path)
    if not ds.start():
        print("  Failed to start DatasetStream")
        return False

    all_errors = {}  # keypoint_name → list of errors
    detection_count = 0
    frame_count = 0
    pose_times = []

    while True:
        frame = ds.get_frame()
        if frame is None:
            break

        frame_count += 1
        gt = ds.get_ground_truth()

        t0 = time.time()
        landmarks = pose.get_landmarks(frame)
        t1 = time.time()
        pose_times.append(t1 - t0)

        if landmarks:
            detection_count += 1

            # Compare against ground truth
            if gt:
                h, w = frame.shape[:2]
                errors = compute_keypoint_error(landmarks, gt, w, h)
                if errors:
                    for kp_name, err_px in errors.items():
                        all_errors.setdefault(kp_name, []).append(err_px)

        # Progress
        if frame_count % 20 == 0:
            print(f"    Processed {frame_count} frames...")

    ds.stop()

    # Results
    print(f"\n  Results:")
    print(f"    Total frames: {frame_count}")
    print(f"    Person detected: {detection_count}/{frame_count} ({100*detection_count/frame_count:.0f}%)")
    if pose_times:
        avg_ms = sum(pose_times) / len(pose_times) * 1000
        print(f"    Avg pose time: {avg_ms:.0f} ms ({1000/avg_ms:.0f} FPS)")

    if all_errors:
        print(f"\n  Keypoint Error vs Ground Truth (pixels):")
        print(f"    {'Keypoint':<18s} {'Mean':>8s} {'Std':>8s} {'Max':>8s} {'N':>5s}")
        print(f"    {'─'*47}")

        total_errors = []
        for kp_name in [
            "nose", "left_shoulder", "right_shoulder",
            "left_hip", "right_hip", "left_knee", "right_knee",
            "left_ankle", "right_ankle", "left_eye", "right_eye",
            "left_elbow", "right_elbow", "left_wrist", "right_wrist",
            "left_ear", "right_ear"
        ]:
            if kp_name in all_errors:
                errs = all_errors[kp_name]
                total_errors.extend(errs)
                print(f"    {kp_name:<18s} {np.mean(errs):>8.1f} {np.std(errs):>8.1f} {np.max(errs):>8.1f} {len(errs):>5d}")

        if total_errors:
            print(f"    {'─'*47}")
            print(f"    {'OVERALL':<18s} {np.mean(total_errors):>8.1f} {np.std(total_errors):>8.1f} {np.max(total_errors):>8.1f} {len(total_errors):>5d}")
    else:
        print(f"\n  No ground-truth comparison available (no keypoint annotations)")

    print(f"\n  ✓ Dataset pose evaluation completed")
    return True


def main():
    parser = argparse.ArgumentParser(description="Stage 3: Pose Estimation Test")
    parser.add_argument("--live", action="store_true", help="Test with live camera")
    parser.add_argument("--dataset", action="store_true", help="Test on dataset")
    parser.add_argument("--image", type=str, help="Test on a single image file")
    parser.add_argument("--dataset-path", type=str, help="Path to dataset images dir")
    parser.add_argument("--labels", type=str, help="Path to dataset labels dir")
    parser.add_argument("--frames", type=int, default=100, help="Number of live frames (default: 100)")
    args = parser.parse_args()

    # Default: if nothing specified, try dataset then live
    if not args.live and not args.dataset and not args.image:
        args.live = True

    print("=" * 50)
    print("  VisioNull — Stage 3: Pose Estimation")
    print("=" * 50)

    pose = PoseEstimator(model_complexity=0)

    if args.image:
        success = test_on_image(pose, args.image)

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
                print("\n  Dataset not found. Download it first:")
                print("    bash tests/download_dataset.sh")
                pose.close()
                return 1

        success = test_on_dataset(pose, source, labels_path=args.labels)

    if args.live:
        success = test_on_live(pose, num_frames=args.frames)

    pose.close()

    print(f"\n{'=' * 50}")
    print("  Stage 3 complete")
    print("  Proceed to Stage 4 (fall detection)")
    print(f"{'=' * 50}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
