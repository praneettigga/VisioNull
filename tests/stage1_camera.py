#!/usr/bin/env python3
"""
Stage 1: Camera Capture Test
Tests that the IMX219 Pi Camera captures frames correctly via picamera2.
Displays a live preview window and saves a test frame.

Usage:
    python3 tests/stage1_camera.py
    python3 tests/stage1_camera.py --opencv     # Force OpenCV fallback
    python3 tests/stage1_camera.py --frames 60  # Capture 60 frames
"""

import sys
import os
import argparse
import time

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np


def test_picamera2_direct():
    """Test picamera2 directly without the VisioNull wrapper."""
    print("\n── Direct picamera2 Test ──")

    try:
        from picamera2 import Picamera2
        from libcamera import controls
    except ImportError as e:
        print(f"  picamera2 not available: {e}")
        print("  Skipping direct test. Use --opencv for OpenCV fallback.")
        return False

    try:
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        picam2.configure(config)
        picam2.start()

        # Let auto-exposure settle
        time.sleep(2)

        frame = picam2.capture_array()
        print(f"  Frame shape: {frame.shape}")
        print(f"  Frame dtype: {frame.dtype}")
        print(f"  Value range: [{frame.min()}, {frame.max()}]")
        print(f"  Color format: RGB888 (will convert to BGR for OpenCV)")

        # Save test frame
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out_path = os.path.join(PROJECT_ROOT, "tests", "test_frame.jpg")
        cv2.imwrite(out_path, frame_bgr)
        print(f"  Saved test frame: {out_path}")

        picam2.stop()
        print("  ✓ Direct picamera2 test PASSED")
        return True

    except Exception as e:
        print(f"  ✗ Direct picamera2 test FAILED: {e}")
        return False


def test_camera_stream(force_opencv=False, num_frames=30):
    """Test using the VisioNull CameraStream wrapper."""
    print("\n── CameraStream Wrapper Test ──")

    from src.camera_stream import CameraStream, PICAMERA2_AVAILABLE

    print(f"  picamera2 available: {PICAMERA2_AVAILABLE}")
    print(f"  Force OpenCV: {force_opencv}")

    camera = CameraStream(
        camera_index=0,
        frame_width=640,
        frame_height=480,
        fps=15,
        force_opencv=force_opencv
    )

    if not camera.start():
        print("  ✗ Failed to start camera")
        return False

    print(f"  Camera started: {type(camera).__name__}")

    frame_times = []
    first_frame = None

    print(f"  Capturing {num_frames} frames...")

    for i in range(num_frames):
        t0 = time.time()
        frame = camera.get_frame()
        t1 = time.time()

        if frame is None:
            print(f"  ✗ Frame {i+1} returned None")
            camera.stop()
            return False

        frame_times.append(t1 - t0)

        if first_frame is None:
            first_frame = frame.copy()
            print(f"  First frame — shape: {frame.shape}, dtype: {frame.dtype}")

        # Show live preview
        info = f"Frame {i+1}/{num_frames} | {1.0/(t1-t0):.1f} FPS"
        display = frame.copy()
        cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, "Press 'q' to quit early", (10, display.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("Stage 1: Camera Test", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("  Quit early by user")
            break

    camera.stop()
    cv2.destroyAllWindows()

    # Statistics
    if frame_times:
        avg_time = sum(frame_times) / len(frame_times)
        avg_fps = 1.0 / avg_time if avg_time > 0 else 0
        min_fps = 1.0 / max(frame_times) if max(frame_times) > 0 else 0
        max_fps = 1.0 / min(frame_times) if min(frame_times) > 0 else 0

        print(f"\n  Results:")
        print(f"    Frames captured: {len(frame_times)}")
        print(f"    Average FPS: {avg_fps:.1f}")
        print(f"    FPS range: [{min_fps:.1f}, {max_fps:.1f}]")
        print(f"    Avg frame time: {avg_time*1000:.1f} ms")

    # Save test frame
    if first_frame is not None:
        out_path = os.path.join(PROJECT_ROOT, "tests", "test_frame.jpg")
        cv2.imwrite(out_path, first_frame)
        print(f"    Saved: {out_path}")

    print("  ✓ CameraStream test PASSED")
    return True


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Camera Capture Test")
    parser.add_argument("--opencv", action="store_true", help="Force OpenCV fallback (skip picamera2)")
    parser.add_argument("--frames", type=int, default=30, help="Number of frames to capture (default: 30)")
    parser.add_argument("--skip-direct", action="store_true", help="Skip direct picamera2 test")
    args = parser.parse_args()

    print("=" * 50)
    print("  VisioNull — Stage 1: Camera Capture")
    print("=" * 50)

    # Test 1: Direct picamera2 (unless skipped or forcing OpenCV)
    if not args.opencv and not args.skip_direct:
        test_picamera2_direct()

    # Test 2: CameraStream wrapper
    success = test_camera_stream(force_opencv=args.opencv, num_frames=args.frames)

    if success:
        print(f"\n{'=' * 50}")
        print("  Stage 1 PASSED — Camera is working")
        print("  Proceed to Stage 2 (dataset) or Stage 3 (pose)")
        print(f"{'=' * 50}\n")
    else:
        print(f"\n{'=' * 50}")
        print("  Stage 1 FAILED — Camera issues detected")
        print("  Check: ribbon cable, rpicam-hello --list-cameras")
        print(f"{'=' * 50}\n")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
