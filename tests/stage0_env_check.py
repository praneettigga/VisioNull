#!/usr/bin/env python3
"""
Stage 0: Environment Verification
Checks that all required tools, libraries, and files are present.
Run this FIRST before any other stage.

Usage:
    python3 tests/stage0_env_check.py
"""

import sys
import os
import subprocess
import shutil

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Colors for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

results = []


def check(name: str, passed: bool, detail: str = ""):
    """Record a check result."""
    status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
    results.append((name, passed, detail))
    msg = f"  [{status}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return passed


def section(title: str):
    print(f"\n{CYAN}{BOLD}{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}{RESET}")


def main():
    print(f"\n{BOLD}{'=' * 50}")
    print("  VisioNull — Stage 0: Environment Check")
    print(f"{'=' * 50}{RESET}")

    # ── OS & Platform ──
    section("OS & Platform")

    # Check OS info
    os_info = "unknown"
    if os.path.exists("/etc/os-release"):
        with open("/etc/os-release") as f:
            for line in f:
                if line.startswith("PRETTY_NAME="):
                    os_info = line.split("=", 1)[1].strip().strip('"')
                    break
    check("OS detected", True, os_info)

    # Check if on Raspberry Pi
    is_pi = os.path.exists("/proc/device-tree/model")
    pi_model = ""
    if is_pi:
        with open("/proc/device-tree/model") as f:
            pi_model = f.read().strip().rstrip('\x00')
    check("Raspberry Pi", is_pi, pi_model if is_pi else "Not a Raspberry Pi (some tests may be skipped)")

    # Check Python version
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    check("Python >= 3.9", sys.version_info >= (3, 9), py_ver)

    # ── Camera CLI Tools ──
    section("Camera CLI Tools")

    # Check for rpicam-hello (Trixie/Bookworm) or libcamera-hello (Bullseye)
    rpicam_hello = shutil.which("rpicam-hello")
    libcamera_hello = shutil.which("libcamera-hello")
    cam_tool = rpicam_hello or libcamera_hello
    cam_tool_name = "rpicam-hello" if rpicam_hello else ("libcamera-hello" if libcamera_hello else "none")
    check("Camera CLI tool", cam_tool is not None, cam_tool_name)

    # List cameras if tool is available
    if cam_tool:
        try:
            result = subprocess.run(
                [cam_tool, "--list-cameras"],
                capture_output=True, text=True, timeout=10
            )
            output = result.stdout + result.stderr
            has_camera = any(kw in output.lower() for kw in ["imx", "ov5647", "available camera", "camera"])
            # Extract camera info
            camera_lines = [l.strip() for l in output.split('\n') if l.strip() and ('imx' in l.lower() or 'ov' in l.lower() or 'camera' in l.lower())]
            camera_detail = camera_lines[0] if camera_lines else output.strip()[:80]
            check("Camera detected", has_camera, camera_detail)
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            check("Camera detected", False, str(e))
    else:
        check("Camera detected", False, "No camera CLI tool found")

    # ── Python Libraries ──
    section("Python Libraries")

    # Check picamera2
    try:
        from picamera2 import Picamera2
        import picamera2
        ver = getattr(picamera2, '__version__', 'unknown')
        check("picamera2", True, f"v{ver}")
    except ImportError as e:
        check("picamera2", False, f"Not importable: {e}")

    # Check libcamera Python bindings
    try:
        from libcamera import controls
        check("libcamera (Python)", True)
    except ImportError as e:
        check("libcamera (Python)", False, f"Not importable: {e}")

    # Check OpenCV
    try:
        import cv2
        check("OpenCV (cv2)", True, f"v{cv2.__version__}")
    except ImportError as e:
        check("OpenCV (cv2)", False, str(e))

    # Check NumPy
    try:
        import numpy as np
        check("NumPy", True, f"v{np.__version__}")
    except ImportError as e:
        check("NumPy", False, str(e))

    # Check MediaPipe
    try:
        import mediapipe as mp
        check("MediaPipe", True, f"v{mp.__version__}")
    except ImportError as e:
        check("MediaPipe", False, str(e))

    # Check scikit-learn (optional)
    try:
        import sklearn
        check("scikit-learn (optional)", True, f"v{sklearn.__version__}")
    except ImportError:
        check("scikit-learn (optional)", False, "Not installed (needed for Stage 5 threshold tuning)")

    # ── Model File ──
    section("Model & Data Files")

    model_path = os.path.join(PROJECT_ROOT, "pose_landmarker_lite.task")
    model_exists = os.path.exists(model_path)
    model_size = ""
    if model_exists:
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        model_size = f"{size_mb:.1f} MB"
    check("Pose model file", model_exists, model_size if model_exists else f"Not found at {model_path}")

    # Check data directory
    data_dir = os.path.join(PROJECT_ROOT, "data")
    check("data/ directory", os.path.isdir(data_dir), data_dir if os.path.isdir(data_dir) else "Missing (run: mkdir data)")

    # Check for dataset
    dataset_dir = os.path.join(data_dir, "cctv-incident")
    if os.path.isdir(dataset_dir):
        # Count images
        img_count = len([f for f in os.listdir(os.path.join(dataset_dir, "images")) if f.endswith(('.jpg', '.png'))]) if os.path.isdir(os.path.join(dataset_dir, "images")) else 0
        check("CCTV dataset", True, f"{img_count} images")
    else:
        check("CCTV dataset (optional)", False, "Not downloaded yet (run: bash tests/download_dataset.sh)")

    # ── VisioNull Modules ──
    section("VisioNull Modules")

    modules = [
        ("src.config", "Configuration"),
        ("src.camera_stream", "Camera Stream"),
        ("src.pose_estimator", "Pose Estimator"),
        ("src.fall_detector", "Fall Detector"),
        ("src.notifier", "Notifier"),
    ]

    for module_name, display_name in modules:
        try:
            __import__(module_name)
            check(display_name, True, f"import {module_name}")
        except Exception as e:
            check(display_name, False, f"{e}")

    # ── Summary ──
    total = len(results)
    passed = sum(1 for _, p, _ in results if p)
    failed = total - passed

    print(f"\n{BOLD}{'=' * 50}")
    print(f"  Results: {GREEN}{passed} passed{RESET}{BOLD}, {RED if failed else GREEN}{failed} failed{RESET}{BOLD}, {total} total")

    if failed == 0:
        print(f"  {GREEN}All checks passed! Ready for Stage 1.{RESET}")
    else:
        print(f"\n  {YELLOW}Failed checks:{RESET}")
        for name, passed_flag, detail in results:
            if not passed_flag:
                print(f"    {RED}✗{RESET} {name}: {detail}")
        print(f"\n  Fix the above issues before proceeding.")

    print(f"{BOLD}{'=' * 50}{RESET}\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
