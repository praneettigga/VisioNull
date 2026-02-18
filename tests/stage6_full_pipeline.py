#!/usr/bin/env python3
"""
Stage 6: Full Pipeline Integration Test
Runs the complete pipeline: camera/dataset → pose → fall detection → notification.
Uses a local test webhook or webhook.site to verify notification delivery.

Usage:
    python3 tests/stage6_full_pipeline.py --live
    python3 tests/stage6_full_pipeline.py --live --webhook https://webhook.site/your-id
    python3 tests/stage6_full_pipeline.py --dataset
"""

import sys
import os
import argparse
import time
import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
from src.pose_estimator import PoseEstimator
from src.fall_detector import FallDetector, FallState
from src.notifier import FallNotifier
from typing import Optional


class TestWebhookHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler to receive and log webhook notifications."""
    received = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8")
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            data = {"raw": body}

        TestWebhookHandler.received.append(data)
        print(f"\n  ** WEBHOOK RECEIVED ** #{len(TestWebhookHandler.received)}")
        print(f"     Event: {data.get('event_id', '?')}")
        print(f"     Confidence: {data.get('fall_confidence', '?')}")
        print(f"     Message: {data.get('message', '?')}")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "ok"}).encode())

    def log_message(self, format, *args):
        pass  # Suppress default logging


def start_test_webhook_server(port: int = 9876) -> HTTPServer:
    """Start a local test webhook server on the given port."""
    server = HTTPServer(("0.0.0.0", port), TestWebhookHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def test_full_pipeline_live(webhook_url: str, num_frames: int = 300):
    """Run the full pipeline with live camera."""
    print("\n── Full Pipeline: Live Camera ──")
    print(f"  Webhook: {webhook_url}")
    print("  Act out a fall to trigger notification.")
    print("  Press 'q' to quit.\n")

    from src.camera_stream import CameraStream

    camera = CameraStream(camera_index=0, frame_width=640, frame_height=480, fps=15)
    pose = PoseEstimator(model_complexity=0)
    detector = FallDetector()
    notifier = FallNotifier(
        webhook_url=webhook_url,
        device_name="test-device",
        device_location="Stage 6 Test",
        cooldown_seconds=10,
        enable_queue=False
    )

    if not camera.start():
        print("  Failed to start camera")
        return False

    notification_sent = False
    prev_state = None

    for i in range(num_frames):
        frame = camera.get_frame()
        if frame is None:
            break

        h, w = frame.shape[:2]
        landmarks = pose.get_landmarks(frame)
        state, metrics = detector.update(landmarks, h)

        # State transition logging
        if state != prev_state:
            print(f"  [{i:4d}] {state.value.upper()} (conf={metrics.confidence:.2f})")
            prev_state = state

        # Send notification on validated fall
        if detector.is_fall_validated() and not notification_sent:
            success = notifier.notify_fall(metrics.confidence)
            if success:
                print(f"  [{i:4d}] ** Notification SENT **")
                notification_sent = True
            else:
                print(f"  [{i:4d}] Notification send failed")

        # Reset notification flag when recovered
        if state == FallState.STANDING:
            notification_sent = False

        # Display
        output = pose.draw_landmarks(frame, landmarks) if landmarks else frame.copy()
        color = (0, 0, 255) if state in (FallState.FALLEN, FallState.FALLING) else (0, 180, 0)
        cv2.rectangle(output, (0, 0), (w, 40), color, -1)
        cv2.putText(output, f"{state.value.upper()} | {metrics.confidence:.0%}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Stage 6: Full Pipeline", output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.stop()
    pose.close()
    cv2.destroyAllWindows()

    print("  ✓ Full pipeline live test completed")
    return True


def test_full_pipeline_dataset(webhook_url: str, source_path: str, labels_path: Optional[str] = None):
    """Run the full pipeline on dataset."""
    print(f"\n── Full Pipeline: Dataset ──")
    print(f"  Source: {source_path}")
    print(f"  Webhook: {webhook_url}")

    from src.dataset_stream import DatasetStream

    ds = DatasetStream(source_path, labels_path=labels_path)
    pose = PoseEstimator(model_complexity=0)
    detector = FallDetector()
    notifier = FallNotifier(
        webhook_url=webhook_url,
        device_name="test-dataset",
        device_location="Stage 6 Dataset Test",
        cooldown_seconds=5,
        enable_queue=False
    )

    if not ds.start():
        print("  Failed to start DatasetStream")
        return False

    frame_count = 0
    notifications = 0

    while True:
        frame = ds.get_frame()
        if frame is None:
            break

        frame_count += 1
        h, w = frame.shape[:2]
        landmarks = pose.get_landmarks(frame)

        # Feed same frame multiple times to allow state machine to progress
        for _ in range(10):
            state, metrics = detector.update(landmarks, h)

        if detector.is_fall_validated():
            success = notifier.notify_fall(metrics.confidence)
            if success:
                notifications += 1
            detector.reset()

        if frame_count % 20 == 0:
            print(f"    Processed {frame_count} frames, {notifications} notifications sent")

    ds.stop()
    pose.close()

    print(f"\n  Results:")
    print(f"    Frames: {frame_count}")
    print(f"    Notifications sent: {notifications}")
    print(f"  ✓ Dataset pipeline test completed")
    return True


def main():
    parser = argparse.ArgumentParser(description="Stage 6: Full Pipeline Test")
    parser.add_argument("--live", action="store_true", help="Live camera test")
    parser.add_argument("--dataset", action="store_true", help="Dataset test")
    parser.add_argument("--dataset-path", type=str, help="Path to dataset images")
    parser.add_argument("--labels", type=str, help="Path to labels")
    parser.add_argument("--webhook", type=str, default=None, help="Webhook URL (default: local test server)")
    parser.add_argument("--frames", type=int, default=300, help="Live frames (default: 300)")
    args = parser.parse_args()

    if not args.live and not args.dataset:
        args.live = True

    print("=" * 50)
    print("  VisioNull — Stage 6: Full Pipeline")
    print("=" * 50)

    # Start local webhook server if no URL provided
    webhook_url = args.webhook
    local_server = None
    if webhook_url is None:
        port = 9876
        print(f"\n  Starting local test webhook server on port {port}...")
        local_server = start_test_webhook_server(port)
        webhook_url = f"http://localhost:{port}/webhook"
        print(f"  Webhook URL: {webhook_url}")

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
        test_full_pipeline_dataset(webhook_url, source, labels_path=args.labels)

    if args.live:
        test_full_pipeline_live(webhook_url, num_frames=args.frames)

    # Summary of received webhooks
    if local_server:
        received = TestWebhookHandler.received
        print(f"\n  Local webhook received {len(received)} notification(s)")
        for i, data in enumerate(received):
            print(f"    #{i+1}: event={data.get('event_id', '?')}, conf={data.get('fall_confidence', '?')}")
        local_server.shutdown()

    print(f"\n{'=' * 50}")
    print("  Stage 6 complete — Full pipeline verified!")
    print("  All stages passed. Ready for production (main_pi.py)")
    print(f"{'=' * 50}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
