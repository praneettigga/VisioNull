# VisioNull — Real-Time Fall Detection System

An edge-computing fall detection system. A **Raspberry Pi** with a camera detects falls in real time using MediaPipe Pose, keeps a rolling **15-second in-memory pre-fall buffer**, and sends alerts over WiFi to a **Laptop** running a Flask dashboard.

```
┌──────────────────┐   1) POST /webhook JSON    ┌──────────────────┐
│   Raspberry Pi   │  ─────────────────────────▶ │  Laptop (Flask)  │
│  Camera + Pose   │                             │    Dashboard     │
│  15s RAM Buffer  │   2) POST /api/events/:id  │  View alerts +   │
│  Fall Detection  │  ──────── clip upload ───▶ │  pre-fall clips  │
└──────────────────┘                             └──────────────────┘
```

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.18-orange.svg)
![Flask](https://img.shields.io/badge/Flask-3.0+-purple.svg)

## Project Structure

```
VisioNull/
├── rpi/                    # Raspberry Pi edge device
│   ├── src/
│   │   ├── pipeline/       # Camera, pose estimation, fall detection, pre-fall buffer
│   │   ├── notification/   # HTTP webhook notifier with offline queue
│   │   ├── config.py       # All RPi settings (thresholds, webhook URL, etc.)
│   │   ├── main.py         # Desktop GUI app (for testing with display)
│   │   └── main_pi.py      # Headless production entry point
│   ├── model/              # MediaPipe pose model (.task file)
│   ├── data/               # Datasets for offline evaluation
│   ├── tests/              # Staged test scripts (stage0–stage6)
│   ├── logs/               # Runtime logs
│   ├── requirements.txt
│   ├── setup.sh            # Automated Pi setup script
│   └── visionull.service   # systemd unit for auto-start
│
├── laptop/                 # Laptop dashboard (Flask)
│   ├── backend/
│   │   ├── app.py          # Flask app factory
│   │   ├── clip_cache.py   # RAM-only transient clip cache with TTL
│   │   ├── config.py       # Server settings
│   │   ├── models.py       # SQLite database (fall_events table)
│   │   └── routes.py       # /webhook + /api/events + clip upload/retrieval
│   ├── frontend/
│   │   ├── templates/      # Jinja2 HTML (dashboard)
│   │   └── static/         # CSS + JS (auto-polling dashboard)
│   ├── data/               # SQLite DB created at runtime
│   ├── requirements.txt
│   └── run.py              # Entry point: python run.py
│
└── README.md               # This file
```

## Quick Start

### 1. Laptop — Start the Dashboard

```bash
cd laptop
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python run.py
```

Open **http://localhost:5000** in your browser. The dashboard polls for new events automatically.

### 2. Raspberry Pi — Setup & Run

See [rpi/README.md](rpi/README.md) for full setup instructions (camera, Python 3.12, MediaPipe, staged testing).

**Quick test after setup:**
```bash
cd rpi
source venv/bin/activate
export VISIONULL_WEBHOOK_URL="http://<laptop-ip>:5000/webhook"
python -m src.main_pi
```

### 3. Test End-to-End

Trigger a fall in front of the Pi camera. Within seconds a notification card should appear on the laptop dashboard.
When available, the card also shows a **View pre-fall clip** action.

Or test the webhook manually:
```bash
curl -X POST http://localhost:5000/webhook \
  -H "Content-Type: application/json" \
  -d '{"timestamp":"2026-02-23T12:00:00","device_name":"test-pi","device_location":"Lab","message":"FALL DETECTED","fall_confidence":0.95,"event_id":"test-001"}'
```

## Event Delivery Contract

### Step 1: Event metadata (`POST /webhook`)

The RPi sends this JSON first:

```json
{
  "timestamp": "2026-02-23T14:30:45",
  "device_name": "living-room-pi",
  "device_location": "Living Room",
  "message": "FALL DETECTED - Immediate attention required!",
  "fall_confidence": 0.95,
  "event_id": "living-room-pi-20260223143045-0001"
}
```

### Step 2: Clip upload (`POST /api/events/<id>/clip`)

After receiving the row `id` from `/webhook`, the RPi uploads a pre-fall clip to:

- `/api/events/<id>/clip` (multipart form-data)
- Form field name: `clip`
- Content type: `video/mp4`

### Clip retention policy

- **RPi:** clip frames are buffered in memory only; no video files are written to disk.
- **Laptop:** uploaded clips are stored in RAM only with TTL expiry (`VISIONULL_CLIP_TTL_SECONDS`).
- Expired clips are not retrievable (`404`).

## Documentation

| Guide | Description |
|-------|-------------|
| [rpi/README.md](rpi/README.md) | Full RPi setup, staged testing, production deployment |
| [laptop/README.md](laptop/README.md) | Laptop dashboard setup and usage |

## License

MIT

