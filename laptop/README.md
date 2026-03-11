# VisioNull — Laptop Dashboard

A **Flask**-based web dashboard that receives fall-detection notifications from the Raspberry Pi over Wi-Fi and displays them in a real-time web interface.

This directory (`laptop/`) contains all laptop/server code. See [../README.md](../README.md) for the full system overview, and [../rpi/README.md](../rpi/README.md) for the Raspberry Pi setup.

---

## Features

- **Webhook endpoint** — receives HTTP POST notifications from the RPi
- **SQLite storage** — persists all fall events to disk
- **Live dashboard** — auto-refreshing web page with event cards
- **Event management** — acknowledge/dismiss individual events
- **Filtering** — show all, unacknowledged only, or acknowledged only
- **REST API** — query events programmatically via `/api/events`

---

## Prerequisites

- **Linux, macOS, or Windows** with Python 3.9+
- **~50 MB disk space**
- **Network access** to the Raspberry Pi (same Wi-Fi network)

---

## Setup

### 1. Navigate to the Laptop Directory

```bash
cd ~/VisioNull/laptop      # adjust path to match your clone location
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install flask
```

> Only Flask is required. SQLite is bundled with Python.

---

## Running the Dashboard

```bash
# Activate virtual environment
source venv/bin/activate

# Start the server
python run.py
```

Open **http://localhost:5000** in your browser to see the dashboard.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VISIONULL_HOST` | `0.0.0.0` | Bind address |
| `VISIONULL_PORT` | `5000` | Port number |
| `VISIONULL_DEBUG` | `true` | Flask debug mode |
| `VISIONULL_SECRET_KEY` | (dev key) | Flask secret key — change in production |

Example:

```bash
VISIONULL_PORT=8080 python run.py
```

---

## API Reference

### `POST /webhook`

Receives fall notifications from the RPi. Expects JSON:

```json
{
  "timestamp": "2026-02-09T14:30:45",
  "device_name": "living-room-pi",
  "device_location": "Living Room",
  "message": "Fall detected!",
  "fall_confidence": 0.85,
  "event_id": "living-room-pi-20260209143045-0001"
}
```

All fields are optional; the server stores whatever is provided.

**Response:** `201 Created`

### `GET /api/events`

Returns stored events as JSON array (newest first).

Query parameters:

| Param | Values | Description |
|-------|--------|-------------|
| `filter` | `all` (default), `unacknowledged`, `acknowledged` | Filter events |

### `POST /api/events/<id>/acknowledge`

Marks an event as acknowledged.

**Response:** `200 OK`

---

## Testing the Webhook (Without RPi)

Send a test notification with `curl`:

```bash
curl -X POST http://localhost:5000/webhook \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "2026-02-09T14:30:45",
    "device_name": "test-device",
    "device_location": "Test Room",
    "message": "Fall detected!",
    "fall_confidence": 0.92,
    "event_id": "test-001"
  }'
```

Refresh the dashboard to see the event appear.

---

## Connecting the Raspberry Pi

On the RPi, set the webhook URL to point at this laptop:

```bash
# Find your laptop's IP address
hostname -I   # on the laptop

# On the RPi, set the environment variable (or edit rpi/src/config.py)
export VISIONULL_WEBHOOK_URL="http://<LAPTOP_IP>:5000/webhook"
python -m src.main_pi
```

Make sure both devices are on the same network.

---

## Project Structure

```
laptop/
├── backend/
│   ├── __init__.py       # Package marker
│   ├── app.py            # Flask app factory
│   ├── config.py         # Configuration (DB path, host, port)
│   ├── models.py         # SQLite database helpers
│   └── routes.py         # Webhook + API + dashboard routes
├── frontend/
│   ├── templates/
│   │   ├── base.html     # HTML shell with navbar
│   │   └── dashboard.html # Dashboard page
│   └── static/
│       ├── css/
│       │   └── style.css  # Dark-theme styling
│       └── js/
│           └── dashboard.js # Auto-polling + event rendering
├── data/
│   └── notifications.db  # SQLite database (auto-created)
├── run.py                # Entry point
└── README.md             # This file
```

---

## Development Tips

### Quick Test Loop

```bash
# Terminal 1 — run the server
source venv/bin/activate
python run.py

# Terminal 2 — send test events
curl -X POST http://localhost:5000/webhook \
  -H "Content-Type: application/json" \
  -d '{"device_name":"dev","message":"Test fall","fall_confidence":0.8}'
```

### Resetting the Database

```bash
rm data/notifications.db
# Restart the server — the database is recreated automatically
```

---

## Laptop Development (RPi Code)

If you want to develop and test the RPi fall-detection pipeline on a laptop (using a webcam), see the instructions below. This applies to running the **RPi source code** on a laptop for development/tuning — not the dashboard.

### Setup

```bash
cd ~/VisioNull/rpi

# Create a virtual environment  
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Dataset Setup

Download the **CCTV Incident Dataset** for offline testing:

```bash
bash tests/download_dataset.sh
```

Or download manually from [Kaggle](https://www.kaggle.com/datasets/simuletic/cctv-incident-dataset-fall-and-lying-down-detection) and extract to `rpi/data/`.

### Staged Testing

```bash
source venv/bin/activate

python3 tests/stage0_env_check.py          # Environment
python3 tests/stage1_camera.py             # Camera
python3 tests/stage2_dataset.py            # Dataset
python3 tests/stage3_pose.py --live        # Pose (live)
python3 tests/stage3_pose.py --dataset     # Pose (offline)
python3 tests/stage4_fall_detection.py --dataset  # Fall detection
python3 tests/stage5_tune.py               # Threshold tuning
python3 tests/stage6_full_pipeline.py --live      # Full pipeline
```

### Running the Desktop App

```bash
python -m src.main               # GUI with webcam
python -m src.main --camera 1    # USB webcam
python -m src.main --no-debug    # Hide debug overlay
```

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `D` | Toggle debug metrics |
| `R` | Reset fall detector |

### Threshold Tuning

```bash
python3 tests/stage5_tune.py         # Grid search
python3 tests/stage5_tune.py --ml    # Compare ML classifiers
```

Apply the best thresholds by editing `rpi/src/config.py`.

---

## License

MIT License — see repository root.
