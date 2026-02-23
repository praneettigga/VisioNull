"""
Laptop Dashboard — Flask routes

Endpoints:
    POST /webhook           — Receive fall notifications from RPi
    GET  /api/events        — List fall events (JSON)
    POST /api/events/<id>/acknowledge — Mark event as seen
    GET  /                  — Dashboard page
"""

from flask import Blueprint, request, jsonify, render_template
from backend.models import insert_event, get_events, acknowledge_event

bp = Blueprint("main", __name__)


@bp.route("/")
def dashboard():
    """Render the dashboard page."""
    return render_template("dashboard.html")


@bp.route("/webhook", methods=["POST"])
def webhook():
    """
    Receive a fall notification from the RPi.

    Expected JSON payload (from RPi notifier.py FallNotification):
    {
        "timestamp": "2026-02-23T12:34:56.789",
        "device_name": "living-room-pi",
        "device_location": "Living Room",
        "message": "FALL DETECTED - Immediate attention required!",
        "fall_confidence": 0.95,
        "event_id": "living-room-pi-20260223123456-0001"
    }
    """
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Invalid JSON body"}), 400

    row_id = insert_event(data)
    print(
        f"[WEBHOOK] Fall event #{row_id} from "
        f"{data.get('device_name', '?')} — "
        f"confidence {data.get('fall_confidence', '?')}"
    )
    return jsonify({"status": "ok", "id": row_id}), 200


@bp.route("/api/events", methods=["GET"])
def api_events():
    """
    Return fall events as JSON.

    Query params:
        acknowledged — 'true', 'false', or omit for all
        limit        — max rows (default 100)
        offset       — pagination offset (default 0)
    """
    ack_param = request.args.get("acknowledged")
    acknowledged = None
    if ack_param is not None:
        acknowledged = ack_param.lower() in ("true", "1", "yes")

    limit = request.args.get("limit", 100, type=int)
    offset = request.args.get("offset", 0, type=int)

    events = get_events(acknowledged=acknowledged, limit=limit, offset=offset)
    return jsonify(events)


@bp.route("/api/events/<int:event_id>/acknowledge", methods=["POST"])
def api_acknowledge(event_id: int):
    """Mark an event as acknowledged."""
    updated = acknowledge_event(event_id)
    if updated:
        return jsonify({"status": "ok"})
    return jsonify({"error": "Event not found"}), 404
