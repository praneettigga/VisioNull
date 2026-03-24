"""
Laptop Dashboard — Flask routes

Endpoints:
    POST /webhook           — Receive fall notifications from RPi
    GET  /api/events        — List fall events (JSON)
    POST /api/events/<id>/acknowledge — Mark event as seen
    GET  /                  — Dashboard page
"""

from flask import Blueprint, request, jsonify, render_template, Response
from backend.config import MAX_CLIP_SIZE_BYTES
from backend.clip_cache import get_clip_cache
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
    clip_cache = get_clip_cache()
    for event in events:
        event["has_clip"] = clip_cache.has_clip(event["id"])
    return jsonify(events)


@bp.route("/api/events/<int:event_id>/acknowledge", methods=["POST"])
def api_acknowledge(event_id: int):
    """Mark an event as acknowledged."""
    updated = acknowledge_event(event_id)
    if updated:
        return jsonify({"status": "ok"})
    return jsonify({"error": "Event not found"}), 404


@bp.route("/api/events/<int:event_id>/clip", methods=["POST"])
def api_upload_clip(event_id: int):
    """Upload an in-memory pre-fall clip for an existing event."""
    if "clip" not in request.files:
        return jsonify({"error": "Missing clip file field"}), 400

    file = request.files["clip"]
    if not file.filename:
        return jsonify({"error": "Empty clip filename"}), 400

    clip_bytes = file.read()
    if not clip_bytes:
        return jsonify({"error": "Empty clip payload"}), 400
    if len(clip_bytes) > MAX_CLIP_SIZE_BYTES:
        return jsonify({"error": "Clip exceeds maximum allowed size"}), 413

    content_type = file.mimetype or "application/octet-stream"
    if not (content_type.startswith("video/") or content_type == "application/octet-stream"):
        return jsonify({"error": "Unsupported clip content type"}), 415

    clip_cache = get_clip_cache()
    clip_cache.put(event_id=event_id, clip_bytes=clip_bytes, mime_type=content_type)

    return jsonify({"status": "ok", "event_id": event_id}), 200


@bp.route("/api/events/<int:event_id>/clip", methods=["GET"])
def api_get_clip(event_id: int):
    """Stream a cached pre-fall clip if still in memory."""
    clip_cache = get_clip_cache()
    clip = clip_cache.get(event_id)
    if clip is None:
        return jsonify({"error": "Clip not found or expired"}), 404

    return Response(clip.clip_bytes, mimetype=clip.mime_type)
