"""
Laptop Dashboard — Database models (SQLite)
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from backend.config import DB_PATH


def _get_db() -> sqlite3.Connection:
    """Get a database connection with row factory."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    """Create tables if they don't exist."""
    conn = _get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fall_events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            device_name TEXT    NOT NULL DEFAULT 'unknown',
            device_location TEXT NOT NULL DEFAULT '',
            confidence  REAL   NOT NULL DEFAULT 0.0,
            message     TEXT   NOT NULL DEFAULT '',
            event_id    TEXT   NOT NULL DEFAULT '',
            payload     TEXT   NOT NULL DEFAULT '{}',
            acknowledged INTEGER NOT NULL DEFAULT 0,
            received_at TEXT   NOT NULL DEFAULT (datetime('now'))
        )
    """)
    conn.commit()
    conn.close()


def insert_event(data: Dict[str, Any]) -> int:
    """
    Insert a fall event from the webhook payload.

    Args:
        data: JSON payload from the RPi notifier.

    Returns:
        Row ID of the inserted event.
    """
    conn = _get_db()
    cur = conn.execute(
        """
        INSERT INTO fall_events
            (timestamp, device_name, device_location, confidence, message, event_id, payload)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            data.get("timestamp", datetime.utcnow().isoformat()),
            data.get("device_name", "unknown"),
            data.get("device_location", ""),
            data.get("fall_confidence", 0.0),
            data.get("message", ""),
            data.get("event_id", ""),
            json.dumps(data),
        ),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def get_events(
    acknowledged: Optional[bool] = None,
    limit: int = 100,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """
    Retrieve fall events, newest first.

    Args:
        acknowledged: Filter by acknowledged status. None = all.
        limit: Max rows to return.
        offset: Pagination offset.

    Returns:
        List of event dicts.
    """
    conn = _get_db()
    query = "SELECT * FROM fall_events"
    params: list = []

    if acknowledged is not None:
        query += " WHERE acknowledged = ?"
        params.append(int(acknowledged))

    query += " ORDER BY id DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def acknowledge_event(event_id: int) -> bool:
    """
    Mark an event as acknowledged.

    Args:
        event_id: Database row ID.

    Returns:
        True if a row was updated.
    """
    conn = _get_db()
    cur = conn.execute(
        "UPDATE fall_events SET acknowledged = 1 WHERE id = ?", (event_id,)
    )
    conn.commit()
    updated = cur.rowcount > 0
    conn.close()
    return updated
