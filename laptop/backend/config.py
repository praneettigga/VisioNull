"""
Laptop Dashboard — Configuration
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Database
DB_PATH = DATA_DIR / "notifications.db"

# Flask
SECRET_KEY = os.environ.get("VISIONULL_SECRET_KEY", "visionull-dev-key-change-in-prod")
HOST = os.environ.get("VISIONULL_HOST", "0.0.0.0")
PORT = int(os.environ.get("VISIONULL_PORT", 5000))
DEBUG = os.environ.get("VISIONULL_DEBUG", "true").lower() in ("true", "1", "yes")
