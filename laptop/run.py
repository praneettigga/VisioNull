#!/usr/bin/env python3
"""
VisioNull — Laptop Dashboard Entry Point

Usage:
    python run.py
    
Then open http://localhost:5000 in your browser.
"""

from dotenv import load_dotenv
load_dotenv()

from backend.app import create_app
from backend.config import HOST, PORT, DEBUG

app = create_app()

if __name__ == "__main__":
    print()
    print("=" * 50)
    print("  VisioNull — Laptop Dashboard")
    print("=" * 50)
    print(f"  URL: http://{HOST}:{PORT}")
    print(f"  Webhook endpoint: http://{HOST}:{PORT}/webhook")
    print(f"  Debug: {DEBUG}")
    print("=" * 50)
    print()
    app.run(host=HOST, port=PORT, debug=DEBUG)
