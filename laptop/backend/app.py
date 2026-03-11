"""
Laptop Dashboard — Flask application factory
"""

import os
from flask import Flask
from backend.config import SECRET_KEY
from backend.models import init_db


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(__file__), "..", "frontend", "templates"),
        static_folder=os.path.join(os.path.dirname(__file__), "..", "frontend", "static"),
    )
    app.secret_key = SECRET_KEY

    # Initialize database
    init_db()

    # Register routes
    from backend.routes import bp
    app.register_blueprint(bp)

    return app
