"""
============================================================
Liver Disease Detection - Database Initialization
File: backend/database/db.py
============================================================
"""

from flask_sqlalchemy import SQLAlchemy

# Initialize SQLAlchemy instance (imported in app.py)
db = SQLAlchemy()


def init_db(app):
    """Initialize database with the Flask app."""
    db.init_app(app)
    with app.app_context():
        db.create_all()
        print("✅ Database initialized successfully.")
