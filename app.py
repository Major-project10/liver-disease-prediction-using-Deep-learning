"""
============================================================
Liver Disease Detection - Main Flask Application
File: app.py
============================================================
Entry point for the Flask backend server.
Run: python app.py
"""

import os
from flask import Flask, render_template, send_from_directory, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ─── App Factory ─────────────────────────────────────────────────────────────

def create_app():
    app = Flask(
        __name__,
        template_folder='frontend/templates',
        static_folder='frontend/static'
    )

    # ── Configuration ──
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-me')
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///liver_disease.db')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))
    app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'backend/uploads')

    # ── Extensions ──
    CORS(app, origins=['http://localhost:3000', 'http://localhost:5000', '*'])

    # ── Database ──
    from backend.database.db import init_db
    init_db(app)

    # ── Register Blueprints ──
    from backend.routes.prediction_routes import prediction_bp
    from backend.routes.patient_routes import patient_bp

    app.register_blueprint(prediction_bp)
    app.register_blueprint(patient_bp)

    # ── Static Files for Explanations ──
    @app.route('/explanations/<path:filename>')
    def serve_explanation(filename):
        return send_from_directory('backend/static/explanations', filename)

    @app.route('/uploads/<path:filename>')
    def serve_upload(filename):
        return send_from_directory('backend/uploads', filename)

    # ── Frontend Routes ──
    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/predict')
    def predict_page():
        return render_template('predict.html')

    @app.route('/history')
    def history_page():
        return render_template('history.html')

    @app.route('/about')
    def about_page():
        return render_template('about.html')

    # ── Error Handlers ──
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({'error': 'Not found'}), 404

    @app.errorhandler(500)
    def server_error(e):
        return jsonify({'error': 'Internal server error'}), 500

    return app


# ─── Run ─────────────────────────────────────────────────────────────────────

app = create_app()

if __name__ == '__main__':
    debug = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    port = int(os.getenv('PORT', 5000))

    print("\n" + "="*60)
    print(" 🔬 Liver Disease Detection System")
    print(f" 🌐 Running at: http://localhost:{port}")
    print(" 📊 API Docs: http://localhost:{port}/api/health")
    print("="*60 + "\n")

    app.run(debug=debug, host='0.0.0.0', port=port)
