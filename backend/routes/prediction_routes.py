"""
============================================================
Liver Disease Detection - Prediction API Routes
File: backend/routes/prediction_routes.py
============================================================
REST API endpoints for making liver disease predictions.
"""

import os
import uuid
import json
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename

from backend.ml_models.predictor import (
    predict_tabular_only, predict_image_only,
    predict_fusion, predict_auto, ModelRegistry
)

prediction_bp = Blueprint('prediction', __name__, url_prefix='/api')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}
UPLOAD_DIR = "backend/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def save_uploaded_image(file) -> str:
    """Save uploaded image and return path."""
    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    save_path = os.path.join(UPLOAD_DIR, unique_name)
    file.save(save_path)
    return save_path


# ─── Health Check ─────────────────────────────────────────────────────────────

@prediction_bp.route('/health', methods=['GET'])
def health_check():
    """System health check and model availability."""
    availability = ModelRegistry.check_models_available()
    return jsonify({
        'status': 'running',
        'timestamp': datetime.utcnow().isoformat(),
        'models': availability,
        'version': '1.0.0'
    }), 200


# ─── Tabular-Only Prediction (ANN) ────────────────────────────────────────────

@prediction_bp.route('/predict/tabular', methods=['POST'])
def predict_tabular():
    """
    Predict liver condition from clinical lab values only.

    Request (JSON):
    {
        "age": 45, "gender": "Male",
        "alt": 55, "ast": 48, "alp": 120,
        "bilirubin_total": 1.8, "bilirubin_direct": 0.6,
        "albumin": 3.5, "total_protein": 6.8, "ag_ratio": 1.1
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        # Validate required fields
        required = ['age', 'gender', 'alt', 'ast', 'alp',
                    'bilirubin_total', 'bilirubin_direct',
                    'albumin', 'total_protein', 'ag_ratio']
        missing = [f for f in required if f not in data]
        if missing:
            return jsonify({'error': f'Missing fields: {missing}'}), 400

        result = predict_tabular_only(data)

        return jsonify({
            'success': True,
            'prediction': result,
            'input_data': data,
            'timestamp': datetime.utcnow().isoformat()
        }), 200

    except FileNotFoundError as e:
        return jsonify({'error': str(e), 'hint': 'Run training first: python -m backend.ml_models.train'}), 503
    except Exception as e:
        current_app.logger.error(f"Tabular prediction error: {e}")
        return jsonify({'error': str(e)}), 500


# ─── Image-Only Prediction (CNN) ──────────────────────────────────────────────

@prediction_bp.route('/predict/image', methods=['POST'])
def predict_image():
    """
    Predict liver condition from ultrasound image only.

    Request (multipart/form-data):
    - image: Image file
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid or missing image file'}), 400

        # Save image
        image_path = save_uploaded_image(file)

        result = predict_image_only(image_path=image_path)

        return jsonify({
            'success': True,
            'prediction': result,
            'image_path': image_path,
            'timestamp': datetime.utcnow().isoformat()
        }), 200

    except FileNotFoundError as e:
        return jsonify({'error': str(e), 'hint': 'Run training first'}), 503
    except Exception as e:
        current_app.logger.error(f"Image prediction error: {e}")
        return jsonify({'error': str(e)}), 500


# ─── Fusion Prediction (ANN + CNN + Explainability) ──────────────────────────

@prediction_bp.route('/predict/fusion', methods=['POST'])
def predict_fusion_endpoint():
    """
    Full multi-modal prediction with SHAP + Grad-CAM explanations.

    Request (multipart/form-data):
    - image: Ultrasound image file
    - data: JSON string with clinical values
    - patient_name: Optional
    - generate_explanation: true/false (default true)
    """
    try:
        # Parse clinical data from form
        data_str = request.form.get('data', '{}')
        clinical_data = json.loads(data_str)

        if not clinical_data:
            return jsonify({'error': 'No clinical data provided'}), 400

        # Handle image
        image_path = None
        if 'image' in request.files:
            file = request.files['image']
            if file.filename and allowed_file(file.filename):
                image_path = save_uploaded_image(file)

        generate_explanation = request.form.get('generate_explanation', 'true').lower() == 'true'

        # Make prediction
        if image_path:
            result = predict_fusion(clinical_data, image_path=image_path)
            result['image_path'] = image_path
        else:
            result = predict_tabular_only(clinical_data)

        # Generate explanations
        explanation = {}

        if generate_explanation:
            try:
                # SHAP explanation for tabular
                import numpy as np
                import joblib
                from backend.ml_models.predictor import ModelRegistry, SCALER_PATH
                from backend.utils.preprocessing import preprocess_single_patient, TABULAR_FEATURES
                from backend.ml_models.train import generate_synthetic_data, CLASS_NAMES
                from backend.explainability.shap_explainer import generate_shap_explanation_simple

                ann_model = ModelRegistry.get_ann_model()
                scaler = ModelRegistry.get_scaler()
                X_sample = preprocess_single_patient(clinical_data, scaler=scaler)

                # Generate background data
                bg_df = generate_synthetic_data(n_samples=200, save_csv=False)
                from backend.utils.preprocessing import preprocess_tabular_data
                X_bg, _ = preprocess_tabular_data(bg_df, fit=False, scaler=scaler)

                shap_result = generate_shap_explanation_simple(
                    ann_model, X_sample, X_bg,
                    predicted_class=result['predicted_class']
                )
                explanation['shap'] = {
                    'plot_path': shap_result.get('plot_path', ''),
                    'top_features': shap_result.get('top_features', []),
                    'feature_importance': shap_result.get('feature_importance', {})
                }
            except Exception as shap_err:
                current_app.logger.warning(f"SHAP failed: {shap_err}")
                explanation['shap'] = {'error': str(shap_err)}

            if image_path:
                try:
                    from backend.explainability.gradcam import generate_gradcam_plot
                    cnn_model = ModelRegistry.get_cnn_model()
                    gradcam_path = generate_gradcam_plot(
                        cnn_model, image_path,
                        predicted_class=result['predicted_class']
                    )
                    explanation['gradcam'] = {'plot_path': gradcam_path}
                except Exception as gcam_err:
                    current_app.logger.warning(f"Grad-CAM failed: {gcam_err}")
                    explanation['gradcam'] = {'error': str(gcam_err)}

        result['explanation'] = explanation

        # ── Save to DB ──
        try:
            from backend.database.db import db
            from backend.database.models import Patient, Prediction, LabValues

            # Create/find patient
            patient_name = request.form.get('patient_name', 'Anonymous')
            patient_age = int(clinical_data.get('age', 0))
            patient_gender = clinical_data.get('gender', 'Unknown')

            patient = Patient(name=patient_name, age=patient_age, gender=patient_gender)
            db.session.add(patient)
            db.session.flush()

            # Create prediction record
            probs = result.get('probabilities', {})
            prediction_record = Prediction(
                patient_id=patient.id,
                predicted_class=result['predicted_class'],
                confidence=result['confidence'],
                prob_normal=probs.get('Normal', 0.0),
                prob_fatty_liver=probs.get('Fatty Liver', 0.0),
                prob_cirrhosis=probs.get('Cirrhosis', 0.0),
                model_type=result.get('model_type', 'fusion'),
                image_path=image_path,
                shap_plot_path=explanation.get('shap', {}).get('plot_path', ''),
                gradcam_plot_path=explanation.get('gradcam', {}).get('plot_path', ''),
            )
            db.session.add(prediction_record)
            db.session.flush()

            # Save lab values
            lab = LabValues(
                prediction_id=prediction_record.id,
                alt=float(clinical_data.get('alt', 0)),
                ast=float(clinical_data.get('ast', 0)),
                alp=float(clinical_data.get('alp', 0)),
                bilirubin_total=float(clinical_data.get('bilirubin_total', 0)),
                bilirubin_direct=float(clinical_data.get('bilirubin_direct', 0)),
                albumin=float(clinical_data.get('albumin', 0)),
                total_protein=float(clinical_data.get('total_protein', 0)),
                ag_ratio=float(clinical_data.get('ag_ratio', 0)),
            )
            db.session.add(lab)
            db.session.commit()

            result['record_id'] = prediction_record.id
            result['patient_id'] = patient.id

        except Exception as db_err:
            current_app.logger.warning(f"DB save failed: {db_err}")

        return jsonify({
            'success': True,
            'prediction': result,
            'clinical_data': clinical_data,
            'timestamp': datetime.utcnow().isoformat()
        }), 200

    except FileNotFoundError as e:
        return jsonify({'error': str(e), 'hint': 'Run training first'}), 503
    except Exception as e:
        current_app.logger.error(f"Fusion prediction error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# ─── Model Status ─────────────────────────────────────────────────────────────

@prediction_bp.route('/models/status', methods=['GET'])
def models_status():
    """Check which models are loaded/available."""
    return jsonify(ModelRegistry.check_models_available()), 200

# ─── History Status ─────────────────────────────────────────────────────────────

@prediction_bp.route('/history', methods=['GET'])
def get_history():
    try:
        from backend.database.models import Patient, Prediction, LabValues
        from backend.database.db import db

        predictions = (
            db.session.query(Prediction)
            .order_by(Prediction.created_at.desc())
            .all()
        )

        data = []

        for p in predictions:
            patient = p.patient
            data.append({
                "id": p.id,
                "patient_name": patient.name,
                "age": patient.age,
                "gender": patient.gender,
                "predicted_class": p.predicted_class,
                "confidence": p.confidence,
                "model_type": p.model_type,
                "date": p.created_at.isoformat()
            })

        return jsonify({
            "success": True,
            "records": data,
            "total": len(data)
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500