"""
============================================================
Liver Disease Detection - Prediction Engine
File: backend/ml_models/predictor.py
============================================================
Handles model loading and inference for all three model types:
  - ANN (tabular only)
  - CNN (image only)
  - Fusion (tabular + image)
"""

import os
import numpy as np
import joblib
from tensorflow import keras

from backend.utils.preprocessing import (
    preprocess_single_patient,
    preprocess_image,
    preprocess_image_from_bytes,
    CLASS_NAMES
)

# ─── Paths ───────────────────────────────────────────────────────────────────

SAVED_MODELS_DIR = "backend/ml_models/saved_models"
ANN_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "ann_model.keras")
CNN_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "cnn_model.keras")
FUSION_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "fusion_model.keras")
SCALER_PATH = os.path.join(SAVED_MODELS_DIR, "scaler.pkl")
LABEL_ENCODER_PATH = os.path.join(SAVED_MODELS_DIR, "label_encoder.pkl")


# ─── Model Registry (Singleton) ──────────────────────────────────────────────

class ModelRegistry:
    """Lazy-loaded model registry (loads models only when needed)."""

    _ann_model = None
    _cnn_model = None
    _fusion_model = None
    _scaler = None
    _label_encoder = None

    @classmethod
    def get_scaler(cls):
        if cls._scaler is None:
            if not os.path.exists(SCALER_PATH):
                raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}. Run training first.")
            cls._scaler = joblib.load(SCALER_PATH)
        return cls._scaler

    @classmethod
    def get_label_encoder(cls):
        if cls._label_encoder is None:
            if not os.path.exists(LABEL_ENCODER_PATH):
                raise FileNotFoundError(f"LabelEncoder not found. Run training first.")
            cls._label_encoder = joblib.load(LABEL_ENCODER_PATH)
        return cls._label_encoder

    @classmethod
    def get_ann_model(cls):
        if cls._ann_model is None:
            if not os.path.exists(ANN_MODEL_PATH):
                raise FileNotFoundError(f"ANN model not found: {ANN_MODEL_PATH}. Run training first.")
            cls._ann_model = keras.models.load_model(ANN_MODEL_PATH)
            print("✅ ANN model loaded.")
        return cls._ann_model

    @classmethod
    def get_cnn_model(cls):
        if cls._cnn_model is None:
            if not os.path.exists(CNN_MODEL_PATH):
                raise FileNotFoundError(f"CNN model not found: {CNN_MODEL_PATH}. Run training first.")
            cls._cnn_model = keras.models.load_model(CNN_MODEL_PATH)
            print("✅ CNN model loaded.")
        return cls._cnn_model

    @classmethod
    def get_fusion_model(cls):
        if cls._fusion_model is None:
            if not os.path.exists(FUSION_MODEL_PATH):
                raise FileNotFoundError(f"Fusion model not found: {FUSION_MODEL_PATH}. Run training first.")
            cls._fusion_model = keras.models.load_model(FUSION_MODEL_PATH)
            print("✅ Fusion model loaded.")
        return cls._fusion_model

    @classmethod
    def check_models_available(cls):
        """Return dict of which models are available."""
        return {
            'ann': os.path.exists(ANN_MODEL_PATH),
            'cnn': os.path.exists(CNN_MODEL_PATH),
            'fusion': os.path.exists(FUSION_MODEL_PATH),
            'scaler': os.path.exists(SCALER_PATH),
            'label_encoder': os.path.exists(LABEL_ENCODER_PATH),
        }


# ─── Prediction Functions ────────────────────────────────────────────────────

def format_prediction_result(probs: np.ndarray, model_type: str) -> dict:
    """
    Format raw probabilities into structured result dict.

    Args:
        probs: Array of shape (3,) with class probabilities
        model_type: 'ann', 'cnn', or 'fusion'

    Returns:
        Structured prediction dict
    """
    pred_class_idx = np.argmax(probs)
    pred_class_name = CLASS_NAMES[pred_class_idx]
    confidence = float(probs[pred_class_idx])

    # Risk level based on prediction
    risk_map = {
        'Normal': ('Low', '#2ecc71'),
        'Fatty Liver': ('Moderate', '#f39c12'),
        'Cirrhosis': ('High', '#e74c3c')
    }
    risk_level, risk_color = risk_map.get(pred_class_name, ('Unknown', '#95a5a6'))

    return {
        'predicted_class': pred_class_name,
        'confidence': confidence,
        'confidence_pct': f"{confidence * 100:.1f}%",
        'risk_level': risk_level,
        'risk_color': risk_color,
        'model_type': model_type,
        'probabilities': {
            cls: float(prob) for cls, prob in zip(CLASS_NAMES, probs)
        },
        'probabilities_pct': {
            cls: f"{float(prob) * 100:.1f}%" for cls, prob in zip(CLASS_NAMES, probs)
        }
    }


def predict_tabular_only(clinical_data: dict) -> dict:
    """
    Predict liver condition using tabular data only (ANN).

    Args:
        clinical_data: Dict with clinical lab values

    Returns:
        Prediction result dict
    """
    ann_model = ModelRegistry.get_ann_model()
    scaler = ModelRegistry.get_scaler()

    X = preprocess_single_patient(clinical_data, scaler=scaler)
    probs = ann_model.predict(X, verbose=0)[0]
    return format_prediction_result(probs, model_type='ann')


def predict_image_only(image_path: str = None, image_bytes: bytes = None) -> dict:
    """
    Predict liver condition using ultrasound image only (CNN).

    Args:
        image_path: Path to image file (or use image_bytes)
        image_bytes: Raw image bytes

    Returns:
        Prediction result dict
    """
    cnn_model = ModelRegistry.get_cnn_model()

    if image_path:
        X_img = preprocess_image(image_path)
    elif image_bytes:
        X_img = preprocess_image_from_bytes(image_bytes)
    else:
        raise ValueError("Either image_path or image_bytes must be provided.")

    probs = cnn_model.predict(X_img, verbose=0)[0]
    return format_prediction_result(probs, model_type='cnn')


def predict_fusion(clinical_data: dict, image_path: str = None, image_bytes: bytes = None) -> dict:
    """
    Predict liver condition using multi-modal fusion (ANN + CNN).

    Args:
        clinical_data: Dict with clinical lab values
        image_path: Path to ultrasound image (or use image_bytes)
        image_bytes: Raw image bytes

    Returns:
        Prediction result dict
    """
    fusion_model = ModelRegistry.get_fusion_model()
    scaler = ModelRegistry.get_scaler()

    X_tab = preprocess_single_patient(clinical_data, scaler=scaler)

    if image_path:
        X_img = preprocess_image(image_path)
    elif image_bytes:
        X_img = preprocess_image_from_bytes(image_bytes)
    else:
        raise ValueError("Either image_path or image_bytes must be provided.")

    probs = fusion_model.predict([X_tab, X_img], verbose=0)[0]
    return format_prediction_result(probs, model_type='fusion')


def predict_auto(clinical_data: dict, image_path: str = None, image_bytes: bytes = None) -> dict:
    """
    Automatically choose best available model:
    1. Fusion (if both image + tabular)
    2. ANN (if tabular only)
    3. CNN (if image only)
    """
    availability = ModelRegistry.check_models_available()
    has_image = image_path is not None or image_bytes is not None
    has_tabular = clinical_data and len(clinical_data) > 0

    if has_image and has_tabular and availability.get('fusion'):
        try:
            return predict_fusion(clinical_data, image_path, image_bytes)
        except Exception as e:
            print(f"⚠️  Fusion failed: {e}. Falling back to ANN.")

    if has_tabular and availability.get('ann'):
        return predict_tabular_only(clinical_data)

    if has_image and availability.get('cnn'):
        return predict_image_only(image_path, image_bytes)

    raise RuntimeError("No models available. Please run training first.")
