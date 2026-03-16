"""
============================================================
Liver Disease Detection - Data Preprocessing
File: backend/utils/preprocessing.py
============================================================
Handles tabular data scaling, image normalization,
and data augmentation for training.
"""

import numpy as np
import pandas as pd
import cv2
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# ─── Constants ───────────────────────────────────────────────────────────────

IMAGE_SIZE = (224, 224)
SCALER_PATH = "backend/ml_models/saved_models/scaler.pkl"
LABEL_ENCODER_PATH = "backend/ml_models/saved_models/label_encoder.pkl"

# Clinical feature columns (must match form input names)
TABULAR_FEATURES = [
    'age', 'gender',
    'alt', 'ast', 'alp',
    'bilirubin_total', 'bilirubin_direct',
    'albumin', 'total_protein', 'ag_ratio'
]

CLASS_NAMES = ['Normal', 'Fatty Liver', 'Cirrhosis']


# ─── Tabular Data Preprocessing ──────────────────────────────────────────────

def preprocess_tabular_data(df: pd.DataFrame, fit=True, scaler=None):
    """
    Preprocess clinical tabular data.

    Args:
        df: DataFrame with clinical features
        fit: If True, fit scaler on data. If False, use existing scaler.
        scaler: Pre-fitted scaler (used when fit=False)

    Returns:
        X_scaled: Scaled feature array
        scaler: Fitted StandardScaler
    """
    df = df.copy()

    # Encode gender: Male=1, Female=0
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'Male': 1, 'Female': 0, 'M': 1, 'F': 0})
        df['gender'] = df['gender'].fillna(0).astype(int)

    # Select features
    available_features = [f for f in TABULAR_FEATURES if f in df.columns]
    X = df[available_features].values.astype(np.float32)

    # Handle missing values
    X = np.nan_to_num(X, nan=0.0)

    # Scale features
    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # Save scaler
        os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)
        print(f"✅ Scaler saved to {SCALER_PATH}")
    else:
        if scaler is None:
            scaler = joblib.load(SCALER_PATH)
        X_scaled = scaler.transform(X)

    return X_scaled.astype(np.float32), scaler


def preprocess_single_patient(data_dict: dict, scaler=None):
    """
    Preprocess a single patient's tabular data for inference.

    Args:
        data_dict: Dictionary with clinical values
        scaler: Pre-fitted scaler

    Returns:
        Scaled feature array of shape (1, n_features)
    """
    df = pd.DataFrame([data_dict])
    X_scaled, _ = preprocess_tabular_data(df, fit=False, scaler=scaler)
    return X_scaled  # shape: (1, n_features)


# ─── Image Preprocessing ────────────────────────────────────────────────────

def preprocess_image(image_path: str) -> np.ndarray:
    """
    Preprocess a single ultrasound image for CNN inference.

    Args:
        image_path: Path to image file

    Returns:
        Preprocessed image array of shape (1, 224, 224, 3)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at: {image_path}")

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize to model input size
    img = cv2.resize(img, IMAGE_SIZE)

    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0

    # ImageNet normalization (for pretrained ResNet50)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std

    # Add batch dimension
    img = np.expand_dims(img, axis=0)  # (1, 224, 224, 3)

    return img.astype(np.float32)


def preprocess_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess image from raw bytes (from uploaded file).

    Args:
        image_bytes: Raw image bytes

    Returns:
        Preprocessed image array of shape (1, 224, 224, 3)
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image from bytes.")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMAGE_SIZE)
    img = img.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std

    img = np.expand_dims(img, axis=0)
    return img.astype(np.float32)


# ─── Label Encoding ─────────────────────────────────────────────────────────

def encode_labels(labels, fit=True, encoder=None):
    """
    Encode string labels to integers and then to one-hot.

    Args:
        labels: List/array of string labels
        fit: Fit new encoder if True
        encoder: Pre-fitted encoder if fit=False

    Returns:
        y_encoded: One-hot encoded labels (n_samples, n_classes)
        encoder: Fitted LabelEncoder
    """
    from tensorflow.keras.utils import to_categorical

    if fit:
        encoder = LabelEncoder()
        encoder.classes_ = np.array(CLASS_NAMES)
        y_int = encoder.transform(labels)
        os.makedirs(os.path.dirname(LABEL_ENCODER_PATH), exist_ok=True)
        joblib.dump(encoder, LABEL_ENCODER_PATH)
    else:
        if encoder is None:
            encoder = joblib.load(LABEL_ENCODER_PATH)
        y_int = encoder.transform(labels)

    y_onehot = to_categorical(y_int, num_classes=len(CLASS_NAMES))
    return y_onehot, encoder


def load_scaler():
    """Load saved scaler from disk."""
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}. Train the model first.")
    return joblib.load(SCALER_PATH)


def load_label_encoder():
    """Load saved label encoder from disk."""
    if not os.path.exists(LABEL_ENCODER_PATH):
        raise FileNotFoundError(f"LabelEncoder not found at {LABEL_ENCODER_PATH}.")
    return joblib.load(LABEL_ENCODER_PATH)


# ─── Dataset Preparation ────────────────────────────────────────────────────

def prepare_dataset(csv_path: str, image_dir: str, test_size=0.2, val_size=0.1):
    """
    Load and split dataset for training.

    Args:
        csv_path: Path to CSV with tabular data
        image_dir: Directory containing liver ultrasound images
        test_size: Fraction for test set
        val_size: Fraction for validation set

    Returns:
        Dictionary with train/val/test splits for both modalities
    """
    df = pd.read_csv(csv_path)
    print(f"📂 Loaded dataset: {len(df)} samples")
    print(f"📊 Class distribution:\n{df['label'].value_counts()}")

    # Preprocess tabular
    X_tab, scaler = preprocess_tabular_data(df, fit=True)
    y, encoder = encode_labels(df['label'].values, fit=True)

    # Load images
    image_paths = [os.path.join(image_dir, fname) for fname in df['image_file']]
    images = []
    valid_indices = []

    for i, img_path in enumerate(image_paths):
        try:
            img = preprocess_image(img_path)
            images.append(img[0])  # Remove batch dim
            valid_indices.append(i)
        except Exception as e:
            print(f"⚠️  Skipping image {img_path}: {e}")

    X_img = np.array(images, dtype=np.float32)
    X_tab = X_tab[valid_indices]
    y = y[valid_indices]

    print(f"✅ Loaded {len(X_img)} valid image-tabular pairs")

    # Split data
    X_tab_train, X_tab_test, X_img_train, X_img_test, y_train, y_test = train_test_split(
        X_tab, X_img, y, test_size=test_size, random_state=42, stratify=np.argmax(y, axis=1)
    )

    val_fraction = val_size / (1 - test_size)
    X_tab_train, X_tab_val, X_img_train, X_img_val, y_train, y_val = train_test_split(
        X_tab_train, X_img_train, y_train,
        test_size=val_fraction, random_state=42, stratify=np.argmax(y_train, axis=1)
    )

    return {
        'train': {'tabular': X_tab_train, 'image': X_img_train, 'labels': y_train},
        'val': {'tabular': X_tab_val, 'image': X_img_val, 'labels': y_val},
        'test': {'tabular': X_tab_test, 'image': X_img_test, 'labels': y_test},
        'scaler': scaler,
        'encoder': encoder,
    }
