"""
============================================================
Liver Disease Detection - Model Training Script
File: backend/ml_models/train.py
============================================================
Run this script to train the ANN, CNN, and Fusion models.
Usage:
    cd liver_disease_detection
    python -m backend.ml_models.train
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint,
    ReduceLROnPlateau, TensorBoard
)
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, accuracy_score
)
from sklearn.preprocessing import label_binarize

# Local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from backend.ml_models.models import (
    build_ann_model, build_cnn_model, build_fusion_model,
    build_ann_feature_extractor, build_cnn_feature_extractor,
    compile_model, save_model
)
from backend.utils.preprocessing import prepare_dataset, load_scaler, CLASS_NAMES

# ─── Paths ───────────────────────────────────────────────────────────────────

RESULTS_DIR = "backend/ml_models/results"
SAVED_MODELS_DIR = "backend/ml_models/saved_models"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

# ─── Callbacks ───────────────────────────────────────────────────────────────

def get_callbacks(model_name: str):
    """Standard training callbacks."""
    return [
        EarlyStopping(
            monitor='val_loss', patience=10,
            restore_best_weights=True, verbose=1
        ),
        ModelCheckpoint(
            filepath=os.path.join(SAVED_MODELS_DIR, f"{model_name}_best.keras"),
            monitor='val_accuracy', save_best_only=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5,
            min_lr=1e-6, verbose=1
        ),
        TensorBoard(
            log_dir=os.path.join(RESULTS_DIR, f"logs/{model_name}"),
            histogram_freq=1
        )
    ]


# ─── Training Functions ───────────────────────────────────────────────────────

def train_ann(data: dict, epochs=50, batch_size=32) -> keras.Model:
    """Train the ANN model on tabular data only."""
    print("\n" + "="*60)
    print("TRAINING ANN MODEL (Tabular Data)")
    print("="*60)

    input_dim = data['train']['tabular'].shape[1]
    ann = build_ann_model(input_dim)
    ann = compile_model(ann, learning_rate=0.001)
    ann.summary()

    history = ann.fit(
        x=data['train']['tabular'],
        y=data['train']['labels'],
        validation_data=(data['val']['tabular'], data['val']['labels']),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=get_callbacks('ann_model'),
        verbose=1
    )

    save_model(ann, 'ann_model')
    plot_training_history(history, 'ANN')
    return ann, history


def train_cnn(data: dict, epochs=30, batch_size=16) -> keras.Model:
    """Train the CNN model on ultrasound images only."""
    print("\n" + "="*60)
    print("TRAINING CNN MODEL (Ultrasound Images)")
    print("="*60)

    cnn = build_cnn_model(trainable_base=False)
    cnn = compile_model(cnn, learning_rate=0.001)

    # Phase 1: Train top layers
    print("Phase 1: Training top layers (base frozen)...")
    history1 = cnn.fit(
        x=data['train']['image'],
        y=data['train']['labels'],
        validation_data=(data['val']['image'], data['val']['labels']),
        epochs=20,
        batch_size=batch_size,
        callbacks=get_callbacks('cnn_model_phase1'),
        verbose=1
    )

    # Phase 2: Fine-tune top layers of ResNet50
    print("\nPhase 2: Fine-tuning ResNet50 top layers...")
    cnn_ft = build_cnn_model(trainable_base=True)
    cnn_ft.set_weights(cnn.get_weights())
    cnn_ft = compile_model(cnn_ft, learning_rate=1e-4)

    history2 = cnn_ft.fit(
        x=data['train']['image'],
        y=data['train']['labels'],
        validation_data=(data['val']['image'], data['val']['labels']),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=get_callbacks('cnn_model'),
        verbose=1
    )

    save_model(cnn_ft, 'cnn_model')
    plot_training_history(history2, 'CNN')
    return cnn_ft, history2


def train_fusion(data: dict, ann_model, cnn_model, epochs=40, batch_size=16) -> keras.Model:
    """Train the multi-modal fusion model."""
    print("\n" + "="*60)
    print("TRAINING FUSION MODEL (Multi-Modal)")
    print("="*60)

    # Build feature extractors from trained models
    ann_feat = build_ann_feature_extractor(ann_model)
    cnn_feat = build_cnn_feature_extractor(cnn_model)

    # Freeze feature extractors initially
    ann_feat.trainable = False
    cnn_feat.trainable = False

    # Build and compile fusion model
    fusion = build_fusion_model(ann_feat, cnn_feat)
    fusion = compile_model(fusion, learning_rate=0.001)
    fusion.summary()

    # Train fusion layers
    history = fusion.fit(
        x=[data['train']['tabular'], data['train']['image']],
        y=data['train']['labels'],
        validation_data=(
            [data['val']['tabular'], data['val']['image']],
            data['val']['labels']
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=get_callbacks('fusion_model'),
        verbose=1
    )

    save_model(fusion, 'fusion_model')
    plot_training_history(history, 'Fusion')
    return fusion, history


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_model(model, test_data, model_type='fusion', model_name='Model'):
    """
    Evaluate model and generate all metrics + plots.
    """
    print(f"\n{'='*60}")
    print(f"EVALUATING {model_name}")
    print(f"{'='*60}")

    # Get predictions
    if model_type == 'fusion':
        y_pred_prob = model.predict([test_data['tabular'], test_data['image']])
    elif model_type == 'ann':
        y_pred_prob = model.predict(test_data['tabular'])
    elif model_type == 'cnn':
        y_pred_prob = model.predict(test_data['image'])

    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(test_data['labels'], axis=1)

    # Accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"\n✅ Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")

    # Classification Report
    print("\n📊 Classification Report:")
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    print(report)

    # Save report
    report_path = os.path.join(RESULTS_DIR, f"{model_name}_report.txt")
    with open(report_path, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Test Accuracy: {acc:.4f}\n\n")
        f.write(report)

    # Confusion Matrix
    plot_confusion_matrix(y_true, y_pred, model_name)

    # ROC Curve
    plot_roc_curve(y_true, y_pred_prob, model_name)

    return acc, y_pred_prob


def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        linewidths=0.5, linecolor='gray'
    )
    plt.title(f'{model_name} - Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{model_name}_confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Confusion matrix saved: {path}")


def plot_roc_curve(y_true, y_pred_prob, model_name):
    """Plot multi-class ROC curves (one-vs-rest)."""
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    colors = ['#e74c3c', '#3498db', '#2ecc71']

    plt.figure(figsize=(10, 7))
    for i, (cls_name, color) in enumerate(zip(CLASS_NAMES, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_pred_prob[:, i])
        auc = roc_auc_score(y_bin[:, i], y_pred_prob[:, i])
        plt.plot(fpr, tpr, color=color, lw=2, label=f'{cls_name} (AUC = {auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'{model_name} - ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{model_name}_roc_curve.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ ROC curve saved: {path}")


def plot_training_history(history, model_name):
    """Plot training/validation accuracy and loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train', color='#3498db', lw=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation', color='#e74c3c', lw=2)
    axes[0].set_title(f'{model_name} - Accuracy', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history['loss'], label='Train', color='#3498db', lw=2)
    axes[1].plot(history.history['val_loss'], label='Validation', color='#e74c3c', lw=2)
    axes[1].set_title(f'{model_name} - Loss', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{model_name}_training_history.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Training history saved: {path}")


# ─── Synthetic Data Generator (for demo/testing) ──────────────────────────────

def generate_synthetic_data(n_samples=600, save_csv=True):
    """
    Generate synthetic liver disease data for demo purposes.
    In production, replace with real clinical dataset.
    """
    np.random.seed(42)
    n_per_class = n_samples // 3
    dfs = []

    # Normal class
    df_normal = pd.DataFrame({
        'age': np.random.randint(20, 60, n_per_class),
        'gender': np.random.choice(['Male', 'Female'], n_per_class),
        'alt': np.random.normal(25, 8, n_per_class).clip(5, 55),
        'ast': np.random.normal(22, 6, n_per_class).clip(5, 45),
        'alp': np.random.normal(80, 20, n_per_class).clip(40, 130),
        'bilirubin_total': np.random.normal(0.8, 0.3, n_per_class).clip(0.2, 1.5),
        'bilirubin_direct': np.random.normal(0.2, 0.1, n_per_class).clip(0.0, 0.5),
        'albumin': np.random.normal(4.2, 0.4, n_per_class).clip(3.5, 5.0),
        'total_protein': np.random.normal(7.2, 0.6, n_per_class).clip(6.0, 8.5),
        'ag_ratio': np.random.normal(1.5, 0.3, n_per_class).clip(1.0, 2.2),
        'label': 'Normal',
        'image_file': [f'normal_{i}.jpg' for i in range(n_per_class)]
    })

    # Fatty Liver class
    df_fatty = pd.DataFrame({
        'age': np.random.randint(30, 65, n_per_class),
        'gender': np.random.choice(['Male', 'Female'], n_per_class),
        'alt': np.random.normal(65, 20, n_per_class).clip(40, 150),
        'ast': np.random.normal(55, 18, n_per_class).clip(35, 120),
        'alp': np.random.normal(110, 30, n_per_class).clip(70, 200),
        'bilirubin_total': np.random.normal(1.5, 0.5, n_per_class).clip(0.8, 3.0),
        'bilirubin_direct': np.random.normal(0.5, 0.2, n_per_class).clip(0.2, 1.2),
        'albumin': np.random.normal(3.8, 0.5, n_per_class).clip(2.8, 4.5),
        'total_protein': np.random.normal(6.8, 0.7, n_per_class).clip(5.5, 8.0),
        'ag_ratio': np.random.normal(1.2, 0.3, n_per_class).clip(0.8, 1.8),
        'label': 'Fatty Liver',
        'image_file': [f'fatty_{i}.jpg' for i in range(n_per_class)]
    })

    # Cirrhosis class
    df_cirrhosis = pd.DataFrame({
        'age': np.random.randint(40, 80, n_per_class),
        'gender': np.random.choice(['Male', 'Female'], n_per_class),
        'alt': np.random.normal(90, 30, n_per_class).clip(40, 200),
        'ast': np.random.normal(110, 35, n_per_class).clip(50, 250),
        'alp': np.random.normal(180, 60, n_per_class).clip(80, 400),
        'bilirubin_total': np.random.normal(4.5, 2.0, n_per_class).clip(1.5, 12.0),
        'bilirubin_direct': np.random.normal(2.5, 1.2, n_per_class).clip(0.5, 7.0),
        'albumin': np.random.normal(2.8, 0.6, n_per_class).clip(1.5, 3.5),
        'total_protein': np.random.normal(5.8, 0.9, n_per_class).clip(4.0, 7.5),
        'ag_ratio': np.random.normal(0.8, 0.25, n_per_class).clip(0.4, 1.3),
        'label': 'Cirrhosis',
        'image_file': [f'cirrhosis_{i}.jpg' for i in range(n_per_class)]
    })

    df = pd.concat([df_normal, df_fatty, df_cirrhosis], ignore_index=True).sample(frac=1, random_state=42)

    if save_csv:
        os.makedirs("data/processed", exist_ok=True)
        csv_path = "data/processed/liver_dataset.csv"
        df.to_csv(csv_path, index=False)
        print(f"✅ Synthetic dataset saved: {csv_path} ({len(df)} samples)")

    return df


# ─── Main Training Pipeline ───────────────────────────────────────────────────

def main():
    """Main training pipeline."""
    print("\n" + "🔬 "*15)
    print("LIVER DISEASE DETECTION - MODEL TRAINING")
    print("🔬 "*15 + "\n")

    # --- For demo: train with tabular-only data (no real images needed) ---
    # In production: use prepare_dataset() with real image directory

    print("📂 Generating synthetic training data...")
    df = generate_synthetic_data(n_samples=900)

    # Preprocess tabular data
    from backend.utils.preprocessing import preprocess_tabular_data, encode_labels
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.utils import to_categorical

    X, scaler = preprocess_tabular_data(df, fit=True)
    labels = df['label'].values
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.classes_ = np.array(CLASS_NAMES)
    import joblib
    joblib.dump(le, "backend/ml_models/saved_models/label_encoder.pkl")
    y_int = le.transform(labels)
    y = to_categorical(y_int, num_classes=3)

    # Create dummy images (224x224x3 noise) for demo
    print("🖼️  Generating placeholder images for demo training...")
    n = len(X)
    X_img_demo = np.random.rand(n, 224, 224, 3).astype(np.float32)

    # Split
    X_tab_train, X_tab_test, X_img_train, X_img_test, y_train, y_test = train_test_split(
        X, X_img_demo, y, test_size=0.2, random_state=42, stratify=y_int
    )
    X_tab_train, X_tab_val, X_img_train, X_img_val, y_train, y_val = train_test_split(
        X_tab_train, X_img_train, y_train, test_size=0.15, random_state=42
    )

    data = {
        'train': {'tabular': X_tab_train, 'image': X_img_train, 'labels': y_train},
        'val': {'tabular': X_tab_val, 'image': X_img_val, 'labels': y_val},
        'test': {'tabular': X_tab_test, 'image': X_img_test, 'labels': y_test},
    }

    print(f"✅ Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")

    # ── Train ANN ──
    ann_model, ann_hist = train_ann(data, epochs=40, batch_size=32)
    evaluate_model(ann_model, data['test'], model_type='ann', model_name='ANN')

    # ── Train CNN ──
    print("\n⚠️  Training CNN with placeholder images (use real ultrasound images in production)")
    cnn_model, cnn_hist = train_cnn(data, epochs=5, batch_size=16)  # Reduced epochs for demo
    evaluate_model(cnn_model, data['test'], model_type='cnn', model_name='CNN')

    # ── Train Fusion ──
    fusion_model, fusion_hist = train_fusion(data, ann_model, cnn_model, epochs=20, batch_size=16)
    evaluate_model(fusion_model, data['test'], model_type='fusion', model_name='FusionModel')

    print("\n" + "✅ "*15)
    print("TRAINING COMPLETE! Models saved in backend/ml_models/saved_models/")
    print("✅ "*15 + "\n")


if __name__ == "__main__":
    main()
