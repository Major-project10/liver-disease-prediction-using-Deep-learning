"""
============================================================
Liver Disease Detection - Deep Learning Models
File: backend/ml_models/models.py
============================================================
Defines:
  1. ANN model for tabular clinical data
  2. CNN model (ResNet50-based) for ultrasound images
  3. Fusion model (concatenation of ANN + CNN features)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.regularizers import l2

# ─── Configuration ───────────────────────────────────────────────────────────

NUM_CLASSES = 3
IMAGE_SHAPE = (224, 224, 3)
SAVED_MODELS_DIR = "backend/ml_models/saved_models"

os.makedirs(SAVED_MODELS_DIR, exist_ok=True)


# ─── 1. ANN Model (Tabular Data) ─────────────────────────────────────────────

def build_ann_model(input_dim: int, num_classes: int = NUM_CLASSES) -> Model:
    """
    Build a Deep ANN for processing clinical tabular data.

    Architecture:
        Input → Dense(256) → BN → Dropout → Dense(128) → BN → Dropout
        → Dense(64) → BN → Dropout → Feature_Output(32)

    Args:
        input_dim: Number of clinical features
        num_classes: Number of output classes

    Returns:
        Keras Model with feature extractor + classifier head
    """
    inputs = keras.Input(shape=(input_dim,), name='tabular_input')

    # Block 1
    x = layers.Dense(256, activation='relu', kernel_regularizer=l2(1e-4), name='dense_1')(inputs)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.Dropout(0.3, name='drop_1')(x)

    # Block 2
    x = layers.Dense(128, activation='relu', kernel_regularizer=l2(1e-4), name='dense_2')(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.Dropout(0.3, name='drop_2')(x)

    # Block 3
    x = layers.Dense(64, activation='relu', kernel_regularizer=l2(1e-4), name='dense_3')(x)
    x = layers.BatchNormalization(name='bn_3')(x)
    x = layers.Dropout(0.2, name='drop_3')(x)

    # Feature embedding (used for fusion)
    features = layers.Dense(32, activation='relu', name='ann_features')(x)

    # Classification head (standalone ANN)
    outputs = layers.Dense(num_classes, activation='softmax', name='ann_output')(features)

    model = Model(inputs=inputs, outputs=outputs, name='ANN_TabularModel')
    return model


def build_ann_feature_extractor(ann_model: Model) -> Model:
    """
    Create feature extractor from trained ANN (removes classifier head).
    Returns the 'ann_features' layer output.
    """
    return Model(
        inputs=ann_model.input,
        outputs=ann_model.get_layer('ann_features').output,
        name='ANN_FeatureExtractor'
    )


# ─── 2. CNN Model (Ultrasound Images) ────────────────────────────────────────

def build_cnn_model(num_classes: int = NUM_CLASSES, trainable_base: bool = False) -> Model:
    """
    Build a CNN using pretrained ResNet50 via transfer learning.

    Architecture:
        ResNet50 (pretrained, ImageNet) → GlobalAvgPool
        → Dense(256) → BN → Dropout → Feature_Output(64) → Softmax

    Args:
        num_classes: Number of output classes
        trainable_base: Whether to fine-tune ResNet50 base layers

    Returns:
        Keras CNN model
    """
    # Load pretrained ResNet50 (exclude top classifier)
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=IMAGE_SHAPE
    )

    # Freeze base model initially
    base_model.trainable = trainable_base

    if trainable_base:
        # Unfreeze last 30 layers for fine-tuning
        for layer in base_model.layers[:-30]:
            layer.trainable = False

    # Build top layers
    inputs = keras.Input(shape=IMAGE_SHAPE, name='image_input')
    x = base_model(inputs, training=trainable_base)
    x = layers.GlobalAveragePooling2D(name='gap')(x)

    # Dense block
    x = layers.Dense(256, activation='relu', kernel_regularizer=l2(1e-4), name='cnn_dense_1')(x)
    x = layers.BatchNormalization(name='cnn_bn_1')(x)
    x = layers.Dropout(0.4, name='cnn_drop_1')(x)

    # Feature embedding (used for fusion)
    features = layers.Dense(64, activation='relu', name='cnn_features')(x)

    # Classification head (standalone CNN)
    outputs = layers.Dense(num_classes, activation='softmax', name='cnn_output')(features)

    model = Model(inputs=inputs, outputs=outputs, name='CNN_UltrasoundModel')
    return model


def build_cnn_feature_extractor(cnn_model: Model) -> Model:
    """
    Create feature extractor from trained CNN (removes classifier head).
    Returns the 'cnn_features' layer output.
    """
    return Model(
        inputs=cnn_model.input,
        outputs=cnn_model.get_layer('cnn_features').output,
        name='CNN_FeatureExtractor'
    )


# ─── 3. Fusion Model (Multi-Modal) ───────────────────────────────────────────

def build_fusion_model(
    ann_feature_extractor: Model,
    cnn_feature_extractor: Model,
    num_classes: int = NUM_CLASSES
) -> Model:
    """
    Build the multi-modal fusion model combining ANN + CNN features.

    Architecture:
        ANN_Features(32) + CNN_Features(64) → Concatenate(96)
        → Dense(128) → BN → Dropout → Dense(64) → BN → Dropout
        → Dense(num_classes) → Softmax

    Args:
        ann_feature_extractor: ANN feature model (outputs 32-dim)
        cnn_feature_extractor: CNN feature model (outputs 64-dim)
        num_classes: Output classes

    Returns:
        Fusion Keras Model
    """
    # Tabular branch
    tabular_input = keras.Input(shape=ann_feature_extractor.input_shape[1:], name='tabular_input')
    ann_out = ann_feature_extractor(tabular_input)  # (batch, 32)

    # Image branch
    image_input = keras.Input(shape=IMAGE_SHAPE, name='image_input')
    cnn_out = cnn_feature_extractor(image_input)   # (batch, 64)

    # Feature fusion via concatenation
    merged = layers.Concatenate(name='feature_fusion')([ann_out, cnn_out])  # (batch, 96)

    # Fully-connected fusion layers
    x = layers.Dense(128, activation='relu', kernel_regularizer=l2(1e-4), name='fusion_dense_1')(merged)
    x = layers.BatchNormalization(name='fusion_bn_1')(x)
    x = layers.Dropout(0.4, name='fusion_drop_1')(x)

    x = layers.Dense(64, activation='relu', kernel_regularizer=l2(1e-4), name='fusion_dense_2')(x)
    x = layers.BatchNormalization(name='fusion_bn_2')(x)
    x = layers.Dropout(0.3, name='fusion_drop_2')(x)

    # Final output
    outputs = layers.Dense(num_classes, activation='softmax', name='fusion_output')(x)

    model = Model(
        inputs=[tabular_input, image_input],
        outputs=outputs,
        name='FusionModel_MultiModal'
    )
    return model


# ─── Model Compilation ───────────────────────────────────────────────────────

def compile_model(model: Model, learning_rate: float = 0.001) -> Model:
    """Compile model with Adam optimizer and categorical crossentropy."""
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy',
                 keras.metrics.AUC(name='auc', multi_label=False),
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')]
    )
    return model


# ─── Model Save / Load ───────────────────────────────────────────────────────

def save_model(model: Model, name: str):
    """Save model to disk in Keras format."""
    path = os.path.join(SAVED_MODELS_DIR, f"{name}.keras")
    model.save(path)
    print(f"✅ Model saved: {path}")
    return path


def load_model(name: str) -> Model:
    """Load model from disk."""
    path = os.path.join(SAVED_MODELS_DIR, f"{name}.keras")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    model = keras.models.load_model(path)
    print(f"✅ Model loaded: {path}")
    return model


# ─── Model Summary Utility ───────────────────────────────────────────────────

def print_all_model_summaries():
    """Print summaries of all three models."""
    input_dim = 10  # Number of clinical features

    print("\n" + "="*60)
    print("ANN MODEL SUMMARY")
    print("="*60)
    ann = build_ann_model(input_dim)
    ann.summary()

    print("\n" + "="*60)
    print("CNN MODEL SUMMARY")
    print("="*60)
    cnn = build_cnn_model()
    cnn.summary()

    print("\n" + "="*60)
    print("FUSION MODEL SUMMARY")
    print("="*60)
    ann_feat = build_ann_feature_extractor(ann)
    cnn_feat = build_cnn_feature_extractor(cnn)
    fusion = build_fusion_model(ann_feat, cnn_feat)
    fusion.summary()


if __name__ == "__main__":
    print_all_model_summaries()
