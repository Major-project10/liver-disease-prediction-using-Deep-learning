"""
============================================================
Liver Disease Detection - Grad-CAM Explainability
File: backend/explainability/gradcam.py
============================================================
Generates Gradient-weighted Class Activation Maps (Grad-CAM)
to highlight important regions in ultrasound images that
influenced the CNN's prediction.
"""

import os
import uuid
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
from tensorflow import keras

from backend.utils.preprocessing import preprocess_image, CLASS_NAMES

EXPLANATIONS_DIR = "backend/static/explanations"
os.makedirs(EXPLANATIONS_DIR, exist_ok=True)


# ─── Grad-CAM Core ───────────────────────────────────────────────────────────

def get_gradcam_heatmap(
    model: keras.Model,
    img_array: np.ndarray,
    pred_class_idx: int,
    last_conv_layer_name: str = None
) -> np.ndarray:
    """
    Compute Grad-CAM heatmap for the given image and class.

    Args:
        model: Trained CNN Keras model
        img_array: Preprocessed image array of shape (1, H, W, 3)
        pred_class_idx: Index of the predicted class
        last_conv_layer_name: Name of last convolutional layer
                              (auto-detected if None)

    Returns:
        Normalized heatmap array of shape (H, W) with values in [0, 1]
    """
    # Auto-detect last convolutional layer if not specified
    if last_conv_layer_name is None:
        last_conv_layer_name = _find_last_conv_layer(model)
        print(f"🔍 Using conv layer: {last_conv_layer_name}")

    # Build grad model: inputs → (conv_outputs, final_predictions)
    grad_model = keras.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    # Record operations for gradient computation
    with tf.GradientTape() as tape:
        inputs = tf.cast(img_array, tf.float32)
        conv_outputs, predictions = grad_model(inputs)
        # Get score for predicted class
        class_score = predictions[:, pred_class_idx]

    # Compute gradients of class score w.r.t. conv feature maps
    grads = tape.gradient(class_score, conv_outputs)  # (1, H', W', C)

    # Pool gradients over spatial dimensions → importance weights per channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (C,)

    # Weight feature maps by pooled gradients
    conv_outputs = conv_outputs[0]  # (H', W', C)
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]  # (H', W', 1)
    heatmap = tf.squeeze(heatmap)  # (H', W')

    # Apply ReLU (keep only positive influences)
    heatmap = tf.maximum(heatmap, 0)

    # Normalize to [0, 1]
    heatmap = heatmap.numpy()
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    return heatmap


def _find_last_conv_layer(model: keras.Model) -> str:
    """Find the name of the last convolutional layer in the model."""
    # For ResNet50, look for the last conv layer in the base model
    for layer in reversed(model.layers):
        # Check if it's a sub-model (like resnet50_base)
        if hasattr(layer, 'layers'):
            for sub_layer in reversed(layer.layers):
                if 'conv' in sub_layer.name.lower() or 'activation' in sub_layer.name.lower():
                    # Return sub-model last activation
                    return layer.name
        if 'conv' in layer.name.lower() and len(layer.output_shape) == 4:
            return layer.name

    # Fallback for ResNet50
    return 'resnet50_base'


def _find_resnet_last_conv(model: keras.Model) -> str:
    """Find the last convolutional layer specifically for ResNet50."""
    # Look for ResNet-specific layer names
    target_names = ['conv5_block3_out', 'conv5_block3_add', 'post_bn', 'top_activation']
    for name in target_names:
        try:
            model.get_layer(name)
            return name
        except ValueError:
            continue

    # Search nested layers
    for layer in model.layers:
        if hasattr(layer, 'layers'):
            for sublayer in reversed(layer.layers):
                if 'conv5' in sublayer.name or 'out' in sublayer.name:
                    return sublayer.name

    return 'resnet50_base'


# ─── Overlay & Visualization ─────────────────────────────────────────────────

def create_gradcam_overlay(
    original_image_path: str,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: str = 'jet'
) -> np.ndarray:
    """
    Superimpose Grad-CAM heatmap on the original image.

    Args:
        original_image_path: Path to original ultrasound image
        heatmap: Normalized heatmap from get_gradcam_heatmap
        alpha: Heatmap opacity (0=invisible, 1=full)
        colormap: Matplotlib colormap name

    Returns:
        RGB overlay image array (H, W, 3) in uint8 format
    """
    # Load original image
    img = cv2.imread(original_image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    # Resize heatmap to match image
    heatmap_resized = cv2.resize(heatmap, (224, 224))

    # Apply colormap
    cmap = cm.get_cmap(colormap)
    heatmap_colored = cmap(heatmap_resized)[:, :, :3]  # Drop alpha channel
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

    # Blend
    img_float = img.astype(np.float32)
    heatmap_float = heatmap_colored.astype(np.float32)
    overlay = (img_float * (1 - alpha) + heatmap_float * alpha).clip(0, 255).astype(np.uint8)

    return overlay


def generate_gradcam_plot(
    cnn_model: keras.Model,
    image_path: str,
    predicted_class: str,
    patient_id: int = None,
    filename: str = None
) -> str:
    """
    Full pipeline: compute Grad-CAM and generate visualization plot.

    Args:
        cnn_model: Trained CNN model
        image_path: Path to ultrasound image
        predicted_class: Predicted class name
        patient_id: Optional patient ID for filename
        filename: Optional custom filename

    Returns:
        Path to saved Grad-CAM plot
    """
    from backend.utils.preprocessing import preprocess_image

    if filename is None:
        uid = f"p{patient_id}_" if patient_id else ""
        filename = f"gradcam_{uid}{uuid.uuid4().hex[:8]}.png"

    save_path = os.path.join(EXPLANATIONS_DIR, filename)

    try:
        # Preprocess image
        img_array = preprocess_image(image_path)  # (1, 224, 224, 3)
        pred_class_idx = CLASS_NAMES.index(predicted_class) if predicted_class in CLASS_NAMES else 0

        # Get heatmap
        heatmap = get_gradcam_heatmap(cnn_model, img_array, pred_class_idx)

        # Load original image for display
        orig_img = cv2.imread(image_path)
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        orig_img_resized = cv2.resize(orig_img, (224, 224))

        # Create overlay
        overlay = create_gradcam_overlay(image_path, heatmap)

        # Plot: Original | Heatmap | Overlay
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.patch.set_facecolor('#1a1a2e')

        # Original
        axes[0].imshow(orig_img_resized)
        axes[0].set_title('Original Ultrasound', color='white', fontsize=12, fontweight='bold', pad=8)
        axes[0].axis('off')

        # Heatmap
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        axes[1].imshow(heatmap_resized, cmap='jet', vmin=0, vmax=1)
        axes[1].set_title('Grad-CAM Heatmap', color='white', fontsize=12, fontweight='bold', pad=8)
        axes[1].axis('off')

        # Overlay
        axes[2].imshow(overlay)
        axes[2].set_title('Activation Overlay', color='white', fontsize=12, fontweight='bold', pad=8)
        axes[2].axis('off')

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04)
        cbar.set_label('Activation Intensity', color='white', fontsize=9)
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

        fig.suptitle(
            f'Grad-CAM Analysis  |  Prediction: {predicted_class}',
            color='white', fontsize=14, fontweight='bold', y=1.02
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
        plt.close()

        print(f"✅ Grad-CAM plot saved: {save_path}")
        return save_path

    except Exception as e:
        print(f"⚠️  Grad-CAM generation failed: {e}")
        # Generate fallback placeholder plot
        return _generate_placeholder_gradcam(image_path, predicted_class, save_path)


def _generate_placeholder_gradcam(image_path: str, predicted_class: str, save_path: str) -> str:
    """Generate a placeholder when Grad-CAM fails (e.g., model not fully compatible)."""
    try:
        img = cv2.imread(image_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(img)
            ax.set_title(f'Ultrasound Image\nPrediction: {predicted_class}',
                        fontsize=12, fontweight='bold', pad=10)
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(save_path, dpi=120, bbox_inches='tight')
            plt.close()
    except Exception:
        pass

    return save_path
