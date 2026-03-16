"""
============================================================
Liver Disease Detection - SHAP Explainability
File: backend/explainability/shap_explainer.py
============================================================
Generates SHAP explanations for ANN tabular predictions.
SHAP (SHapley Additive exPlanations) shows feature importance
for individual predictions.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import uuid
from backend.utils.preprocessing import TABULAR_FEATURES, CLASS_NAMES

EXPLANATIONS_DIR = "backend/static/explanations"
os.makedirs(EXPLANATIONS_DIR, exist_ok=True)

# ─── Feature display names ────────────────────────────────────────────────────

FEATURE_DISPLAY_NAMES = {
    'age': 'Age (years)',
    'gender': 'Gender (M=1/F=0)',
    'alt': 'ALT (U/L)',
    'ast': 'AST (U/L)',
    'alp': 'ALP (U/L)',
    'bilirubin_total': 'Bilirubin Total (mg/dL)',
    'bilirubin_direct': 'Bilirubin Direct (mg/dL)',
    'albumin': 'Albumin (g/dL)',
    'total_protein': 'Total Protein (g/dL)',
    'ag_ratio': 'A/G Ratio'
}


# ─── SHAP Explainer Class ─────────────────────────────────────────────────────

class SHAPExplainer:
    """SHAP-based feature importance explainer for ANN model."""

    def __init__(self, ann_model, background_data: np.ndarray, feature_names: list = None):
        """
        Initialize SHAP explainer.

        Args:
            ann_model: Trained Keras ANN model
            background_data: Background samples for SHAP baseline (e.g., training data)
            feature_names: List of feature names for display
        """
        self.model = ann_model
        self.feature_names = feature_names or [
            FEATURE_DISPLAY_NAMES.get(f, f) for f in TABULAR_FEATURES
        ]

        # Use DeepExplainer for neural networks
        # Use a subset of background data (50-100 samples) for efficiency
        n_bg = min(100, len(background_data))
        bg_sample = background_data[np.random.choice(len(background_data), n_bg, replace=False)]

        print(f"🔍 Initializing SHAP DeepExplainer with {n_bg} background samples...")
        self.explainer = shap.DeepExplainer(self.model, bg_sample)
        print("✅ SHAP explainer ready.")

    def explain_single(self, X_sample: np.ndarray, predicted_class: str = None) -> dict:
        """
        Generate SHAP explanation for a single prediction.

        Args:
            X_sample: Input sample of shape (1, n_features)
            predicted_class: Name of predicted class

        Returns:
            Dict with shap_values, plot_path, feature_importance
        """
        # Compute SHAP values
        shap_values = self.explainer.shap_values(X_sample)

        # shap_values is a list of arrays, one per class
        # Shape: [n_classes, n_samples, n_features]
        if isinstance(shap_values, list):
            # Multi-class: take SHAP for predicted class or aggregate
            class_idx = CLASS_NAMES.index(predicted_class) if predicted_class in CLASS_NAMES else 0
            shap_for_class = shap_values[class_idx][0]  # (n_features,)
        else:
            shap_for_class = shap_values[0]

        # Feature importance (absolute SHAP values)
        feature_importance = dict(zip(self.feature_names, np.abs(shap_for_class)))
        feature_shap = dict(zip(self.feature_names, shap_for_class.tolist()))

        # Generate and save force plot
        plot_path = self._generate_force_plot(
            X_sample[0], shap_for_class, predicted_class
        )

        # Generate bar chart
        bar_path = self._generate_bar_chart(shap_for_class, predicted_class)

        return {
            'shap_values': shap_for_class.tolist(),
            'feature_importance': feature_importance,
            'feature_shap': feature_shap,
            'plot_path': plot_path,
            'bar_path': bar_path,
            'top_features': sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        }

    def _generate_force_plot(self, x_input, shap_vals, predicted_class, filename=None):
        """Generate SHAP force plot and save as PNG."""
        if filename is None:
            filename = f"shap_force_{uuid.uuid4().hex[:8]}.png"

        fig, ax = plt.subplots(figsize=(14, 4))

        # Manual force-plot style bar chart
        features = self.feature_names
        values = shap_vals
        sorted_idx = np.argsort(np.abs(values))[::-1]

        pos_features = [(features[i], values[i], x_input[i]) for i in sorted_idx if values[i] > 0]
        neg_features = [(features[i], values[i], x_input[i]) for i in sorted_idx if values[i] < 0]

        colors = {'pos': '#e74c3c', 'neg': '#3498db'}
        all_features = pos_features + neg_features
        labels = [f"{f[0]}\n={f[2]:.2f}" for f in all_features]
        vals = [f[1] for f in all_features]
        bar_colors = [colors['pos'] if v > 0 else colors['neg'] for v in vals]

        y_pos = range(len(labels))
        bars = ax.barh(list(y_pos), vals, color=bar_colors, alpha=0.85, edgecolor='white', linewidth=0.5)

        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(labels, fontsize=9)
        ax.axvline(x=0, color='black', linewidth=1.2, alpha=0.8)
        ax.set_xlabel('SHAP Value (Impact on Prediction)', fontsize=11)
        ax.set_title(f'Feature Contribution — Predicted: {predicted_class}', fontsize=13, fontweight='bold', pad=10)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=colors['pos'], label='Increases Risk →'),
            Patch(facecolor=colors['neg'], label='← Decreases Risk')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
        ax.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(EXPLANATIONS_DIR, filename)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        return save_path

    def _generate_bar_chart(self, shap_vals, predicted_class, filename=None):
        """Generate simple SHAP importance bar chart."""
        if filename is None:
            filename = f"shap_bar_{uuid.uuid4().hex[:8]}.png"

        abs_shap = np.abs(shap_vals)
        sorted_idx = np.argsort(abs_shap)

        fig, ax = plt.subplots(figsize=(10, 6))
        feature_labels = [self.feature_names[i] for i in sorted_idx]
        bar_vals = [abs_shap[i] for i in sorted_idx]
        raw_vals = [shap_vals[i] for i in sorted_idx]

        colors = ['#e74c3c' if v > 0 else '#3498db' for v in raw_vals]
        ax.barh(range(len(feature_labels)), bar_vals, color=colors, alpha=0.85)
        ax.set_yticks(range(len(feature_labels)))
        ax.set_yticklabels(feature_labels, fontsize=10)
        ax.set_xlabel('|SHAP Value| (Feature Importance)', fontsize=11)
        ax.set_title(f'Feature Importance — {predicted_class}', fontsize=13, fontweight='bold')

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#e74c3c', label='Positive impact (↑ risk)'),
            Patch(facecolor='#3498db', label='Negative impact (↓ risk)')
        ]
        ax.legend(handles=legend_elements, fontsize=9)
        ax.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(EXPLANATIONS_DIR, filename)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        return save_path


# ─── Lightweight SHAP (no DeepExplainer required) ────────────────────────────

def generate_shap_explanation_simple(
    ann_model,
    X_sample: np.ndarray,
    background_data: np.ndarray,
    predicted_class: str,
    feature_names: list = None
) -> dict:
    """
    Generate SHAP explanation using GradientExplainer (faster alternative).
    Works without requiring expensive DeepExplainer initialization.
    """
    if feature_names is None:
        feature_names = [FEATURE_DISPLAY_NAMES.get(f, f) for f in TABULAR_FEATURES]

    n_bg = min(50, len(background_data))
    bg = background_data[:n_bg]

    try:
        explainer = shap.GradientExplainer(ann_model, bg)
        shap_values = explainer.shap_values(X_sample)
    except Exception:
        # Fallback: use KernelExplainer with predict function
        def model_predict(x):
            return ann_model.predict(x, verbose=0)
        explainer = shap.KernelExplainer(model_predict, bg)
        shap_values = explainer.shap_values(X_sample, nsamples=100)

    if isinstance(shap_values, list):
        class_idx = CLASS_NAMES.index(predicted_class) if predicted_class in CLASS_NAMES else 0
        shap_for_class = shap_values[class_idx][0]
    else:
        shap_for_class = shap_values[0]

    # Create bar chart
    filename = f"shap_{uuid.uuid4().hex[:8]}.png"
    abs_shap = np.abs(shap_for_class)
    sorted_idx = np.argsort(abs_shap)

    fig, ax = plt.subplots(figsize=(10, 6))
    labels = [feature_names[i] for i in sorted_idx]
    vals = [abs_shap[i] for i in sorted_idx]
    raw = [shap_for_class[i] for i in sorted_idx]
    colors = ['#e74c3c' if v > 0 else '#3498db' for v in raw]

    ax.barh(range(len(labels)), vals, color=colors, alpha=0.85, edgecolor='white', linewidth=0.3)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('|SHAP Value| (Contribution Magnitude)', fontsize=11)
    ax.set_title(f'SHAP Feature Importance\nPredicted: {predicted_class}', fontsize=13, fontweight='bold')

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor='#e74c3c', label='↑ Increases risk'),
        Patch(facecolor='#3498db', label='↓ Decreases risk')
    ], fontsize=9)
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()

    os.makedirs(EXPLANATIONS_DIR, exist_ok=True)
    save_path = os.path.join(EXPLANATIONS_DIR, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    feature_importance = {
        feature_names[i]: float(abs_shap[i]) for i in range(len(feature_names))
    }
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        'shap_values': shap_for_class.tolist(),
        'feature_importance': feature_importance,
        'plot_path': save_path,
        'top_features': top_features
    }
