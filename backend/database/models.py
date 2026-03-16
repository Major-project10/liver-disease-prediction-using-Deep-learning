"""
============================================================
Liver Disease Detection - Database Models
File: backend/database/models.py
============================================================
"""

from datetime import datetime
from backend.database.db import db


class Patient(db.Model):
    """Patient record model."""
    __tablename__ = 'patients'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    email = db.Column(db.String(120), nullable=True)
    phone = db.Column(db.String(20), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship to predictions
    predictions = db.relationship('Prediction', backref='patient', lazy=True, cascade='all, delete-orphan')

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'age': self.age,
            'gender': self.gender,
            'email': self.email,
            'phone': self.phone,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


class LabValues(db.Model):
    """Clinical lab values model."""
    __tablename__ = 'lab_values'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    prediction_id = db.Column(db.Integer, db.ForeignKey('predictions.id'), nullable=False)

    # Liver Function Tests
    alt = db.Column(db.Float, nullable=False)           # Alanine Aminotransferase (U/L)
    ast = db.Column(db.Float, nullable=False)           # Aspartate Aminotransferase (U/L)
    alp = db.Column(db.Float, nullable=False)           # Alkaline Phosphatase (U/L)
    bilirubin_total = db.Column(db.Float, nullable=False)   # Total Bilirubin (mg/dL)
    bilirubin_direct = db.Column(db.Float, nullable=False)  # Direct Bilirubin (mg/dL)
    albumin = db.Column(db.Float, nullable=False)        # Albumin (g/dL)
    total_protein = db.Column(db.Float, nullable=False)  # Total Protein (g/dL)
    ag_ratio = db.Column(db.Float, nullable=False)       # A/G Ratio

    def to_dict(self):
        return {
            'alt': self.alt,
            'ast': self.ast,
            'alp': self.alp,
            'bilirubin_total': self.bilirubin_total,
            'bilirubin_direct': self.bilirubin_direct,
            'albumin': self.albumin,
            'total_protein': self.total_protein,
            'ag_ratio': self.ag_ratio,
        }


class Prediction(db.Model):
    """Prediction results model."""
    __tablename__ = 'predictions'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=False)

    # Prediction Results
    predicted_class = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)

    # Probability scores (JSON)
    prob_normal = db.Column(db.Float, nullable=True)
    prob_fatty_liver = db.Column(db.Float, nullable=True)
    prob_cirrhosis = db.Column(db.Float, nullable=True)

    # Model type used
    model_type = db.Column(db.String(20), nullable=False, default='fusion')  # ann, cnn, fusion

    # Image path
    image_path = db.Column(db.String(255), nullable=True)

    # Explanation paths
    shap_plot_path = db.Column(db.String(255), nullable=True)
    gradcam_plot_path = db.Column(db.String(255), nullable=True)

    # Metadata
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    notes = db.Column(db.Text, nullable=True)

    # Relationship to lab values
    lab_values = db.relationship('LabValues', backref='prediction', lazy=True, uselist=False, cascade='all, delete-orphan')

    def to_dict(self):
        return {
            'id': self.id,
            'patient_id': self.patient_id,
            'predicted_class': self.predicted_class,
            'confidence': self.confidence,
            'probabilities': {
                'Normal': self.prob_normal,
                'Fatty Liver': self.prob_fatty_liver,
                'Cirrhosis': self.prob_cirrhosis,
            },
            'model_type': self.model_type,
            'image_path': self.image_path,
            'shap_plot_path': self.shap_plot_path,
            'gradcam_plot_path': self.gradcam_plot_path,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'notes': self.notes,
            'lab_values': self.lab_values.to_dict() if self.lab_values else None,
        }
