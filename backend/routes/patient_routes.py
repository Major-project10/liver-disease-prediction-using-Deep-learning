"""
============================================================
Liver Disease Detection - Patient API Routes
File: backend/routes/patient_routes.py
============================================================
REST API endpoints for patient and prediction history management.
"""

from flask import Blueprint, request, jsonify
from backend.database.db import db
from backend.database.models import Patient, Prediction, LabValues

patient_bp = Blueprint('patient', __name__, url_prefix='/api/patients')


# ─── Get All Patients ─────────────────────────────────────────────────────────

@patient_bp.route('/', methods=['GET'])
def get_all_patients():
    """Retrieve all patients with optional search."""
    search   = request.args.get('search', '')
    page     = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 20))

    query = Patient.query
    if search:
        query = query.filter(Patient.name.ilike(f'%{search}%'))

    paginated = query.order_by(Patient.created_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )

    return jsonify({
        'patients':     [p.to_dict() for p in paginated.items],
        'total':        paginated.total,
        'pages':        paginated.pages,
        'current_page': page
    }), 200


# ─── Get Single Patient ───────────────────────────────────────────────────────

@patient_bp.route('/<int:patient_id>', methods=['GET'])
def get_patient(patient_id):
    """Get a specific patient by ID."""
    patient = Patient.query.get_or_404(patient_id)
    data = patient.to_dict()
    data['predictions'] = [p.to_dict() for p in patient.predictions]
    return jsonify(data), 200


# ─── Create Patient ───────────────────────────────────────────────────────────

@patient_bp.route('/', methods=['POST'])
def create_patient():
    """Create a new patient record."""
    body = request.get_json()
    if not body:
        return jsonify({'error': 'No data provided'}), 400

    required = ['name', 'age', 'gender']
    missing  = [f for f in required if f not in body]
    if missing:
        return jsonify({'error': f'Missing: {missing}'}), 400

    patient = Patient(
        name=body['name'],
        age=int(body['age']),
        gender=body['gender'],
        email=body.get('email'),
        phone=body.get('phone')
    )
    db.session.add(patient)
    db.session.commit()

    return jsonify({'success': True, 'patient': patient.to_dict()}), 201


# ─── Update Patient ───────────────────────────────────────────────────────────

@patient_bp.route('/<int:patient_id>', methods=['PUT'])
def update_patient(patient_id):
    """Update patient information."""
    patient = Patient.query.get_or_404(patient_id)
    body    = request.get_json() or {}

    for field in ['name', 'age', 'gender', 'email', 'phone']:
        if field in body:
            setattr(patient, field, body[field])

    db.session.commit()
    return jsonify({'success': True, 'patient': patient.to_dict()}), 200


# ─── Delete Patient ───────────────────────────────────────────────────────────

@patient_bp.route('/<int:patient_id>', methods=['DELETE'])
def delete_patient(patient_id):
    """Delete a patient and all associated records."""
    patient = Patient.query.get_or_404(patient_id)
    db.session.delete(patient)
    db.session.commit()
    return jsonify({'success': True, 'message': f'Patient {patient_id} deleted'}), 200


# ─── Get Patient Predictions ──────────────────────────────────────────────────

@patient_bp.route('/<int:patient_id>/predictions', methods=['GET'])
def get_patient_predictions(patient_id):
    """Get all predictions for a specific patient."""
    Patient.query.get_or_404(patient_id)
    predictions = Prediction.query.filter_by(patient_id=patient_id) \
        .order_by(Prediction.created_at.desc()).all()

    return jsonify({
        'patient_id':  patient_id,
        'predictions': [p.to_dict() for p in predictions],
        'count':       len(predictions)
    }), 200


# ─── Get Single Prediction Detail ─────────────────────────────────────────────
# FIX: This entire endpoint was missing — the View modal calls it.

@patient_bp.route('/predictions/<int:prediction_id>', methods=['GET'])
def get_prediction_detail(prediction_id):
    """
    Get full detail for one prediction including patient info and lab values.
    Called by showDetail() in history.js when the user clicks View.
    """
    pred = Prediction.query.get_or_404(prediction_id)
    data = pred.to_dict()

    # Inline patient fields so the modal can access them directly
    if pred.patient:
        data['patient_name'] = pred.patient.name
        data['age']          = pred.patient.age
        data['gender']       = pred.patient.gender

    # Inline lab values if stored in a related LabValues table
    if hasattr(pred, 'lab_values') and pred.lab_values:
        lab = pred.lab_values
        data.update({
            'alt':              lab.alt,
            'ast':              lab.ast,
            'alp':              lab.alp,
            'bilirubin_total':  lab.bilirubin_total,
            'bilirubin_direct': lab.bilirubin_direct,
            'albumin':          lab.albumin,
            'total_protein':    lab.total_protein,
            'ag_ratio':         lab.ag_ratio,
        })

    return jsonify({'prediction': data}), 200


# ─── Get All Predictions (History Page) ──────────────────────────────────────

@patient_bp.route('/predictions/all', methods=['GET'])
def get_all_predictions():
    """
    Get all predictions with patient info — used by the history dashboard.

    Query params accepted:
        page        int    (default 1)
        per_page    int    (default 10)
        search      str    patient name substring
        diagnosis   str    'Normal' | 'Fatty Liver' | 'Cirrhosis'
        model_type  str    'ann' | 'cnn' | 'fusion'
    """
    page       = int(request.args.get('page', 1))
    per_page   = int(request.args.get('per_page', 10))
    search     = request.args.get('search', '').strip()
    # FIX: was reading 'class' — JS sends 'diagnosis'
    diagnosis  = request.args.get('diagnosis', '').strip()
    # FIX: was reading 'model' — JS sends 'model_type'
    model_type = request.args.get('model_type', '').strip()

    # Join Patient table so we can filter by name and return age/gender
    query = db.session.query(Prediction).join(Patient, Prediction.patient_id == Patient.id)

    if search:
        query = query.filter(Patient.name.ilike(f'%{search}%'))
    if diagnosis:
        query = query.filter(Prediction.predicted_class == diagnosis)
    if model_type:
        query = query.filter(Prediction.model_type == model_type)

    paginated = query.order_by(Prediction.created_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )

    results = []
    for pred in paginated.items:
        d = pred.to_dict()
        # FIX: age and gender were never included — showed as '—' in the table
        d['patient_name'] = pred.patient.name   if pred.patient else 'Unknown'
        d['age']          = pred.patient.age    if pred.patient else None
        d['gender']       = pred.patient.gender if pred.patient else None
        results.append(d)

    return jsonify({
        'predictions':  results,
        'total':        paginated.total,
        # FIX: was 'pages' — JS expects 'total_pages'
        'total_pages':  paginated.pages,
        'current_page': page
    }), 200


# ─── Statistics Summary ───────────────────────────────────────────────────────

@patient_bp.route('/stats/summary', methods=['GET'])
def get_stats():
    """Dashboard statistics shown in the stats bar on the history page."""
    total_patients    = Patient.query.count()
    total_predictions = Prediction.query.count()

    normal_count      = Prediction.query.filter_by(predicted_class='Normal').count()
    fatty_liver_count = Prediction.query.filter_by(predicted_class='Fatty Liver').count()
    cirrhosis_count   = Prediction.query.filter_by(predicted_class='Cirrhosis').count()

    recent = Prediction.query.order_by(Prediction.created_at.desc()).limit(5).all()

    return jsonify({
        'total_patients':    total_patients,
        'total_predictions': total_predictions,
        # Nested dict — matches what history.js reads via class_distribution
        'class_distribution': {
            'Normal':      normal_count,
            'Fatty Liver': fatty_liver_count,
            'Cirrhosis':   cirrhosis_count,
        },
        'recent_predictions': [p.to_dict() for p in recent]
    }), 200
