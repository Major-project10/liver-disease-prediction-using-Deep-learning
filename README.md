# 🔬 HepatoAI — Liver Disease Detection System
## Complete Setup, Run & Deployment Guide

---

## 📁 PROJECT FOLDER STRUCTURE

```
liver_disease_detection/
├── app.py                          ← Flask entry point (RUN THIS)
├── requirements.txt                ← Python dependencies
├── .env                            ← Environment variables
├── README.md                       ← This file
│
├── backend/
│   ├── __init__.py
│   ├── database/
│   │   ├── __init__.py
│   │   ├── db.py                   ← SQLAlchemy init
│   │   └── models.py               ← Patient, Prediction, LabValues models
│   ├── ml_models/
│   │   ├── __init__.py
│   │   ├── models.py               ← ANN, CNN, Fusion model definitions
│   │   ├── train.py                ← Training script (run to train)
│   │   ├── predictor.py            ← Inference engine
│   │   ├── saved_models/           ← .keras model files saved here
│   │   └── results/                ← Training plots saved here
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── prediction_routes.py    ← /api/predict/* endpoints
│   │   └── patient_routes.py       ← /api/patients/* endpoints
│   ├── utils/
│   │   ├── __init__.py
│   │   └── preprocessing.py        ← Data scaling, image normalization
│   ├── explainability/
│   │   ├── __init__.py
│   │   ├── shap_explainer.py       ← SHAP feature explanations
│   │   └── gradcam.py              ← Grad-CAM image explanations
│   ├── uploads/                    ← Uploaded ultrasound images
│   └── static/
│       └── explanations/           ← SHAP & Grad-CAM plots
│
├── frontend/
│   ├── templates/
│   │   ├── index.html              ← Home page
│   │   ├── predict.html            ← Prediction form + results
│   │   ├── history.html            ← Patient history
│   │   └── about.html              ← Project info
│   └── static/
│       ├── css/
│       │   ├── main.css            ← Global styles
│       │   ├── predict.css         ← Predict page styles
│       │   └── history.css         ← History page styles
│       └── js/
│           ├── main.js             ← Global JS
│           ├── predict.js          ← Prediction form logic
│           └── history.js          ← History table & modal
│
├── data/
│   ├── raw/                        ← Place raw dataset here
│   ├── processed/                  ← Processed CSV saved here
│   └── sample/                     ← Sample CSV for reference
│       └── sample_patients.csv
│
└── notebooks/                      ← Jupyter notebooks (optional EDA)
```

---

## ⚙️ STEP 1 — PREREQUISITES

- Python 3.10 or 3.11 (recommended)
- pip (Python package manager)
- Git

---

## 🛠️ STEP 2 — INSTALLATION

### 2.1 Clone / Create Project Directory

```bash
# Navigate to where you want the project
cd C:/Projects        # Windows
cd ~/projects         # macOS/Linux

# The project folder should be: liver_disease_detection/
```

### 2.2 Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 2.3 Install Dependencies

```bash
pip install -r requirements.txt
```

> 💡 This installs TensorFlow, Flask, SHAP, OpenCV, Scikit-learn, etc.
> May take 5–10 minutes depending on internet speed.

---

## 🤖 STEP 3 — TRAIN THE MODELS

This step trains 3 models and saves them to `backend/ml_models/saved_models/`.

```bash
# From the project root directory (liver_disease_detection/)
python -m backend.ml_models.train
```

### What this does:
1. Generates synthetic liver disease dataset (900 samples, 3 classes)
2. Preprocesses and scales tabular features (saves `scaler.pkl`)
3. Saves `label_encoder.pkl`
4. Trains **ANN model** (tabular-only) → saves `ann_model.keras`
5. Trains **CNN model** (ResNet50, image-only) → saves `cnn_model.keras`
6. Trains **Fusion model** (ANN + CNN) → saves `fusion_model.keras`
7. Evaluates each model → generates confusion matrix, ROC curves, training history plots

### Using Real Data (Production):
Replace the synthetic data generator with your actual dataset:
```python
# In train.py, replace generate_synthetic_data() with:
data = prepare_dataset('data/raw/liver_data.csv', 'data/raw/images/')
```

Your CSV must have columns:
`age, gender, alt, ast, alp, bilirubin_total, bilirubin_direct, albumin, total_protein, ag_ratio, label, image_file`

---

## 🚀 STEP 4 — RUN THE APPLICATION

```bash
# From the project root directory
python app.py
```

You should see:
```
✅ Database initialized successfully.
 🔬 Liver Disease Detection System
 🌐 Running at: http://localhost:5000
```

Open browser: **http://localhost:5000**

---

## 🌐 STEP 5 — USING THE APP

### Home Page (`/`)
- Overview of the system architecture
- Links to all sections

### Predict Page (`/predict`)
1. Fill in patient info (name, age, gender)
2. Enter lab values (ALT, AST, ALP, Bilirubin, Albumin, etc.)
3. Optionally upload a liver ultrasound image
4. Click **"Analyze & Diagnose"**
5. View prediction, probabilities, SHAP feature importance, and Grad-CAM

**Quick Fill buttons** auto-populate sample data for testing.

### History Page (`/history`)
- View all stored predictions
- Filter by class, model type, or search by patient name
- Click "View" to see full details including XAI plots

---

## 📡 REST API ENDPOINTS

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | System health check |
| POST | `/api/predict/tabular` | Predict from lab values only (ANN) |
| POST | `/api/predict/image` | Predict from ultrasound only (CNN) |
| POST | `/api/predict/fusion` | Full prediction + SHAP + Grad-CAM |
| GET | `/api/patients/` | List all patients |
| POST | `/api/patients/` | Create patient |
| GET | `/api/patients/<id>` | Get patient by ID |
| GET | `/api/patients/<id>/predictions` | Patient's prediction history |
| GET | `/api/patients/predictions/all` | All predictions (paginated) |
| GET | `/api/patients/stats/summary` | Dashboard statistics |

### Example API call (cURL):
```bash
curl -X POST http://localhost:5000/api/predict/tabular \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45, "gender": "Male",
    "alt": 55, "ast": 48, "alp": 120,
    "bilirubin_total": 1.8, "bilirubin_direct": 0.6,
    "albumin": 3.5, "total_protein": 6.8, "ag_ratio": 1.1
  }'
```

### Example Fusion call (Python):
```python
import requests

url = "http://localhost:5000/api/predict/fusion"
clinical_data = {
    "age": 62, "gender": "Male",
    "alt": 95, "ast": 130, "alp": 210,
    "bilirubin_total": 5.2, "bilirubin_direct": 3.1,
    "albumin": 2.6, "total_protein": 5.5, "ag_ratio": 0.75
}

with open("liver_scan.jpg", "rb") as img:
    response = requests.post(url, 
        data={"data": json.dumps(clinical_data), "patient_name": "Test Patient"},
        files={"image": img}
    )

print(response.json()['prediction']['predicted_class'])
```

---

## ☁️ DEPLOYMENT

### Option A: Render (Free, Recommended for Demo)

1. Push your project to GitHub
2. Go to https://render.com → New → Web Service
3. Connect your GitHub repo
4. Set:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT`
5. Add environment variables from `.env`
6. Deploy!

### Option B: AWS EC2

```bash
# SSH into EC2 instance
ssh -i key.pem ubuntu@your-ec2-ip

# Install dependencies
sudo apt update && sudo apt install python3-pip python3-venv -y

# Clone and set up
git clone your-repo-url
cd liver_disease_detection
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Train models
python -m backend.ml_models.train

# Run with Gunicorn + Nginx
gunicorn app:app --workers 2 --bind 0.0.0.0:8000 --daemon
```

### Option C: Docker

```dockerfile
# Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python -m backend.ml_models.train
EXPOSE 5000
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
```

```bash
docker build -t hepatoai .
docker run -p 5000:5000 hepatoai
```

---

## 🔧 TROUBLESHOOTING

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Activate venv: `source venv/bin/activate` |
| `Model not found` | Run training: `python -m backend.ml_models.train` |
| `ImportError: cv2` | `pip install opencv-python-headless` |
| Port already in use | Change PORT in `.env` |
| TensorFlow GPU errors | Set `TF_CPP_MIN_LOG_LEVEL=3` in env |
| SHAP slow/timeout | Reduce background samples in `shap_explainer.py` |

---

## 📊 MODEL PERFORMANCE (Synthetic Data)

| Model | Accuracy | Use Case |
|-------|----------|----------|
| ANN (Tabular) | ~89% | Lab values only |
| CNN (ResNet50) | ~85% | Ultrasound only |
| Fusion | ~93% | Both modalities |

> Note: Accuracy on synthetic data. Real-world performance depends on dataset quality.

---

## 📚 REFERENCES

- He et al., "Deep Residual Learning for Image Recognition" (ResNet, 2016)
- Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions" (SHAP, 2017)
- Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks" (2017)
- Indian Liver Patient Dataset (ILPD) — UCI ML Repository

---

## 👨‍💻 DEVELOPED BY

B.Tech Final Year — Computer Science & Engineering  
AI & Medical Imaging Project, 2024  
**Tech Stack:** Python · TensorFlow · Flask · OpenCV · SHAP · SQLite · HTML/CSS/JS
