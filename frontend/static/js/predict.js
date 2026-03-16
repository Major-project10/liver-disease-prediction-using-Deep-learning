/* ============================================================
   HepatoAI — Predict Page JavaScript
   File: frontend/static/js/predict.js
   ============================================================ */

// ─── Sample Data ─────────────────────────────────────────────────────────────
const SAMPLE_DATA = {
  normal: {
    patient_name: 'Alex Johnson', age: 32, gender: 'Male',
    alt: 24, ast: 20, alp: 82,
    bilirubin_total: 0.8, bilirubin_direct: 0.2,
    albumin: 4.3, total_protein: 7.2, ag_ratio: 1.6,
  },
  fatty: {
    patient_name: 'Maria Silva', age: 48, gender: 'Female',
    alt: 72, ast: 58, alp: 118,
    bilirubin_total: 1.6, bilirubin_direct: 0.55,
    albumin: 3.7, total_protein: 6.6, ag_ratio: 1.1,
  },
  cirrhosis: {
    patient_name: 'David Chen', age: 62, gender: 'Male',
    alt: 95, ast: 130, alp: 210,
    bilirubin_total: 5.2, bilirubin_direct: 3.1,
    albumin: 2.6, total_protein: 5.5, ag_ratio: 0.75,
  }
};

// ─── Quick Fill ───────────────────────────────────────────────────────────────
function fillSampleData(type) {
  const data = SAMPLE_DATA[type];
  if (!data) return;

  document.getElementById('patient_name').value = data.patient_name;
  document.getElementById('age').value = data.age;
  document.getElementById('gender').value = data.gender;
  document.getElementById('alt').value = data.alt;
  document.getElementById('ast').value = data.ast;
  document.getElementById('alp').value = data.alp;
  document.getElementById('bilirubin_total').value = data.bilirubin_total;
  document.getElementById('bilirubin_direct').value = data.bilirubin_direct;
  document.getElementById('albumin').value = data.albumin;
  document.getElementById('total_protein').value = data.total_protein;
  document.getElementById('ag_ratio').value = data.ag_ratio;

  // Highlight the filled fields briefly
  document.querySelectorAll('.form-input').forEach(input => {
    input.style.borderColor = 'var(--accent-blue)';
    setTimeout(() => { input.style.borderColor = ''; }, 1000);
  });
}

// ─── Image Upload ─────────────────────────────────────────────────────────────
let uploadedFile = null;

document.addEventListener('DOMContentLoaded', () => {
  const zone = document.getElementById('uploadZone');
  const fileInput = document.getElementById('imageFile');
  const content = document.getElementById('uploadContent');
  const preview = document.getElementById('uploadPreview');
  const previewImg = document.getElementById('previewImg');
  const removeBtn = document.getElementById('removeImage');

  // Click to browse
  zone.addEventListener('click', (e) => {
    if (e.target !== removeBtn && !removeBtn.contains(e.target)) {
      fileInput.click();
    }
  });

  // File selected
  fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) handleImageFile(file);
  });

  // Drag and drop
  zone.addEventListener('dragover', (e) => {
    e.preventDefault();
    zone.classList.add('dragover');
  });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) handleImageFile(file);
  });

  removeBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    removeImage();
  });

  function handleImageFile(file) {
    uploadedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      content.style.display = 'none';
      preview.style.display = 'flex';
    };
    reader.readAsDataURL(file);
  }

  function removeImage() {
    uploadedFile = null;
    fileInput.value = '';
    previewImg.src = '';
    content.style.display = 'block';
    preview.style.display = 'none';
  }
});

// ─── Validation ───────────────────────────────────────────────────────────────
function validateForm() {
  const required = ['age', 'gender', 'alt', 'ast', 'alp',
                    'bilirubin_total', 'bilirubin_direct',
                    'albumin', 'total_protein', 'ag_ratio'];
  const missing = [];

  for (const field of required) {
    const el = document.getElementById(field);
    const val = el?.value?.trim();
    if (!val) {
      missing.push(field);
      el?.classList.add('input-error');
    } else {
      el?.classList.remove('input-error');
    }
  }

  if (missing.length > 0) {
    showToast(`Please fill in: ${missing.join(', ')}`, 'error');
    return false;
  }
  return true;
}

// ─── Main Prediction ──────────────────────────────────────────────────────────
async function runPrediction() {
  if (!validateForm()) return;

  const btn = document.getElementById('predictBtn');
  setLoading(btn, true);
  showResultsState('empty');

  try {
    const clinicalData = {
      age: parseFloat(document.getElementById('age').value),
      gender: document.getElementById('gender').value,
      alt: parseFloat(document.getElementById('alt').value),
      ast: parseFloat(document.getElementById('ast').value),
      alp: parseFloat(document.getElementById('alp').value),
      bilirubin_total: parseFloat(document.getElementById('bilirubin_total').value),
      bilirubin_direct: parseFloat(document.getElementById('bilirubin_direct').value),
      albumin: parseFloat(document.getElementById('albumin').value),
      total_protein: parseFloat(document.getElementById('total_protein').value),
      ag_ratio: parseFloat(document.getElementById('ag_ratio').value),
    };

    const patientName = document.getElementById('patient_name').value || 'Anonymous';

    // Build form data
    const formData = new FormData();
    formData.append('data', JSON.stringify(clinicalData));
    formData.append('patient_name', patientName);
    formData.append('generate_explanation', 'true');

    if (uploadedFile) {
      formData.append('image', uploadedFile);
    }

    const response = await fetch('/api/predict/fusion', {
      method: 'POST',
      body: formData,
    });

    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.error || 'Prediction failed');
    }

    displayResults(result);

  } catch (err) {
    console.error('Prediction error:', err);
    showResultsState('error', err.message,
      err.message.includes('model') ? 'Tip: Run python -m backend.ml_models.train first to train models.' : ''
    );
  } finally {
    setLoading(btn, false);
  }
}

// ─── Display Results ──────────────────────────────────────────────────────────
function displayResults(apiResponse) {
  const prediction = apiResponse.prediction;
  if (!prediction) {
    showResultsState('error', 'Invalid response from server');
    return;
  }

  showResultsState('results');

  const cls = prediction.predicted_class;
  const probs = prediction.probabilities || {};
  const explanation = prediction.explanation || {};

  // Class icon mapping
  const icons = { 'Normal': '✅', 'Fatty Liver': '⚠️', 'Cirrhosis': '🚨' };

  // Update result badge
  document.getElementById('resultIcon').textContent = icons[cls] || '🎯';
  document.getElementById('resultClass').textContent = cls;
  document.getElementById('resultConfidence').textContent = `Confidence: ${prediction.confidence_pct || '—'}`;

  // Risk badge
  const riskEl = document.getElementById('resultRisk');
  const riskClass = (prediction.risk_level || '').toLowerCase().replace(' ', '-');
  riskEl.textContent = `${prediction.risk_level || '—'} Risk`;
  riskEl.className = `result-risk-badge risk-${riskClass}`;

  // Color code result header based on class
  const headerColors = {
    'Normal': 'rgba(16,185,129,0.07)',
    'Fatty Liver': 'rgba(245,158,11,0.07)',
    'Cirrhosis': 'rgba(239,68,68,0.07)'
  };
  document.querySelector('.result-header').style.background = headerColors[cls] || 'var(--bg-secondary)';

  // Probability bars (animate after small delay)
  const probMap = {
    'Normal': { pct: document.getElementById('probNormal'), bar: document.getElementById('barNormal') },
    'Fatty Liver': { pct: document.getElementById('probFatty'), bar: document.getElementById('barFatty') },
    'Cirrhosis': { pct: document.getElementById('probCirrhosis'), bar: document.getElementById('barCirrhosis') }
  };

  setTimeout(() => {
    for (const [cls_name, val] of Object.entries(probs)) {
      const els = probMap[cls_name];
      if (els) {
        const pct = (val * 100).toFixed(1);
        els.pct.textContent = `${pct}%`;
        els.bar.style.width = `${pct}%`;
      }
    }
  }, 100);

  // Model info
  const modelLabels = { fusion: '🔀 Fusion (ANN+CNN)', ann: '🧠 ANN (Tabular)', cnn: '👁️ CNN (Image)' };
  document.getElementById('modelType').textContent = modelLabels[prediction.model_type] || prediction.model_type;
  document.getElementById('patientId').textContent = prediction.patient_id ? `#${prediction.patient_id}` : '—';
  document.getElementById('recordId').textContent = prediction.record_id ? `#${prediction.record_id}` : '—';

  // SHAP section
  const shapData = explanation.shap || {};
  if (shapData.top_features && shapData.top_features.length > 0) {
    document.getElementById('shapSection').style.display = 'block';

    const topFeaturesEl = document.getElementById('topFeatures');
    topFeaturesEl.innerHTML = '';
    shapData.top_features.slice(0, 5).forEach(([name, val], i) => {
      topFeaturesEl.innerHTML += `
        <div class="top-feature-item">
          <div class="top-feature-rank">${i + 1}</div>
          <div class="top-feature-name">${name}</div>
          <div class="top-feature-val">${parseFloat(val).toFixed(4)}</div>
        </div>`;
    });

    if (shapData.plot_path) {
      const shapPlot = document.getElementById('shapPlot');
      const shapImg = document.getElementById('shapImg');
      const imgFilename = shapData.plot_path.split('/').pop();
      shapImg.src = `/explanations/${imgFilename}`;
      shapImg.onload = () => { shapPlot.style.display = 'block'; };
      shapImg.onerror = () => { shapPlot.style.display = 'none'; };
    }
  }

  // Grad-CAM section
  const gradcamData = explanation.gradcam || {};
  if (gradcamData.plot_path) {
    document.getElementById('gradcamSection').style.display = 'block';
    const gradImg = document.getElementById('gradcamImg');
    const imgFilename = gradcamData.plot_path.split('/').pop();
    gradImg.src = `/explanations/${imgFilename}`;
    gradImg.onerror = () => { document.getElementById('gradcamSection').style.display = 'none'; };
  }

  // Scroll to results on mobile
  if (window.innerWidth < 1100) {
    document.getElementById('resultsPanel').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}

// ─── UI State Helpers ─────────────────────────────────────────────────────────
function showResultsState(state, errorMsg = '', errorHint = '') {
  document.getElementById('resultsEmpty').style.display = state === 'empty' ? 'flex' : 'none';
  document.getElementById('resultsContent').style.display = state === 'results' ? 'block' : 'none';
  document.getElementById('resultsError').style.display = state === 'error' ? 'flex' : 'none';

  if (state === 'error') {
    document.getElementById('errorMessage').textContent = errorMsg;
    document.getElementById('errorHint').textContent = errorHint;
  }

  if (state === 'results') {
    // Reset SHAP/Grad-CAM visibility
    document.getElementById('shapSection').style.display = 'none';
    document.getElementById('gradcamSection').style.display = 'none';
    document.getElementById('shapPlot').style.display = 'none';
    // Reset prob bars
    ['barNormal', 'barFatty', 'barCirrhosis'].forEach(id => {
      document.getElementById(id).style.width = '0%';
    });
  }
}

function setLoading(btn, loading) {
  const btnContent = btn.querySelector('.predict-btn-content');
  const loadingContent = btn.querySelector('.loading-content');
  btn.disabled = loading;
  btnContent.style.display = loading ? 'none' : 'flex';
  loadingContent.style.display = loading ? 'flex' : 'none';
}

// ─── Utility Actions ──────────────────────────────────────────────────────────
function printResult() { window.print(); }

function newPrediction() {
  showResultsState('empty');
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function showToast(message, type = 'info') {
  const toast = document.createElement('div');
  toast.style.cssText = `
    position: fixed; bottom: 24px; right: 24px; z-index: 9999;
    background: ${type === 'error' ? 'rgba(239,68,68,0.95)' : 'rgba(59,130,246,0.95)'};
    color: white; padding: 12px 20px; border-radius: 10px;
    font-family: 'Outfit', sans-serif; font-size: 0.9rem; font-weight: 500;
    box-shadow: 0 8px 24px rgba(0,0,0,0.4);
    animation: slideIn 0.3s ease;
    max-width: 350px;
  `;
  toast.textContent = message;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 4000);
}
