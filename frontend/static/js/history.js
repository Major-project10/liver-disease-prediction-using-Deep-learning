/* ============================================================
   HepatoAI — History Page JavaScript
   File: frontend/static/js/history.js
   ============================================================ */

let currentPage = 1;
const perPage = 10;

document.addEventListener('DOMContentLoaded', () => {
  loadStats();
  loadHistory();

  // Live search
  let searchTimeout;
  document.getElementById('searchInput').addEventListener('input', () => {
    clearTimeout(searchTimeout);
    searchTimeout = setTimeout(() => {
      currentPage = 1;
      loadHistory();
    }, 400);
  });
});

// ─── Load Stats ───────────────────────────────────────────────────────────────
async function loadStats() {
  try {
    const res = await fetch('/api/patients/stats/summary');
    const data = await res.json();

    document.getElementById('totalPatients').textContent    = data.total_patients    ?? '—';
    document.getElementById('totalPredictions').textContent = data.total_predictions ?? '—';
    document.getElementById('normalCount').textContent      = data.class_distribution?.Normal           ?? 0;
    document.getElementById('fattyCount').textContent       = data.class_distribution?.['Fatty Liver']  ?? 0;
    document.getElementById('cirrhosisCount').textContent   = data.class_distribution?.Cirrhosis        ?? 0;
  } catch (err) {
    console.error('Stats error:', err);
  }
}

// ─── Load History ─────────────────────────────────────────────────────────────
async function loadHistory() {
  const loading = document.getElementById('tableLoading');
  const table   = document.getElementById('historyTable');
  const empty   = document.getElementById('emptyState');

  loading.style.display = 'flex';
  table.style.display   = 'none';
  empty.style.display   = 'none';

  try {
    const search = document.getElementById('searchInput').value.trim();
    const cls    = document.getElementById('classFilter').value;
    const model  = document.getElementById('modelFilter').value;

    const params = new URLSearchParams({ page: currentPage, per_page: perPage });

    // FIX 1: was 'class'  → backend expects 'diagnosis'
    if (cls)    params.append('diagnosis',  cls);
    // FIX 2: was 'model'  → backend expects 'model_type'
    if (model)  params.append('model_type', model);
    // FIX 3: search was built but never appended to params — filtering did nothing
    if (search) params.append('search',     search);

    const res  = await fetch(`/api/patients/predictions/all?${params}`);
    const data = await res.json();

    loading.style.display = 'none';

    const predictions = data.predictions || [];

    if (predictions.length === 0) {
      empty.style.display = 'block';
      return;
    }

    table.style.display = 'table';
    renderTable(predictions, search);

    // FIX 4: was data.pages → backend now returns 'total_pages'
    renderPagination(data.total, data.total_pages ?? data.pages, data.current_page);

  } catch (err) {
    loading.style.display = 'none';
    empty.style.display   = 'block';
    console.error('History load error:', err);
  }
}

// ─── Render Table ─────────────────────────────────────────────────────────────
function renderTable(predictions, search = '') {
  const tbody = document.getElementById('historyBody');
  tbody.innerHTML = '';

  const classIcons  = { 'Normal': '✅', 'Fatty Liver': '⚠️', 'Cirrhosis': '🚨' };
  const classCssMap = { 'Normal': 'normal', 'Fatty Liver': 'fatty', 'Cirrhosis': 'cirrhosis' };
  const modelCssMap = { 'fusion': 'model-fusion', 'ann': 'model-ann', 'cnn': 'model-cnn' };
  const modelLabels = { 'fusion': 'Fusion', 'ann': 'ANN', 'cnn': 'CNN' };

  predictions.forEach((pred) => {
    const name     = highlightText(pred.patient_name || 'Anonymous', search);
    const cls      = pred.predicted_class;
    const clsCss   = classCssMap[cls]             || '';
    const modelCss = modelCssMap[pred.model_type] || '';
    const conf     = ((pred.confidence || 0) * 100).toFixed(1);
    const date     = pred.created_at
      ? new Date(pred.created_at).toLocaleDateString('en-IN', { day: '2-digit', month: 'short', year: 'numeric' })
      : '—';

    // FIX 5: age/gender were hardcoded as '—' — backend now returns them
    const ageSex = (pred.age && pred.gender) ? `${pred.age} / ${pred.gender}` : '—';

    const row = document.createElement('tr');
    row.innerHTML = `
      <td><span style="color:var(--text-muted);font-size:0.8rem;font-family:var(--font-mono);">#${pred.id || '—'}</span></td>
      <td>
        <div class="patient-cell">${name}</div>
        <div class="patient-id">Patient #${pred.patient_id || '—'}</div>
      </td>
      <td class="age-gender">${ageSex}</td>
      <td>
        <span class="class-pill ${clsCss}">
          ${classIcons[cls] || ''} ${cls}
        </span>
      </td>
      <td>
        <span class="confidence-val" style="color:${getConfColor(conf)}">${conf}%</span>
      </td>
      <td>
        <span class="model-badge ${modelCss}">${modelLabels[pred.model_type] || pred.model_type}</span>
      </td>
      <td class="date-cell">${date}</td>
      <td class="action-cell">
        <button class="btn-view" onclick="showDetail(${pred.id})">View</button>
      </td>
    `;
    tbody.appendChild(row);
  });
}

// ─── Pagination ───────────────────────────────────────────────────────────────
function renderPagination(total, pages, current) {
  const pag = document.getElementById('pagination');
  pag.innerHTML = '';
  if (!pages || pages <= 1) return;

  const prevBtn = createPageBtn('← Prev', current > 1, () => { currentPage--; loadHistory(); });
  pag.appendChild(prevBtn);

  // Windowed pagination — avoids flooding when there are many pages
  const start = Math.max(1, current - 2);
  const end   = Math.min(pages, current + 2);
  for (let i = start; i <= end; i++) {
    const btn = createPageBtn(i, true, () => { currentPage = i; loadHistory(); });
    if (i === current) btn.classList.add('active');
    pag.appendChild(btn);
  }

  const nextBtn = createPageBtn('Next →', current < pages, () => { currentPage++; loadHistory(); });
  pag.appendChild(nextBtn);
}

function createPageBtn(label, enabled, onClick) {
  const btn       = document.createElement('button');
  btn.className   = 'page-btn';
  btn.textContent = label;
  btn.disabled    = !enabled;
  if (enabled) btn.addEventListener('click', onClick);
  return btn;
}

// ─── Detail Modal ─────────────────────────────────────────────────────────────
async function showDetail(predictionId) {
  const modal = document.getElementById('modalOverlay');
  const body  = document.getElementById('modalBody');

  body.innerHTML = `<div style="text-align:center;padding:2rem;color:var(--text-muted)">
    <div class="loader" style="margin:0 auto 1rem"></div>
    Loading details...
  </div>`;
  modal.classList.add('open');

  try {
    // FIX 6: was fetching ALL predictions then doing .find() in JS —
    // completely breaks when the record is beyond the first 100.
    // Now calls the dedicated single-prediction endpoint.
    const res = await fetch(`/api/patients/predictions/${predictionId}`);
    if (!res.ok) throw new Error(`Server returned ${res.status}`);
    const data = await res.json();
    const pred = data.prediction ?? data;

    const cls   = pred.predicted_class;
    const icons = { 'Normal': '✅', 'Fatty Liver': '⚠️', 'Cirrhosis': '🚨' };

    // probabilities may be stored as a JSON string in SQLite
    const probs = (typeof pred.probabilities === 'string')
      ? JSON.parse(pred.probabilities)
      : (pred.probabilities || {});

    const conf = ((pred.confidence || 0) * 100).toFixed(1);
    const date = pred.created_at ? new Date(pred.created_at).toLocaleString() : '—';

    // Lab values are inlined on the prediction object by the backend
    const labKeys    = ['alt','ast','alp','bilirubin_total','bilirubin_direct','albumin','total_protein','ag_ratio'];
    const labEntries = labKeys.filter(k => pred[k] != null).map(k => [k, pred[k]]);

    body.innerHTML = `
      <div class="modal-section">
        <h4>Diagnosis Result</h4>
        <div style="display:flex;align-items:center;gap:12px;padding:1rem;background:var(--bg-secondary);border-radius:12px;margin-bottom:0.5rem">
          <span style="font-size:2.5rem">${icons[cls] || '🎯'}</span>
          <div>
            <div style="font-size:1.4rem;font-weight:700">${cls}</div>
            <div style="font-size:0.85rem;color:var(--text-muted)">Confidence: ${conf}% | Model: ${pred.model_type || '—'}</div>
          </div>
        </div>
      </div>

      ${Object.keys(probs).length > 0 ? `
      <div class="modal-section">
        <h4>Class Probabilities</h4>
        <div class="modal-grid">
          ${Object.entries(probs).map(([c, v]) => `
            <div class="modal-kv">
              <div class="modal-key">${c}</div>
              <div class="modal-val">${(v * 100).toFixed(1)}%</div>
            </div>`).join('')}
        </div>
      </div>` : ''}

      ${labEntries.length > 0 ? `
      <div class="modal-section">
        <h4>Lab Values</h4>
        <div class="modal-grid" style="grid-template-columns: 1fr 1fr 1fr">
          ${labEntries.map(([k, v]) => `
            <div class="modal-kv">
              <div class="modal-key">${k.replace(/_/g, ' ').toUpperCase()}</div>
              <div class="modal-val">${v ?? '—'}</div>
            </div>`).join('')}
        </div>
      </div>` : ''}

      <div class="modal-section">
        <h4>Record Info</h4>
        <div class="modal-grid">
          <div class="modal-kv"><div class="modal-key">Patient</div><div class="modal-val">${pred.patient_name || '—'}</div></div>
          <div class="modal-kv"><div class="modal-key">Age / Gender</div><div class="modal-val">${pred.age ?? '—'} / ${pred.gender ?? '—'}</div></div>
          <div class="modal-kv"><div class="modal-key">Date</div><div class="modal-val" style="font-size:0.8rem">${date}</div></div>
          <div class="modal-kv"><div class="modal-key">Record ID</div><div class="modal-val">#${pred.id}</div></div>
          <div class="modal-kv"><div class="modal-key">Patient ID</div><div class="modal-val">#${pred.patient_id}</div></div>
        </div>
      </div>

      ${pred.shap_plot_path ? `
      <div class="modal-section">
        <h4>🔍 SHAP Feature Importance</h4>
        <img class="modal-img" src="/explanations/${pred.shap_plot_path.split('/').pop()}" alt="SHAP" onerror="this.style.display='none'">
      </div>` : ''}

      ${pred.gradcam_plot_path ? `
      <div class="modal-section">
        <h4>🌡️ Grad-CAM Visualization</h4>
        <img class="modal-img" src="/explanations/${pred.gradcam_plot_path.split('/').pop()}" alt="Grad-CAM" onerror="this.style.display='none'">
      </div>` : ''}
    `;

  } catch (err) {
    console.error('Detail error:', err);
    document.getElementById('modalBody').innerHTML =
      `<p style="color:var(--accent-red)">Failed to load: ${err.message}</p>`;
  }
}

function closeModal() {
  document.getElementById('modalOverlay').classList.remove('open');
}

// ─── Utilities ────────────────────────────────────────────────────────────────
function getConfColor(conf) {
  if (conf >= 80) return 'var(--accent-green)';
  if (conf >= 60) return 'var(--accent-amber)';
  return 'var(--accent-red)';
}

function highlightText(text, search) {
  if (!search) return text;
  // FIX 7: escape regex special chars so inputs like "Mr. (test)" don't throw errors
  const escaped = search.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const re = new RegExp(`(${escaped})`, 'gi');
  return text.replace(re, '<mark style="background:rgba(59,130,246,0.3);color:white;border-radius:2px">$1</mark>');
}
