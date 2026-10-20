// TBI-AI Platform Frontend JavaScript

const API_URL = window.location.origin;
const app = document.getElementById('app');
const toast = document.getElementById('toast');
const apiHealthBadge = document.getElementById('apiHealthBadge');
const modal = document.getElementById('imageModal');
const modalImage = document.getElementById('modalImage');
const modalCloseBtn = document.getElementById('modalCloseBtn');

const MODEL_PERFORMANCE = {
  baselineAccuracy: 44.25,
  cnnAccuracy: 45.69,
  macroF1: 44.89,
  balancedAccuracy: 45.36,
  binaryAccuracy: 87.86,
  binaryMacroF1: 76.22,
  testImages: 626,
  correctPredictions: 286,
  unetDice: 6.56,
  unetIoU: 3.39,
  unetPrecision: 20.45,
  unetRecall: 3.91,
  unetSlices: 1538,
  unetThreshold: 0.25,
  classPerformance: [
    { className: 'Epidural', precision: 0.76, recall: 0.53, f1: 0.63 },
    { className: 'Intraparenchymal', precision: 0.31, recall: 0.28, f1: 0.30 },
    { className: 'Intraventricular', precision: 0.44, recall: 0.62, f1: 0.51 },
    { className: 'Normal', precision: 0.58, recall: 0.64, f1: 0.61 },
    { className: 'Subarachnoid', precision: 0.31, recall: 0.26, f1: 0.28 },
    { className: 'Subdural', precision: 0.34, recall: 0.39, f1: 0.36 },
  ],
};

const STORAGE_KEYS = {
  latestResult: 'tbi-ai-latest-result',
  history: 'tbi-ai-history',
};

const DEFAULT_ROUTE = '#/home';
let currentPrediction = null;
let currentFile = null;
let selectedImagePreview = null;
let currentTab = 'original';
let zoomScale = 1;
let panX = 0;
let panY = 0;
let isPanning = false;
let startPanX = 0;
let startPanY = 0;

// Utility functions
function showToast(message, isError = false) {
  if (!toast) return;
  toast.textContent = message;
  toast.style.borderColor = isError ? 'var(--danger-color)' : 'var(--primary-cyan)';
  toast.classList.remove('hidden');
  clearTimeout(showToast.timeoutId);
  showToast.timeoutId = setTimeout(() => {
    toast.classList.add('hidden');
  }, 3000);
}

function formattedPercent(value) {
  if (Number.isFinite(value)) return `${(value * 100).toFixed(2)}%`;
  return '0.00%';
}

function sanitizeText(text) {
  return String(text ?? '').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function storeLatestResult(result) {
  try {
    localStorage.setItem(STORAGE_KEYS.latestResult, JSON.stringify(result));
    let history = JSON.parse(localStorage.getItem(STORAGE_KEYS.history) || '[]');
    history.unshift({
      filename: result.filename || 'ct_scan.png',
      prediction: result.prediction || 'Unknown',
      confidence: result.confidence || 0,
      timestamp: result.timestamp || new Date().toLocaleString(),
    });
    history = history.slice(0, 10);
    localStorage.setItem(STORAGE_KEYS.history, JSON.stringify(history));
  } catch (err) {
    console.warn('Could not store result in localStorage', err);
  }
}

function loadLatestResult() {
  try {
    const val = localStorage.getItem(STORAGE_KEYS.latestResult);
    return val ? JSON.parse(val) : null;
  } catch (err) {
    return null;
  }
}

// Check Backend Health
async function checkApiHealth() {
  if (!apiHealthBadge) return;
  try {
    const res = await fetch(`${API_URL}/health`);
    if (res.ok) {
      const data = await res.json();
      apiHealthBadge.className = 'status-pill status-online';
      apiHealthBadge.innerHTML = `<span class="status-dot"></span><span class="status-text">API Online | Models Loaded</span>`;
    } else {
      throw new Error('Non-200 health response');
    }
  } catch (err) {
    apiHealthBadge.className = 'status-pill status-offline';
    apiHealthBadge.innerHTML = `<span class="status-dot"></span><span class="status-text">Backend Offline</span>`;
  }
}

// Modal handling
function openModal(imgSrc) {
  if (!modal || !modalImage) return;
  modalImage.src = imgSrc;
  modal.classList.remove('hidden');
}

if (modalCloseBtn) {
  modalCloseBtn.addEventListener('click', () => modal.classList.add('hidden'));
}
if (modal) {
  modal.addEventListener('click', (e) => {
    if (e.target === modal) modal.classList.add('hidden');
  });
}

// Router
function getRoute() {
  const hash = window.location.hash || DEFAULT_ROUTE;
  if (hash.startsWith('#/')) return hash;
  return DEFAULT_ROUTE;
}

function setActiveNav(route) {
  const links = document.querySelectorAll('.nav-link');
  links.forEach((link) => {
    const targetRoute = link.getAttribute('href');
    link.classList.toggle('active', targetRoute === route);
  });
}

// Render Pages
function renderHomePage() {
  return `
    <section class="page">
      <div class="hero">
        <div class="hero-copy">
          <div class="section-kicker">Academic Research Platform</div>
          <h1>Traumatic Brain Injury Detection</h1>
          <p class="subtitle">Deep learning-assisted analysis of head CT scans for 6-class intracranial hemorrhage prediction and experimental lesion segmentation.</p>
          <div class="hero-actions">
            <a class="btn primary" href="#/analyze">Analyze CT Scan →</a>
            <a class="btn secondary" href="#/performance">View Model Metrics</a>
          </div>
        </div>
        <div class="hero-panel">
          <div class="kpi-card" style="margin-bottom: 16px;">
            <strong>6 Classes</strong>
            <span>Hemorrhage Categories</span>
          </div>
          <div class="kpi-row">
            <div class="kpi-box">
              <strong>45.69%</strong>
              <span>Six-Class Accuracy</span>
            </div>
            <div class="kpi-box">
              <strong>87.86%</strong>
              <span>Binary Accuracy</span>
            </div>
            <div class="kpi-box">
              <strong>6.56%</strong>
              <span>U-Net Dice</span>
            </div>
            <div class="kpi-box">
              <strong>6,000</strong>
              <span>CT Dataset</span>
            </div>
          </div>
        </div>
      </div>

      <div class="section-kicker">Capabilities</div>
      <h2 style="font-size: 1.8rem; font-weight: 800; margin-bottom: 24px;">Platform Architecture</h2>

      <div class="feature-grid">
        <div class="feature-card">
          <div class="feature-icon">🧠</div>
          <h3>Verified Two-Output Classification</h3>
          <p>Reports normal-vs-TBI screening plus six-class injury probabilities from the selected EfficientNetB0-embedding ensemble.</p>
        </div>
        <div class="feature-card">
          <div class="feature-icon">🩺</div>
          <h3>Experimental U-Net Branch</h3>
          <p>Outlines suspect lesion regions using 256×256 grayscale input with a threshold of 0.25.</p>
        </div>
        <div class="feature-card">
          <div class="feature-icon">🖼️</div>
          <h3>Interactive CT Viewer</h3>
          <p>Allows side-by-side inspection of Original CT, Segmentation Mask, and Red Overlay with zoom and pan controls.</p>
        </div>
      </div>
    </section>
  `;
}

function renderAnalyzePage() {
  return `
    <section class="page">
      <div class="page-header">
        <div class="section-kicker">CT Scan Analysis</div>
        <h1>Upload CT Scan</h1>
        <p>Upload a head CT scan image (PNG, JPG, JPEG, DICOM .dcm) to execute inference through the classification and segmentation models.</p>
      </div>

      <div class="workspace-grid">
        <div class="panel-card">
          <div id="dropzone" class="dropzone" tabindex="0">
            <input id="fileInput" type="file" accept=".png,.jpg,.jpeg,.dcm" hidden />
            <div class="upload-icon-circle">⬆</div>
            <div class="dropzone-text">
              <p><strong>Drag & drop</strong> CT scan here, or click to browse</p>
              <p class="accepted-types">Supported formats: PNG, JPG, JPEG, DICOM (.dcm) • Max size: 25MB</p>
            </div>
            <button type="button" class="btn secondary" id="browseBtn" style="margin-top: 16px;">Browse Files</button>
          </div>

          <div id="previewBox" class="preview-box hidden">
            <img id="previewImage" class="preview-thumbnail" alt="CT scan preview" />
            <div class="preview-meta">
              <div class="meta-item">
                <div class="meta-label">File Name</div>
                <div class="meta-val" id="metaFileName">-</div>
              </div>
              <div class="meta-item">
                <div class="meta-label">File Size</div>
                <div class="meta-val" id="metaFileSize">-</div>
              </div>
              <div class="meta-item">
                <div class="meta-label">Dimensions</div>
                <div class="meta-val" id="metaDimensions">-</div>
              </div>
              <div class="meta-item">
                <div class="meta-label">Format</div>
                <div class="meta-val" id="metaFileType">-</div>
              </div>
            </div>
          </div>

          <div class="btn-row" style="display: flex; gap: 14px; margin-top: 24px;">
            <button type="button" class="btn danger hidden" id="removeFileBtn">Remove File</button>
            <button type="button" class="btn primary" id="analyzeBtn" disabled>Analyze CT Scan</button>
          </div>
        </div>

        <aside class="panel-card">
          <div class="section-kicker">Inference Pipeline</div>
          <h3 style="font-size: 1.3rem; margin-bottom: 12px;">Pre-Inference Notes</h3>
          <ul style="color: var(--text-muted); font-size: 0.92rem; line-height: 1.8; margin-left: 18px; margin-bottom: 20px;">
            <li>Classifier receives 224 × 224 RGB unscaled inputs.</li>
            <li>U-Net branch receives 256 × 256 grayscale normalized inputs.</li>
            <li>Segmentation threshold is fixed at 0.25.</li>
          </ul>

          <div class="warning-callout">
            <span>⚠️</span>
            <div>
              <strong>Academic Prototype Notice:</strong> Model outputs are for research evaluation only and are not validated for clinical diagnosis.
            </div>
          </div>
        </aside>
      </div>

      <!-- Loading State Overlay -->
      <div id="loadingOverlay" class="loading-overlay hidden" style="margin-top: 32px;">
        <div class="spinner-wrap">
          <div class="spinner"></div>
        </div>
        <h2 style="font-size: 1.5rem; font-weight: 800;">ANALYZING CT SCAN</h2>
        <p style="color: var(--text-muted); margin-top: 6px;">Running inference pipelines through saved model weights...</p>
        
        <div class="step-list">
          <div class="step-item done" id="step1">
            <span class="step-icon">✓</span>
            <span>Image received & format validated</span>
          </div>
          <div class="step-item active" id="step2">
            <span class="step-icon">●</span>
            <span>Preprocessing & intensity windowing</span>
          </div>
          <div class="step-item" id="step3">
            <span class="step-icon">○</span>
            <span>Running binary detector and six-class ensemble</span>
          </div>
          <div class="step-item" id="step4">
            <span class="step-icon">○</span>
            <span>Running experimental U-Net segmentation</span>
          </div>
        </div>
      </div>
    </section>
  `;
}

function renderResultsPage() {
  const result = currentPrediction || loadLatestResult();

  if (!result) {
    return `
      <section class="page">
        <div class="page-header">
          <div class="section-kicker">Results</div>
          <h1>No Analysis Result Available</h1>
          <p>Upload a CT scan from the Analyze page to run model inference.</p>
        </div>
        <div style="margin-top: 24px;">
          <a href="#/analyze" class="btn primary">Go to Analyze Page →</a>
        </div>
      </section>
    `;
  }

  const classProbs = result.class_probabilities || result.classification?.probabilities || {};
  const predictedClass = result.prediction || result.classification?.predicted_class || 'Unknown';
  const confidence = Number(result.confidence || result.classification?.confidence || 0);
  const tbiStatus = result.tbi_status || result.classification?.tbi_status || 'Unknown';
  const tbiProbability = Number(result.tbi_probability || result.classification?.tbi_probability || 0);
  const injuryType = result.injury_type || result.classification?.injury_type || 'Not applicable';
  const segData = result.segmentation || {};

  const classesOrdered = [
    'Epidural',
    'Intraparenchymal',
    'Intraventricular',
    'Normal',
    'Subarachnoid',
    'Subdural',
  ];

  const probRowsHtml = classesOrdered
    .map((cls) => {
      const prob = Number(classProbs[cls] || 0);
      const isPredicted = cls === predictedClass;
      return `
      <div class="probability-row ${isPredicted ? 'highlight' : ''}">
        <div class="probability-head">
          <span>${cls} ${isPredicted ? '★ (Predicted Class)' : ''}</span>
          <span>${formattedPercent(prob)}</span>
        </div>
        <div class="bar-track">
          <div class="bar-fill" style="width: ${Math.max(prob * 100, 2)}%"></div>
        </div>
      </div>
    `;
    })
    .join('');

  return `
    <section class="page">
      <div class="page-header">
        <div class="section-kicker">Inference Results</div>
        <h1>CT Analysis Results</h1>
        <p>Real predictions returned from the verified binary detector, six-class ensemble, and U-Net.</p>
      </div>

      <div class="prediction-summary-card">
        <div class="prediction-top">
          <div class="section-kicker">Classification Outcome</div>
          <div class="prediction-pill">Research Prototype</div>
        </div>

        <div class="predicted-condition-display">
          <div class="pred-block">
            <div class="label">TBI Screening</div>
            <div class="value">${sanitizeText(tbiStatus)}</div>
            <div class="label">Abnormal probability: ${formattedPercent(tbiProbability)}</div>
          </div>
          <div class="pred-block">
            <div class="label">Six-Class Prediction</div>
            <div class="value">${sanitizeText(predictedClass)}</div>
          </div>
          <div class="pred-block">
            <div class="label">Injury Type / Confidence</div>
            <div class="value">${sanitizeText(injuryType)}</div>
            <div class="value-conf">${formattedPercent(confidence)}</div>
          </div>
        </div>

        <h3 style="font-size: 1.1rem; font-weight: 700; margin-bottom: 16px;">Six-Class Probability Distribution</h3>
        <div class="probability-list">
          ${probRowsHtml}
        </div>
      </div>

      <!-- CT Viewer -->
      <div class="viewer-card">
        <div class="viewer-header">
          <div>
            <div class="section-kicker">Interactive CT Viewer</div>
            <h2 style="font-size: 1.5rem; font-weight: 800;">Imaging & Segmentation Visualization</h2>
          </div>
          <div class="tabs">
            <button type="button" class="tab-btn active" data-tab="original">Original CT</button>
            <button type="button" class="tab-btn" data-tab="mask">Segmentation Mask</button>
            <button type="button" class="tab-btn" data-tab="overlay">Overlay</button>
          </div>
        </div>

        <div class="viewer-panel">
          <div class="viewer-stage" id="viewerStage">
            <img id="viewerImage" src="${result.original_image_url || ''}" alt="CT Analysis View" />
          </div>
          <div class="viewer-toolbar">
            <button type="button" class="viewer-btn" id="zoomInBtn">🔍 Zoom In (+)</button>
            <button type="button" class="viewer-btn" id="zoomOutBtn">🔍 Zoom Out (-)</button>
            <button type="button" class="viewer-btn" id="resetZoomBtn">🔄 Reset View</button>
            <button type="button" class="viewer-btn" id="fullscreenBtn">⛶ Fullscreen</button>
          </div>
        </div>

        <div class="warning-callout">
          <span>⚠️</span>
          <div>
            <strong>EXPERIMENTAL U-NET SEGMENTATION:</strong> ${segData.warning || 'Current model requires further optimization and is not validated for clinical use.'}
          </div>
        </div>
      </div>

      <!-- Action Buttons -->
      <div style="display: flex; gap: 16px; margin-top: 32px; flex-wrap: wrap;">
        <button type="button" class="btn primary" id="downloadReportBtn">📥 Download AI Analysis Summary</button>
        <button type="button" class="btn secondary" id="clearAnalysisBtn">Clear Analysis</button>
      </div>
    </section>
  `;
}

function renderPerformancePage() {
  const p = MODEL_PERFORMANCE;

  const tableRowsHtml = p.classPerformance
    .map(
      (r) => `
    <tr>
      <td style="font-weight: 700;">${r.className}</td>
      <td style="font-family: var(--font-mono);">${r.precision.toFixed(2)}</td>
      <td style="font-family: var(--font-mono);">${r.recall.toFixed(2)}</td>
      <td style="font-family: var(--font-mono);">${r.f1.toFixed(2)}</td>
    </tr>
  `
    )
    .join('');

  return `
    <section class="page">
      <div class="page-header">
        <div class="section-kicker">Verified Project Metrics</div>
        <h1>Model Performance</h1>
        <p>Quantitative metrics computed from the saved model checkpoints on the test datasets.</p>
      </div>

      <div class="feature-grid" style="margin-bottom: 36px;">
        <div class="card">
          <div class="section-kicker">Verified Six-Class Ensemble</div>
          <div style="font-size: 3rem; font-weight: 900; color: var(--primary-cyan); font-family: var(--font-mono); line-height: 1;">${p.cnnAccuracy}%</div>
          <div style="font-size: 0.88rem; color: var(--text-muted); margin-top: 4px;">Test Classification Accuracy</div>
          <div style="font-size: 0.82rem; color: var(--text-muted); margin-top: 8px;">Baseline ${p.baselineAccuracy}% → Improved ${p.cnnAccuracy}%</div>
          <div class="kpi-row" style="margin-top: 20px;">
            <div class="kpi-box">
              <strong>${p.testImages}</strong>
              <span>Test Images</span>
            </div>
            <div class="kpi-box">
              <strong>${p.correctPredictions}</strong>
              <span>Correct</span>
            </div>
            <div class="kpi-box">
              <strong>${p.macroF1}%</strong>
              <span>Macro F1</span>
            </div>
            <div class="kpi-box">
              <strong>${p.balancedAccuracy}%</strong>
              <span>Balanced Accuracy</span>
            </div>
          </div>
        </div>

        <div class="card">
          <div class="section-kicker">Normal vs TBI / Abnormal</div>
          <div style="font-size: 3rem; font-weight: 900; color: var(--primary-cyan); font-family: var(--font-mono); line-height: 1;">${p.binaryAccuracy}%</div>
          <div style="font-size: 0.88rem; color: var(--text-muted); margin-top: 4px;">Frozen-Test Binary Accuracy</div>
          <div class="kpi-row" style="margin-top: 20px;">
            <div class="kpi-box">
              <strong>${p.binaryMacroF1}%</strong>
              <span>Macro F1</span>
            </div>
          </div>
        </div>

        <div class="card">
          <div class="section-kicker">Experimental U-Net Segmentation</div>
          <div style="font-size: 3rem; font-weight: 900; color: var(--primary-teal); font-family: var(--font-mono); line-height: 1;">${p.unetDice}%</div>
          <div style="font-size: 0.88rem; color: var(--text-muted); margin-top: 4px;">Dice Coefficient (Threshold: ${p.unetThreshold})</div>
          <div class="kpi-row" style="margin-top: 20px;">
            <div class="kpi-box">
              <strong>${p.unetIoU}%</strong>
              <span>IoU</span>
            </div>
            <div class="kpi-box">
              <strong>${p.unetPrecision}%</strong>
              <span>Precision</span>
            </div>
            <div class="kpi-box">
              <strong>${p.unetRecall}%</strong>
              <span>Recall</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Class Performance Table -->
      <div class="card" style="margin-bottom: 36px;">
        <div class="section-kicker">Classification Evaluation</div>
        <h2 style="font-size: 1.4rem; font-weight: 800; margin-bottom: 16px;">Per-Class Performance Report</h2>
        <div class="performance-table-wrap">
          <table class="performance-table">
            <thead>
              <tr>
                <th>Hemorrhage Category</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1-Score</th>
              </tr>
            </thead>
            <tbody>
              ${tableRowsHtml}
            </tbody>
          </table>
        </div>
      </div>

      <!-- Confusion Matrix Card -->
      <div class="card">
        <div class="section-kicker">Confusion Matrix</div>
        <h2 style="font-size: 1.4rem; font-weight: 800; margin-bottom: 8px;">TBI-6000 Test Set Confusion Matrix</h2>
        <p style="color: var(--text-muted); font-size: 0.95rem; margin-bottom: 16px;">Click matrix image to inspect in high resolution.</p>
        <div class="confusion-box" id="confusionBox">
          <img src="/static/confusion_matrix.png" alt="TBI-6000 Test Set Confusion Matrix" id="confusionImg" />
        </div>
      </div>
    </section>
  `;
}

function renderResearchPage() {
  return `
    <section class="page">
      <div class="page-header">
        <div class="section-kicker">Research & Dataset</div>
        <h1>Research Documentation</h1>
        <p>Overview of project objectives, dataset structures, and deep learning pipeline architecture.</p>
      </div>

      <div class="feature-grid" style="margin-bottom: 32px;">
        <div class="card">
          <div class="section-kicker">Dataset 1</div>
          <h3 style="font-size: 1.4rem; font-weight: 800; margin-bottom: 10px;">TBI-6000 Classification</h3>
          <p style="color: var(--text-muted); margin-bottom: 16px;">6,000 head CT images balanced across 6 categories (1,000 images per class).</p>
          <ul style="color: var(--text-muted); font-size: 0.9rem; line-height: 1.8; margin-left: 18px;">
            <li>Epidural</li>
            <li>Intraparenchymal</li>
            <li>Intraventricular</li>
            <li>Normal</li>
            <li>Subarachnoid</li>
            <li>Subdural</li>
          </ul>
        </div>

        <div class="card">
          <div class="section-kicker">Dataset 2</div>
          <h3 style="font-size: 1.4rem; font-weight: 800; margin-bottom: 10px;">U-Net Lesion Segmentation</h3>
          <p style="color: var(--text-muted); margin-bottom: 16px;">1,538 test slice images paired with ground-truth lesion masks.</p>
          <ul style="color: var(--text-muted); font-size: 0.9rem; line-height: 1.8; margin-left: 18px;">
            <li>Input resolution: 256 × 256 grayscale</li>
            <li>Threshold: 0.25</li>
            <li>Metrics: Dice 6.56%, IoU 3.39%</li>
          </ul>
        </div>
      </div>

      <div class="card">
        <div class="section-kicker">Architecture Diagram</div>
        <h2 style="font-size: 1.4rem; font-weight: 800; margin-bottom: 16px;">Dual-Branch Pipeline</h2>
        <div style="background: rgba(10, 20, 32, 0.8); border: 1px solid var(--border-color); border-radius: var(--radius-md); padding: 24px; text-align: center;">
          <div style="display: flex; flex-direction: column; gap: 20px; align-items: center;">
            <div style="padding: 10px 24px; background: rgba(77, 231, 255, 0.15); border: 1px solid var(--primary-cyan); border-radius: 8px; font-weight: 800;">
              Input CT Scan Image
            </div>
            <div style="font-size: 1.2rem; color: var(--text-muted);">↙ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ↘</div>
            <div style="display: flex; gap: 40px; justify-content: center; flex-wrap: wrap;">
              <div style="background: rgba(15, 29, 46, 0.9); border: 1px solid var(--border-color); padding: 20px; border-radius: 12px; max-width: 300px;">
                <div style="color: var(--primary-cyan); font-weight: 800; margin-bottom: 6px;">Classification Branch</div>
                <div style="font-size: 0.88rem; color: var(--text-muted);">224×224 RGB → EfficientNetB0 → Softmax Soft Probabilities → Predicted Category</div>
              </div>
              <div style="background: rgba(15, 29, 46, 0.9); border: 1px solid var(--border-color); padding: 20px; border-radius: 12px; max-width: 300px;">
                <div style="color: var(--primary-teal); font-weight: 800; margin-bottom: 6px;">Segmentation Branch</div>
                <div style="font-size: 0.88rem; color: var(--text-muted);">256×256 Grayscale [0,1] → U-Net → Threshold 0.25 → Experimental Mask & Overlay</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  `;
}

function renderAboutPage() {
  return `
    <section class="page">
      <div class="page-header">
        <div class="section-kicker">About</div>
        <h1>Traumatic Brain Injury Detection</h1>
        <p>Academic Deep Learning Research Project</p>
      </div>

      <div class="feature-grid">
        <div class="card">
          <h3>Project Objective</h3>
          <p style="color: var(--text-muted); margin-top: 8px;">Development and evaluation of automated deep learning models for classification of 6 TBI hemorrhage types and experimental lesion region segmentation from head CT scans.</p>
        </div>
        <div class="card">
          <h3>Technology Stack</h3>
          <p style="color: var(--text-muted); margin-top: 8px;">FastAPI, TensorFlow 2.21, Keras, EfficientNetB0, U-Net, PyDICOM, PIL, Vanilla HTML/CSS/JS.</p>
        </div>
        <div class="card">
          <h3>Academic Scope</h3>
          <p style="color: var(--text-muted); margin-top: 8px;">Designed purely for scientific research prototype demonstration and method benchmark review.</p>
        </div>
      </div>
    </section>
  `;
}

// Router dispatch
function renderPage() {
  const route = getRoute();
  setActiveNav(route);

  let html = '';
  if (route === '#/home') html = renderHomePage();
  else if (route === '#/analyze') html = renderAnalyzePage();
  else if (route === '#/results') html = renderResultsPage();
  else if (route === '#/performance') html = renderPerformancePage();
  else if (route === '#/research') html = renderResearchPage();
  else if (route === '#/about') html = renderAboutPage();
  else html = renderHomePage();

  app.innerHTML = html;
  window.scrollTo(0, 0);
  wirePageEvents(route);
}

// Wire interactive events after render
function wirePageEvents(route) {
  if (route === '#/analyze') {
    wireAnalyzePage();
  } else if (route === '#/results') {
    wireResultsPage();
  } else if (route === '#/performance') {
    const confBox = document.getElementById('confusionBox');
    if (confBox) {
      confBox.addEventListener('click', () => openModal('/static/confusion_matrix.png'));
    }
  }
}

// Analyze Page Interactions
function wireAnalyzePage() {
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const browseBtn = document.getElementById('browseBtn');
  const previewBox = document.getElementById('previewBox');
  const previewImage = document.getElementById('previewImage');
  const metaFileName = document.getElementById('metaFileName');
  const metaFileSize = document.getElementById('metaFileSize');
  const metaDimensions = document.getElementById('metaDimensions');
  const metaFileType = document.getElementById('metaFileType');
  const removeFileBtn = document.getElementById('removeFileBtn');
  const analyzeBtn = document.getElementById('analyzeBtn');
  const loadingOverlay = document.getElementById('loadingOverlay');

  if (browseBtn && fileInput) {
    browseBtn.addEventListener('click', () => fileInput.click());
  }

  if (dropzone) {
    dropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });
    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
    dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleFileSelected(e.dataTransfer.files[0]);
      }
    });
  }

  if (fileInput) {
    fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleFileSelected(e.target.files[0]);
      }
    });
  }

  function handleFileSelected(file) {
    const ext = file.name.split('.').pop().toLowerCase();
    const allowed = ['png', 'jpg', 'jpeg', 'dcm'];
    if (!allowed.includes(ext)) {
      showToast(`Unsupported file extension '.${ext}'. Please choose PNG, JPG, JPEG, or DICOM (.dcm).`, true);
      return;
    }

    if (file.size > 25 * 1024 * 1024) {
      showToast('File size exceeds 25MB limit.', true);
      return;
    }

    currentFile = file;
    metaFileName.textContent = file.name;
    metaFileSize.textContent = `${(file.size / (1024 * 1024)).toFixed(2)} MB`;
    metaFileType.textContent = ext.toUpperCase();

    // Create thumbnail preview
    if (ext === 'dcm') {
      previewImage.src = 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 100 100"><rect width="100" height="100" fill="%230c1827"/><text x="50" y="55" font-size="14" fill="%234de7ff" text-anchor="middle" font-weight="bold">DICOM</text></svg>';
      metaDimensions.textContent = 'DICOM Array';
    } else {
      const reader = new FileReader();
      reader.onload = (e) => {
        previewImage.src = e.target.result;
        const imgObj = new Image();
        imgObj.src = e.target.result;
        imgObj.onload = () => {
          metaDimensions.textContent = `${imgObj.width} × ${imgObj.height}`;
        };
      };
      reader.readAsDataURL(file);
    }

    previewBox.classList.remove('hidden');
    removeFileBtn.classList.remove('hidden');
    analyzeBtn.disabled = false;
  }

  if (removeFileBtn) {
    removeFileBtn.addEventListener('click', () => {
      currentFile = null;
      previewBox.classList.add('hidden');
      removeFileBtn.classList.add('hidden');
      analyzeBtn.disabled = true;
      if (fileInput) fileInput.value = '';
    });
  }

  if (analyzeBtn) {
    analyzeBtn.addEventListener('click', async () => {
      if (!currentFile) return;

      loadingOverlay.classList.remove('hidden');
      analyzeBtn.disabled = true;

      const formData = new FormData();
      formData.append('file', currentFile);

      try {
        const res = await fetch(`${API_URL}/predict`, {
          method: 'POST',
          body: formData,
        });

        if (!res.ok) {
          const errJson = await res.json();
          throw new Error(errJson.detail || 'Inference call failed');
        }

        const data = await res.json();
        currentPrediction = data;
        storeLatestResult(data);

        showToast('Inference completed successfully!');
        setTimeout(() => {
          window.location.hash = '#/results';
        }, 400);

      } catch (err) {
        showToast(`Inference error: ${err.message}`, true);
        loadingOverlay.classList.add('hidden');
        analyzeBtn.disabled = false;
      }
    });
  }
}

// Results Page CT Viewer & Action Handlers
function wireResultsPage() {
  const result = currentPrediction || loadLatestResult();
  if (!result) return;

  const viewerImage = document.getElementById('viewerImage');
  const tabBtns = document.querySelectorAll('.tab-btn');
  const zoomInBtn = document.getElementById('zoomInBtn');
  const zoomOutBtn = document.getElementById('zoomOutBtn');
  const resetZoomBtn = document.getElementById('resetZoomBtn');
  const fullscreenBtn = document.getElementById('fullscreenBtn');
  const downloadReportBtn = document.getElementById('downloadReportBtn');
  const clearAnalysisBtn = document.getElementById('clearAnalysisBtn');
  const viewerStage = document.getElementById('viewerStage');

  currentTab = 'original';
  zoomScale = 1;
  panX = 0;
  panY = 0;

  function updateTransform() {
    if (viewerImage) {
      viewerImage.style.transform = `translate(${panX}px, ${panY}px) scale(${zoomScale})`;
    }
  }

  tabBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      tabBtns.forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      const tab = btn.getAttribute('data-tab');
      currentTab = tab;

      if (tab === 'original') {
        viewerImage.src = result.original_image_url || '';
      } else if (tab === 'mask') {
        viewerImage.src = result.segmentation?.mask_data_url || '';
      } else if (tab === 'overlay') {
        viewerImage.src = result.segmentation?.overlay_data_url || '';
      }
    });
  });

  if (zoomInBtn) {
    zoomInBtn.addEventListener('click', () => {
      zoomScale = Math.min(zoomScale + 0.3, 4);
      updateTransform();
    });
  }

  if (zoomOutBtn) {
    zoomOutBtn.addEventListener('click', () => {
      zoomScale = Math.max(zoomScale - 0.3, 0.6);
      updateTransform();
    });
  }

  if (resetZoomBtn) {
    resetZoomBtn.addEventListener('click', () => {
      zoomScale = 1;
      panX = 0;
      panY = 0;
      updateTransform();
    });
  }

  if (fullscreenBtn && viewerImage) {
    fullscreenBtn.addEventListener('click', () => {
      openModal(viewerImage.src);
    });
  }

  // Mouse wheel zoom
  if (viewerStage) {
    viewerStage.addEventListener('wheel', (e) => {
      e.preventDefault();
      if (e.deltaY < 0) {
        zoomScale = Math.min(zoomScale + 0.15, 4);
      } else {
        zoomScale = Math.max(zoomScale - 0.15, 0.6);
      }
      updateTransform();
    });

    // Panning
    viewerStage.addEventListener('mousedown', (e) => {
      isPanning = true;
      startPanX = e.clientX - panX;
      startPanY = e.clientY - panY;
    });

    window.addEventListener('mousemove', (e) => {
      if (!isPanning) return;
      panX = e.clientX - startPanX;
      panY = e.clientY - startPanY;
      updateTransform();
    });

    window.addEventListener('mouseup', () => {
      isPanning = false;
    });
  }

  // Download AI Analysis Summary Report
  if (downloadReportBtn) {
    downloadReportBtn.addEventListener('click', () => {
      const text = `
============================================================
TBI-AI PLATFORM — ANALYSIS SUMMARY REPORT
============================================================

Timestamp       : ${result.timestamp || new Date().toLocaleString()}
File Name       : ${result.filename || 'ct_scan.png'}
Original Size   : ${result.image_size || 'Unknown'}

MODEL PREDICTION SUMMARY:
------------------------------------------------------------
Predicted Class : ${result.prediction || result.classification?.predicted_class}
Model Confidence: ${formattedPercent(result.confidence || result.classification?.confidence)}
TBI Status      : ${result.tbi_status || result.classification?.tbi_status || 'Unknown'}
TBI Probability : ${formattedPercent(result.tbi_probability || result.classification?.tbi_probability || 0)}
Injury Type     : ${result.injury_type || result.classification?.injury_type || 'Not applicable'}

SIX-CLASS PROBABILITY BREAKDOWN:
------------------------------------------------------------
${Object.entries(result.class_probabilities || result.classification?.probabilities || {})
  .map(([k, v]) => `  ${k.padEnd(18)}: ${formattedPercent(v)}`)
  .join('\n')}

EXPERIMENTAL U-NET SEGMENTATION:
------------------------------------------------------------
Status          : ${result.segmentation?.available ? 'Available' : 'Unavailable'}
Threshold       : ${result.segmentation?.threshold || 0.25}
Positive Pixels : ${result.segmentation?.positive_pixels || 0} (${result.segmentation?.positive_percentage || 0}%)
Warning         : Experimental segmentation output is not validated for clinical use.

SYSTEM INFORMATION:
------------------------------------------------------------
Classification  : EfficientNetB0 embedding + ExtraTrees ensemble (224x224 RGB)
Segmentation    : U-Net (256x256 Grayscale)

DISCLAIMER:
------------------------------------------------------------
${result.disclaimer || 'This summary is generated by an academic research prototype and is not intended for clinical diagnosis or treatment.'}
============================================================
`;
      const blob = new Blob([text], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `TBI_AI_Analysis_Summary_${Date.now()}.txt`;
      a.click();
      URL.revokeObjectURL(url);
      showToast('AI Analysis Summary report downloaded!');
    });
  }

  // Clear Analysis
  if (clearAnalysisBtn) {
    clearAnalysisBtn.addEventListener('click', () => {
      currentPrediction = null;
      localStorage.removeItem(STORAGE_KEYS.latestResult);
      showToast('Analysis cleared.');
      window.location.hash = '#/analyze';
    });
  }
}

// Initial boot
window.addEventListener('hashchange', renderPage);
window.addEventListener('DOMContentLoaded', () => {
  checkApiHealth();
  setInterval(checkApiHealth, 15000);
  renderPage();
});
