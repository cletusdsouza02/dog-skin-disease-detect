/**
 * Dog Skin Disease Detection - Main Application Script
 * Orchestrates UI interactions, theme switching, disease catalog filtering,
 * modal views, drag & drop image ingestion, and backend health monitoring.
 */

const APP = {
  activeFilter: "All",

  init() {
    this.initTheme();
    this.renderCatalog("All");
    this.initFilterButtons();
    this.initDropzone();
    this.initAnalyzeButton();
    this.checkBackendHealth();
  },

  /**
   * Theme Manager (Dark / Light) with LocalStorage persistence
   */
  initTheme() {
    const savedTheme = localStorage.getItem("dogskin_theme") || "dark";
    this.setTheme(savedTheme);

    const toggleBtn = document.getElementById("themeToggleBtn");
    if (toggleBtn) {
      toggleBtn.addEventListener("click", () => {
        const currentTheme = document.documentElement.getAttribute("data-theme") || "dark";
        const nextTheme = currentTheme === "dark" ? "light" : "dark";
        this.setTheme(nextTheme);
      });
    }
  },

  setTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("dogskin_theme", theme);

    const themeIcon = document.getElementById("themeIcon");
    if (themeIcon) {
      if (theme === "light") {
        themeIcon.classList.remove("fa-moon");
        themeIcon.classList.add("fa-sun");
      } else {
        themeIcon.classList.remove("fa-sun");
        themeIcon.classList.add("fa-moon");
      }
    }
  },

  /**
   * Filterable Disease Catalog Grid
   */
  initFilterButtons() {
    const buttons = document.querySelectorAll(".btn-filter");
    buttons.forEach(btn => {
      btn.addEventListener("click", () => {
        buttons.forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        const filter = btn.getAttribute("data-filter") || "All";
        this.activeFilter = filter;
        this.renderCatalog(filter);
      });
    });
  },

  renderCatalog(filter) {
    const grid = document.getElementById("diseaseCatalogGrid");
    if (!grid) return;

    const diseaseKeys = Object.keys(DISEASE_CATALOG);
    const filteredKeys = diseaseKeys.filter(key => {
      if (filter === "All") return true;
      const d = DISEASE_CATALOG[key];
      return d.category.toLowerCase().includes(filter.toLowerCase());
    });

    if (filteredKeys.length === 0) {
      grid.innerHTML = `
        <div class="col-12 text-center py-5">
          <p class="text-muted fs-5">No disease conditions found for the selected category.</p>
        </div>
      `;
      return;
    }

    grid.innerHTML = filteredKeys.map(key => {
      const d = DISEASE_CATALOG[key];
      const severityBadgeClass = d.severityLevel === "high" 
        ? "severity-high" 
        : (d.severityLevel === "low" ? "severity-low" : "severity-medium");

      return `
        <div class="col-md-6 col-lg-4">
          <div class="disease-card">
            <div class="disease-card-img-wrap">
              <img src="${d.image}" alt="${d.name}" class="disease-card-img" 
                onerror="this.src='assets/images/sample_healthy.jpg'">
              <span class="badge-severity ${severityBadgeClass} position-absolute top-0 end-0 m-3 shadow-sm">
                ${d.severity}
              </span>
            </div>
            <div class="disease-card-body">
              <div class="d-flex align-items-center justify-content-between mb-2">
                <span class="badge bg-primary text-white">${d.category}</span>
                <span class="text-muted extra-small">${d.subCategory || ''}</span>
              </div>
              <h4 class="fs-5 fw-bold text-white mb-2">${d.name}</h4>
              <p class="text-muted small mb-3 flex-grow-1">${d.shortDesc}</p>
              
              <div class="pt-2 border-top border-secondary border-opacity-25 mt-auto">
                <button class="btn btn-sm btn-custom-secondary w-100 justify-content-center" 
                  onclick="APP.openDiseaseModal('${key}')">
                  <i class="fa-solid fa-stethoscope me-1 text-primary"></i> View Clinical Profile
                </button>
              </div>
            </div>
          </div>
        </div>
      `;
    }).join("");
  },

  /**
   * Opens detailed clinical profile modal
   */
  openDiseaseModal(key) {
    const d = DISEASE_CATALOG[key];
    if (!d) return;

    const modalTitle = document.getElementById("diseaseModalTitle");
    const modalBody = document.getElementById("diseaseModalBody");

    modalTitle.innerHTML = `<i class="fa-solid fa-notes-medical text-primary me-2"></i> ${d.name}`;

    const details = d.topicDetails || {};

    const renderAccordionSection = (topicKey, topicData, fallbackItems) => {
      if (topicData?.keyPoints && topicData.keyPoints.length > 0) {
        return `
          <p class="mb-3 text-light small lead-sub">${topicData.summary || ''}</p>
          <div class="topic-keypoints-grid mb-2">
            ${topicData.keyPoints.map(pt => `
              <div class="topic-point-item d-flex gap-2 p-2 rounded-2 mb-2">
                <i class="fa-solid fa-circle-check text-primary mt-1 small"></i>
                <div>
                  <strong class="text-white small">${pt.label}:</strong>
                  <span class="text-light small opacity-90">${pt.detail}</span>
                </div>
              </div>
            `).join('')}
          </div>
          ${topicData.veterinaryNote ? `
            <div class="p-2 rounded-2 mt-2 bg-info bg-opacity-10 border border-info border-opacity-25 small text-light">
              <i class="fa-solid fa-user-doctor text-info me-1"></i> <strong>Veterinary Pearl:</strong> ${topicData.veterinaryNote}
            </div>
          ` : ''}
          ${topicData.caution ? `
            <div class="p-2 rounded-2 mt-2 bg-danger bg-opacity-10 border border-danger border-opacity-25 small text-light">
              <i class="fa-solid fa-triangle-exclamation text-danger me-1"></i> <strong>Critical Warning:</strong> ${topicData.caution}
            </div>
          ` : ''}
        `;
      }
      return `<ul class="list-unstyled mb-0">${(fallbackItems || []).map(item => `<li class="mb-2"><i class="fa-solid fa-check text-primary me-2"></i>${item}</li>`).join('')}</ul>`;
    };

    const symptomsHtml = renderAccordionSection("symptoms", details.symptoms, d.symptoms);
    const causesHtml = renderAccordionSection("causes", details.causes, d.causes);
    const treatmentHtml = renderAccordionSection("treatment", details.treatment, d.treatment);
    const preventionHtml = renderAccordionSection("prevention", details.prevention, d.prevention);

    modalBody.innerHTML = `
      <div class="row g-4 mb-4">
        <div class="col-md-5">
          <img src="${d.image}" alt="${d.name}" class="img-fluid rounded-3 border shadow-sm w-100" 
            style="max-height: 240px; object-fit: cover;" onerror="this.src='assets/images/sample_healthy.jpg'">
          <div class="mt-3 p-3 bg-dark-glass rounded-3 border">
            <div class="small text-muted mb-1">Severity Classification:</div>
            <span class="badge-severity severity-${d.severityLevel || 'medium'} d-inline-block">${d.severity}</span>
            <div class="small text-muted mt-2 mb-1">Clinical Domain:</div>
            <span class="badge bg-secondary">${d.category}</span>
          </div>
        </div>
        <div class="col-md-7">
          <h5 class="fw-bold text-white mb-2">Overview</h5>
          <p class="text-light small mb-3">${d.shortDesc}</p>
          <div class="alert alert-info py-2 px-3 small border-0 bg-opacity-25 bg-info text-light mb-0">
            <i class="fa-solid fa-circle-info me-1"></i> Active 6-class deep learning dataset clinical profile.
          </div>
        </div>
      </div>

      <div class="accordion" id="diseaseAccordion">
        <div class="accordion-item bg-dark border mb-2 rounded-3 overflow-hidden">
          <h2 class="accordion-header">
            <button class="accordion-button bg-dark text-white fw-bold collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#accSymptoms">
              <i class="fa-solid fa-stethoscope text-primary me-2"></i> 1. Clinical Symptoms & Signs
            </button>
          </h2>
          <div id="accSymptoms" class="accordion-collapse collapse show" data-bs-parent="#diseaseAccordion">
            <div class="accordion-body bg-dark text-light small">
              ${symptomsHtml}
            </div>
          </div>
        </div>

        <div class="accordion-item bg-dark border mb-2 rounded-3 overflow-hidden">
          <h2 class="accordion-header">
            <button class="accordion-button bg-dark text-white fw-bold collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#accCauses">
              <i class="fa-solid fa-circle-exclamation text-warning me-2"></i> 2. Pathogenesis & Underlying Causes
            </button>
          </h2>
          <div id="accCauses" class="accordion-collapse collapse" data-bs-parent="#diseaseAccordion">
            <div class="accordion-body bg-dark text-light small">
              ${causesHtml}
            </div>
          </div>
        </div>

        <div class="accordion-item bg-dark border mb-2 rounded-3 overflow-hidden">
          <h2 class="accordion-header">
            <button class="accordion-button bg-dark text-white fw-bold collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#accTreatment">
              <i class="fa-solid fa-kit-medical text-success me-2"></i> 3. Veterinary Treatment Protocols
            </button>
          </h2>
          <div id="accTreatment" class="accordion-collapse collapse" data-bs-parent="#diseaseAccordion">
            <div class="accordion-body bg-dark text-light small">
              ${treatmentHtml}
            </div>
          </div>
        </div>

        <div class="accordion-item bg-dark border rounded-3 overflow-hidden">
          <h2 class="accordion-header">
            <button class="accordion-button bg-dark text-white fw-bold collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#accPrevention">
              <i class="fa-solid fa-shield-cat text-info me-2"></i> 4. Long-Term Prevention & Management
            </button>
          </h2>
          <div id="accPrevention" class="accordion-collapse collapse" data-bs-parent="#diseaseAccordion">
            <div class="accordion-body bg-dark text-light small">
              ${preventionHtml}
            </div>
          </div>
        </div>
      </div>
    `;

    const modalEl = document.getElementById("diseaseDetailModal");
    const modalInstance = bootstrap.Modal.getOrCreateInstance(modalEl);
    modalInstance.show();
  },

  /**
   * Drag & Drop File Ingestion
   */
  initDropzone() {
    const dropzone = document.getElementById("uploadDropzone");
    const fileInput = document.getElementById("imageFileInput");

    if (!dropzone || !fileInput) return;

    dropzone.addEventListener("click", () => {
      fileInput.click();
    });

    fileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files[0]) {
        AI_ENGINE.setImageFromFile(e.target.files[0]);
      }
    });

    ["dragenter", "dragover"].forEach(eventName => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.add("dragover");
      });
    });

    ["dragleave", "drop"].forEach(eventName => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.remove("dragover");
      });
    });

    dropzone.addEventListener("drop", (e) => {
      if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]) {
        AI_ENGINE.setImageFromFile(e.dataTransfer.files[0]);
      }
    });
  },

  /**
   * Hook Analyze Image Button
   */
  initAnalyzeButton() {
    const btn = document.getElementById("analyzeBtn");
    if (btn) {
      btn.addEventListener("click", () => {
        AI_ENGINE.analyzeCurrentImage();
      });
    }
  },

  /**
   * Background Status Check against Python Flask Server
   */
  async checkBackendHealth() {
    const ticker = document.querySelector(".ai-status-ticker");
    if (!ticker) return;

    try {
      const res = await fetch("/api/health");
      if (res.ok) {
        const data = await res.json();
        ticker.innerHTML = `
          <span class="status-dot-pulse"></span>
          <span class="text-white small fw-bold"><i class="fa-solid fa-server text-success me-1"></i> Flask API Connected (${data.model || 'MobileNetV2'})</span>
          <span class="text-muted extra-small">| Status: Online | 6 Classes Active</span>
        `;
        return;
      }
    } catch (e) {
      // Backend not running on same origin
    }

    try {
      const res = await fetch("http://localhost:5000/api/health");
      if (res.ok) {
        const data = await res.json();
        ticker.innerHTML = `
          <span class="status-dot-pulse"></span>
          <span class="text-white small fw-bold"><i class="fa-solid fa-server text-success me-1"></i> Flask Backend Online (Port 5000)</span>
          <span class="text-muted extra-small">| 6 Classes Loaded | Real-Time Inference</span>
        `;
        return;
      }
    } catch (e) {
      // Offline / standalone browser mode
      ticker.innerHTML = `
        <span class="status-dot-pulse" style="background:#38BDF8; box-shadow:0 0 10px #38BDF8;"></span>
        <span class="text-white small fw-bold"><i class="fa-solid fa-microchip text-info me-1"></i> Client-Side AI Engine Ready</span>
        <span class="text-muted extra-small">| Start app.py for Flask GPU/CPU Server</span>
      `;
    }
  }
};

// Auto-boot on DOM ready
document.addEventListener("DOMContentLoaded", () => {
  APP.init();
});

// Expose globally for HTML event handlers
window.APP = APP;
