// Main JavaScript Application for Al2O3 Photocatalytic Degradation ML System

document.addEventListener("DOMContentLoaded", () => {
    // Chart instances
    let chartParity = null;
    let chartResiduals = null;
    let chartImportance = null;
    let chartMetricsCompare = null;

    let globalVisualData = null;
    let globalMetrics = [];
    let currentParityModel = "Support Vector Regressor (SVR)";
    let selectedFigureModel = "Support Vector Regressor (SVR)";
    let currentTheme = "dark";

    // DOM Elements
    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabContents = document.querySelectorAll(".tab-content");
    const themeToggleBtn = document.getElementById("themeToggleBtn");
    
    // Sliders & Inputs
    const inp_pH = document.getElementById("inp_pH");
    const val_pH = document.getElementById("val_pH");
    const inp_Temp = document.getElementById("inp_Temp");
    const val_Temp = document.getElementById("val_Temp");
    const inp_Time = document.getElementById("inp_Time");
    const val_Time = document.getElementById("val_Time");
    const inp_Al2O3 = document.getElementById("inp_Al2O3");
    const val_Al2O3 = document.getElementById("val_Al2O3");
    const inp_Dye = document.getElementById("inp_Dye");
    const val_Dye = document.getElementById("val_Dye");
    const selectModel = document.getElementById("selectModel");
    const parityModelSelect = document.getElementById("parityModelSelect");
    const figureModelSelect = document.getElementById("figureModelSelect");
    const predictionForm = document.getElementById("predictionForm");
    const btnRetrainAll = document.getElementById("btnRetrainAll");
    const retrainStatus = document.getElementById("retrainStatus");

    // Initialize
    initTheme();
    initTabs();
    initRangeListeners();
    initPresets();
    initFigureModelSelector();
    loadDatasetOverview();
    loadVisualizations();
    loadResearchFigures(selectedFigureModel);

    // ====================================================================
    // 0. Theme Switcher (Black / White / Dark / Light)
    // ====================================================================
    function initTheme() {
        const savedTheme = localStorage.getItem("nano_theme") || "dark";
        setTheme(savedTheme);

        if (themeToggleBtn) {
            themeToggleBtn.addEventListener("click", () => {
                const newTheme = document.body.classList.contains("light-theme") ? "dark" : "light";
                setTheme(newTheme);
            });
        }
    }

    function setTheme(theme) {
        currentTheme = theme;
        if (theme === "light") {
            document.body.classList.remove("dark-theme");
            document.body.classList.add("light-theme");
        } else {
            document.body.classList.remove("light-theme");
            document.body.classList.add("dark-theme");
        }
        localStorage.setItem("nano_theme", theme);

        if (globalVisualData) {
            renderAllCharts();
        }
    }

    function getThemeColors() {
        const isLight = document.body.classList.contains("light-theme");
        return {
            textColor: isLight ? "#1e293b" : "#cbd5e1",
            tickColor: isLight ? "#475569" : "#94a3b8",
            gridColor: isLight ? "rgba(0, 0, 0, 0.07)" : "rgba(255, 255, 255, 0.06)",
            parityLineColor: isLight ? "rgba(15, 23, 42, 0.85)" : "rgba(255, 255, 255, 0.75)",
            baselineColor: isLight ? "rgba(15, 23, 42, 0.6)" : "rgba(255, 255, 255, 0.5)"
        };
    }

    // ====================================================================
    // 1. Tab Navigation
    // ====================================================================
    function initTabs() {
        tabBtns.forEach(btn => {
            btn.addEventListener("click", () => {
                const targetTab = btn.getAttribute("data-tab");
                tabBtns.forEach(b => b.classList.remove("active"));
                tabContents.forEach(c => c.classList.remove("active"));

                btn.classList.add("active");
                const targetElem = document.getElementById(targetTab);
                if (targetElem) targetElem.classList.add("active");

                if (targetTab === "tab-graphs" && globalVisualData) {
                    renderAllCharts();
                } else if (targetTab === "tab-figures") {
                    loadResearchFigures(selectedFigureModel);
                }
            });
        });
    }

    // ====================================================================
    // 2. Input Synchronization
    // ====================================================================
    function initRangeListeners() {
        inp_pH.addEventListener("input", (e) => {
            val_pH.innerText = parseFloat(e.target.value).toFixed(1);
        });

        inp_Temp.addEventListener("input", (e) => {
            val_Temp.innerText = `${e.target.value} °C`;
        });

        inp_Time.addEventListener("input", (e) => {
            val_Time.innerText = `${e.target.value} min`;
        });

        inp_Al2O3.addEventListener("input", (e) => {
            val_Al2O3.innerText = `${parseFloat(e.target.value).toFixed(1)} mg`;
        });

        inp_Dye.addEventListener("change", (e) => {
            const map = {
                "0.000025": "2.5 × 10⁻⁵ M",
                "0.000030": "3.0 × 10⁻⁵ M",
                "0.000035": "3.5 × 10⁻⁵ M",
                "0.000040": "4.0 × 10⁻⁵ M",
                "0.000045": "4.5 × 10⁻⁵ M"
            };
            val_Dye.innerText = map[e.target.value] || `${e.target.value} M`;
        });

        if (parityModelSelect) {
            parityModelSelect.addEventListener("change", (e) => {
                currentParityModel = e.target.value;
                if (globalVisualData) {
                    renderParityChart(globalVisualData, currentParityModel);
                }
            });
        }
    }

    // ====================================================================
    // 3. Preset Buttons
    // ====================================================================
    function initPresets() {
        document.querySelectorAll(".btn-preset").forEach(btn => {
            btn.addEventListener("click", () => {
                const ph = btn.getAttribute("data-ph");
                const temp = btn.getAttribute("data-temp");
                const time = btn.getAttribute("data-time");
                const alo = btn.getAttribute("data-alo");
                const dye = btn.getAttribute("data-dye");

                if (ph) { inp_pH.value = ph; val_pH.innerText = parseFloat(ph).toFixed(1); }
                if (temp) { inp_Temp.value = temp; val_Temp.innerText = `${temp} °C`; }
                if (time) { inp_Time.value = time; val_Time.innerText = `${time} min`; }
                if (alo) { inp_Al2O3.value = alo; val_Al2O3.innerText = `${parseFloat(alo).toFixed(1)} mg`; }
                if (dye) {
                    inp_Dye.value = dye;
                    const map = {
                        "0.000025": "2.5 × 10⁻⁵ M",
                        "0.000030": "3.0 × 10⁻⁵ M",
                        "0.000035": "3.5 × 10⁻⁵ M",
                        "0.000040": "4.0 × 10⁻⁵ M",
                        "0.000045": "4.5 × 10⁻⁵ M"
                    };
                    val_Dye.innerText = map[dye] || `${dye} M`;
                }

                // Trigger prediction
                runPrediction();
            });
        });
    }

    // ====================================================================
    // 3b. Publication Figures Model Selector
    // ====================================================================
    function initFigureModelSelector() {
        if (figureModelSelect) {
            figureModelSelect.addEventListener("change", (e) => {
                selectedFigureModel = e.target.value;
                syncFigureModelPills(selectedFigureModel);
                loadResearchFigures(selectedFigureModel);
            });
        }

        document.querySelectorAll(".btn-fig-model-pill").forEach(pill => {
            pill.addEventListener("click", () => {
                const m = pill.getAttribute("data-model");
                if (m) {
                    selectedFigureModel = m;
                    if (figureModelSelect) figureModelSelect.value = m;
                    syncFigureModelPills(m);
                    loadResearchFigures(m);
                }
            });
        });
    }

    function syncFigureModelPills(activeModel) {
        document.querySelectorAll(".btn-fig-model-pill").forEach(p => {
            if (p.getAttribute("data-model") === activeModel) {
                p.classList.add("active");
            } else {
                p.classList.remove("active");
            }
        });
    }

    // ====================================================================
    // 4. Run Single Prediction
    // ====================================================================
    predictionForm.addEventListener("submit", (e) => {
        e.preventDefault();
        runPrediction();
    });

    async function runPrediction() {
        const payload = {
            model: selectModel.value,
            inputs: {
                pH: parseFloat(inp_pH.value),
                Temperature_C: parseFloat(inp_Temp.value),
                Time_min: parseFloat(inp_Time.value),
                Al2O3_mg: parseFloat(inp_Al2O3.value),
                Dye_Conc_M: parseFloat(inp_Dye.value)
            }
        };

        const btn = document.getElementById("btnRunPrediction");
        btn.disabled = true;
        btn.innerText = "⏳ Computing...";

        try {
            const res = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            const data = await res.json();
            btn.disabled = false;
            btn.innerText = "⚡ Compute Degradation Prediction";

            if (data.status === "success") {
                const predVal = data.predicted_value;
                document.getElementById("predValText").innerText = predVal.toFixed(2);
                document.getElementById("badgeModelUsed").innerText = data.model_used;

                // Update Gauge Ring
                const circ = 440;
                const offset = circ - (predVal / 100) * circ;
                const progressCircle = document.getElementById("gaugeProgressCircle");
                progressCircle.style.strokeDashoffset = offset;

                // Color code gauge
                if (predVal >= 75) {
                    progressCircle.style.stroke = "#10b981"; // Emerald
                } else if (predVal >= 50) {
                    progressCircle.style.stroke = "#f59e0b"; // Amber
                } else {
                    progressCircle.style.stroke = "#ef4444"; // Red
                }

                // Progress bar
                document.getElementById("effProgressBar").style.width = `${predVal}%`;

                // Mechanism & kinetics notes
                const interp = data.interpretation;
                document.getElementById("featureDriverSummary").innerHTML = interp.feature_summary || "";
                
                const listHtml = interp.mechanism_notes.map(note => `<li>${note}</li>`).join("");
                document.getElementById("mechanismList").innerHTML = listHtml;
            } else {
                alert(`Error: ${data.message}`);
            }
        } catch (err) {
            btn.disabled = false;
            btn.innerText = "⚡ Compute Degradation Prediction";
            console.error("Prediction failed:", err);
        }
    }

    // ====================================================================
    // 5. Load Dataset Inspection
    // ====================================================================
    async function loadDatasetOverview() {
        try {
            const res = await fetch("/api/data/inspect");
            const data = await res.json();

            if (data.status === "success") {
                const summary = data.summary;
                document.getElementById("statRunsCount").innerText = summary.shape[0];
                document.getElementById("statInputParams").innerText = summary.columns.length - 1;

                renderPreviewTable(summary.head);
            }
        } catch (err) {
            console.error("Failed to load dataset inspection:", err);
        }
    }

    function renderPreviewTable(records) {
        if (!records || records.length === 0) return;
        const columns = Object.keys(records[0]);
        const colLabels = {
            "pH": "pH",
            "Temperature_C": "Temp (°C)",
            "Time_min": "Time (min)",
            "Al2O3_mg": "Al₂O₃ NP (mg)",
            "Dye_Conc_M": "Dye Conc (M)",
            "Degradation_Percent": "Dye Degradation (%)"
        };

        const headHtml = `<tr>${columns.map(k => `<th>${colLabels[k] || k}</th>`).join("")}</tr>`;
        const bodyHtml = records.map(r => {
            const cells = columns.map(k => `<td>${r[k]}</td>`).join("");
            return `<tr>${cells}</tr>`;
        }).join("");

        document.querySelector("#datasetPreviewTable thead").innerHTML = headHtml;
        document.querySelector("#datasetPreviewTable tbody").innerHTML = bodyHtml;
    }

    // ====================================================================
    // 6. Visualizations & Charts
    // ====================================================================
    async function loadVisualizations() {
        try {
            const res = await fetch("/api/visualizations");
            const data = await res.json();

            if (data.status === "success") {
                globalVisualData = data;
                globalMetrics = data.metrics;
                renderAllCharts();
                renderMetricsTable(data.metrics, data.best_model);
            }
        } catch (err) {
            console.error("Failed to load visualizations:", err);
        }
    }

    function renderAllCharts() {
        if (!globalVisualData) return;
        renderParityChart(globalVisualData, currentParityModel);
        renderResidualsChart(globalVisualData);
        renderImportanceChart(globalVisualData.feature_importances);
        renderMetricsCompareChart(globalVisualData.metrics);
    }

    // Parity Chart (Real vs Predicted)
    function renderParityChart(data, modelName) {
        const ctx = document.getElementById("chartParity").getContext("2d");
        if (chartParity) chartParity.destroy();

        const theme = getThemeColors();
        const actualTrain = data.actual_vs_pred.actual_train;
        const actualTest = data.actual_vs_pred.actual_test;
        const predTrain = data.actual_vs_pred.predictions_train[modelName] || data.actual_vs_pred.predictions_train["Support Vector Regressor (SVR)"];
        const predTest = data.actual_vs_pred.predictions_test[modelName] || data.actual_vs_pred.predictions_test["Support Vector Regressor (SVR)"];

        const trainPoints = actualTrain.map((act, i) => ({ x: act, y: predTrain[i] }));
        const testPoints = actualTest.map((act, i) => ({ x: act, y: predTest[i] }));

        // 1:1 Parity Line
        const minVal = 25;
        const maxVal = 95;
        const parityLine = [{ x: minVal, y: minVal }, { x: maxVal, y: maxVal }];

        chartParity = new Chart(ctx, {
            type: "scatter",
            data: {
                datasets: [
                    {
                        label: "Ideal Parity (y = x)",
                        data: parityLine,
                        type: "line",
                        borderColor: theme.parityLineColor,
                        borderWidth: 1.8,
                        borderDash: [5, 5],
                        pointRadius: 0,
                        fill: false
                    },
                    {
                        label: `Training Set (N=${actualTrain.length})`,
                        data: trainPoints,
                        backgroundColor: "rgba(59, 130, 246, 0.75)",
                        borderColor: "#3b82f6",
                        borderWidth: 1,
                        pointRadius: 5,
                        pointHoverRadius: 7
                    },
                    {
                        label: `Testing Set (N=${actualTest.length})`,
                        data: testPoints,
                        backgroundColor: "rgba(249, 115, 22, 0.9)",
                        borderColor: "#ea580c",
                        borderWidth: 1,
                        pointRadius: 6,
                        pointStyle: "rectRot",
                        pointHoverRadius: 8
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: theme.textColor } },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => `Actual: ${ctx.raw.x.toFixed(2)}% | Pred: ${ctx.raw.y.toFixed(2)}%`
                        }
                    }
                },
                scales: {
                    x: {
                        title: { display: true, text: "Experimental Actual Degradation (%)", color: theme.tickColor, font: { weight: "bold" } },
                        grid: { color: theme.gridColor },
                        ticks: { color: theme.tickColor },
                        min: minVal,
                        max: maxVal
                    },
                    y: {
                        title: { display: true, text: "Model Predicted Degradation (%)", color: theme.tickColor, font: { weight: "bold" } },
                        grid: { color: theme.gridColor },
                        ticks: { color: theme.tickColor },
                        min: minVal,
                        max: maxVal
                    }
                }
            }
        });
    }

    // Residuals Chart
    function renderResidualsChart(data) {
        const ctx = document.getElementById("chartResiduals").getContext("2d");
        if (chartResiduals) chartResiduals.destroy();

        const theme = getThemeColors();
        const resData = data.residuals;
        const points = resData.predicted.map((pred, i) => ({ x: pred, y: resData.residuals[i] }));
        const minX = Math.min(...resData.predicted) - 5;
        const maxX = Math.max(...resData.predicted) + 5;

        chartResiduals = new Chart(ctx, {
            type: "scatter",
            data: {
                datasets: [
                    {
                        label: "Zero Baseline (Zero Error)",
                        data: [{ x: minX, y: 0 }, { x: maxX, y: 0 }],
                        type: "line",
                        borderColor: theme.baselineColor,
                        borderWidth: 1.5,
                        borderDash: [4, 4],
                        pointRadius: 0
                    },
                    {
                        label: "Residuals (Actual - Predicted)",
                        data: points,
                        backgroundColor: "rgba(16, 185, 129, 0.85)",
                        borderColor: "#059669",
                        borderWidth: 1,
                        pointRadius: 6,
                        pointHoverRadius: 8
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: theme.textColor } },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => `Pred: ${ctx.raw.x.toFixed(2)}% | Error: ${ctx.raw.y.toFixed(2)}%`
                        }
                    }
                },
                scales: {
                    x: {
                        title: { display: true, text: "Predicted Degradation (%)", color: theme.tickColor, font: { weight: "bold" } },
                        grid: { color: theme.gridColor },
                        ticks: { color: theme.tickColor }
                    },
                    y: {
                        title: { display: true, text: "Residual Error (%)", color: theme.tickColor, font: { weight: "bold" } },
                        grid: { color: theme.gridColor },
                        ticks: { color: theme.tickColor }
                    }
                }
            }
        });
    }

    // Feature Importance Chart
    function renderImportanceChart(importances) {
        const ctx = document.getElementById("chartImportance").getContext("2d");
        if (chartImportance) chartImportance.destroy();

        const theme = getThemeColors();
        const featData = importances["Gradient Boosting"] || importances["Random Forest"] || importances["Global_Consensus"] || {};
        const labelMap = {
            "pH": "Solution pH",
            "Temperature_C": "Temperature (°C)",
            "Time_min": "Irradiation Time (min)",
            "Al2O3_mg": "Al₂O₃ Dosage (mg)",
            "Dye_Conc_M": "Dye Conc (M)"
        };

        const keys = Object.keys(featData);
        const labels = keys.map(k => labelMap[k] || k);
        const values = keys.map(k => (featData[k] * 100).toFixed(1));

        chartImportance = new Chart(ctx, {
            type: "bar",
            data: {
                labels: labels,
                datasets: [{
                    label: "Relative Importance (%)",
                    data: values,
                    backgroundColor: [
                        "rgba(59, 130, 246, 0.85)",
                        "rgba(16, 185, 129, 0.85)",
                        "rgba(139, 92, 246, 0.85)",
                        "rgba(245, 158, 11, 0.85)",
                        "rgba(239, 68, 68, 0.85)"
                    ],
                    borderColor: "rgba(255, 255, 255, 0.2)",
                    borderWidth: 1,
                    borderRadius: 6
                }]
            },
            options: {
                indexAxis: "y",
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: {
                        title: { display: true, text: "Relative Contribution (%)", color: theme.tickColor },
                        grid: { color: theme.gridColor },
                        ticks: { color: theme.tickColor }
                    },
                    y: {
                        grid: { display: false },
                        ticks: { color: theme.textColor, font: { weight: "bold" } }
                    }
                }
            }
        });
    }

    // Models Metrics Benchmark Chart
    function renderMetricsCompareChart(metrics) {
        const ctx = document.getElementById("chartMetricsCompare").getContext("2d");
        if (chartMetricsCompare) chartMetricsCompare.destroy();

        const theme = getThemeColors();
        const sorted = [...metrics].sort((a, b) => b.Test_R2 - a.Test_R2).slice(0, 6);
        const labels = sorted.map(m => m.Model);
        const testR2 = sorted.map(m => m.Test_R2);
        const cvR2 = sorted.map(m => m.CV_R2_Mean);
        const testRMSE = sorted.map(m => m.Test_RMSE);

        chartMetricsCompare = new Chart(ctx, {
            type: "bar",
            data: {
                labels: labels,
                datasets: [
                    {
                        label: "Test R² Score",
                        data: testR2,
                        backgroundColor: "rgba(16, 185, 129, 0.85)",
                        borderRadius: 4
                    },
                    {
                        label: "5-Fold CV R²",
                        data: cvR2,
                        backgroundColor: "rgba(59, 130, 246, 0.85)",
                        borderRadius: 4
                    },
                    {
                        label: "Test RMSE (%)",
                        data: testRMSE,
                        backgroundColor: "rgba(239, 68, 68, 0.85)",
                        borderRadius: 4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: theme.textColor } }
                },
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: { color: theme.tickColor, maxRotation: 25, minRotation: 25 }
                    },
                    y: {
                        grid: { color: theme.gridColor },
                        ticks: { color: theme.tickColor }
                    }
                }
            }
        });
    }

    // ====================================================================
    // 7. Render Benchmark Table
    // ====================================================================
    function renderMetricsTable(metrics, bestModel) {
        const tbody = document.querySelector("#metricsTable tbody");
        if (!tbody) return;

        const rows = metrics.map(m => {
            const isBest = m.Model === bestModel;
            const rowClass = isBest ? "class='highlight-best-row'" : "";
            const statusBadge = isBest ? "🏆 Best Model" : "Evaluated";

            return `
                <tr ${rowClass}>
                    <td><strong>${m.Model}</strong></td>
                    <td>${m.Train_R2.toFixed(4)}</td>
                    <td>${m.Test_R2.toFixed(4)}</td>
                    <td>${m.Adj_R2.toFixed(4)}</td>
                    <td>${m.Test_RMSE.toFixed(2)}%</td>
                    <td>${m.Test_MAE.toFixed(2)}%</td>
                    <td>${m.Test_MAPE.toFixed(2)}%</td>
                    <td>${m.Pearson_r.toFixed(4)}</td>
                    <td>${m.CV_R2_Mean.toFixed(4)} ± ${m.CV_R2_Std.toFixed(4)}</td>
                    <td><span class="badge-pill">${statusBadge}</span></td>
                </tr>
            `;
        }).join("");

        tbody.innerHTML = rows;
    }

    // ====================================================================
    // 8. Load Publication Figures (300 DPI) per Selected Model
    // ====================================================================
    async function loadResearchFigures(modelName = "Support Vector Regressor (SVR)") {
        const gallery = document.getElementById("figuresGalleryGrid");
        if (!gallery) return;

        // Show loading state
        gallery.innerHTML = `
            <div style="grid-column: 1 / -1; text-align: center; padding: 40px; color: var(--text-secondary);">
                <div style="font-size: 28px; margin-bottom: 10px;">⏳</div>
                <p>Generating & loading publication-grade figures for <strong>${modelName}</strong>...</p>
            </div>
        `;

        try {
            const res = await fetch(`/api/research-figures?model=${encodeURIComponent(modelName)}`);
            const data = await res.json();

            if (data.status === "success") {
                const cardsHtml = data.figures.map(fig => `
                    <div class="figure-card">
                        <div class="figure-img-box" onclick="openImageModal('${fig.url}?v=${Date.now()}', '${fig.title}', '${fig.id}')">
                            <img src="${fig.url}?v=${Date.now()}" alt="${fig.title}" loading="lazy">
                            <span class="figure-zoom-tag">🔍 Click to Enlarge</span>
                        </div>
                        <div class="figure-info">
                            <div>
                                <h4>${fig.title}</h4>
                                <div class="figure-subtitle">${fig.subtitle}</div>
                                <p class="figure-desc">${fig.description}</p>
                            </div>
                            <div class="figure-actions">
                                <a href="/api/download-figure/${fig.id}" class="btn btn-primary btn-block" download>
                                    💾 Download High-Res (300 DPI)
                                </a>
                            </div>
                        </div>
                    </div>
                `).join("");

                gallery.innerHTML = cardsHtml;
            } else {
                gallery.innerHTML = `<p style="color: var(--danger); padding: 20px;">Failed to load figures: ${data.message}</p>`;
            }
        } catch (err) {
            console.error("Failed to load publication figures:", err);
            gallery.innerHTML = `<p style="color: var(--danger); padding: 20px;">Error connecting to server for figures.</p>`;
        }
    }

    // ====================================================================
    // 9. Retrain All Pipeline
    // ====================================================================
    if (btnRetrainAll) {
        btnRetrainAll.addEventListener("click", async () => {
            btnRetrainAll.disabled = true;
            btnRetrainAll.innerText = "⏳ Retraining Pipeline & Generating Figures...";
            retrainStatus.className = "status-alert";
            retrainStatus.classList.remove("hidden");
            retrainStatus.innerText = "Training all 9 machine learning models on Al2O3 dataset and rendering 300 DPI research figures...";

            try {
                const res = await fetch("/api/train", { method: "POST" });
                const data = await res.json();
                btnRetrainAll.disabled = false;
                btnRetrainAll.innerText = "🔄 Retrain & Re-evaluate Pipeline";

                if (data.status === "success") {
                    retrainStatus.className = "status-alert success";
                    retrainStatus.innerText = `✅ Success! All models retrained. Best model: ${data.best_model} (Test R² = ${data.metrics[0].Test_R2.toFixed(4)}). Research figures updated.`;
                    renderMetricsTable(data.metrics, data.best_model);
                    loadVisualizations();
                    loadResearchFigures(selectedFigureModel);
                } else {
                    retrainStatus.className = "status-alert error";
                    retrainStatus.innerText = `❌ Error: ${data.message}`;
                }
            } catch (err) {
                btnRetrainAll.disabled = false;
                btnRetrainAll.innerText = "🔄 Retrain & Re-evaluate Pipeline";
                retrainStatus.className = "status-alert error";
                retrainStatus.innerText = "❌ Retraining request failed. Server connection error.";
            }
        });
    }

    // ====================================================================
    // 10. Modal Zoom for Figures
    // ====================================================================
    window.openImageModal = function(imgSrc, title, figId) {
        const modal = document.getElementById("imageModal");
        const modalImg = document.getElementById("modalImg");
        const modalTitle = document.getElementById("modalFigTitle");
        const modalDownload = document.getElementById("modalDownloadBtn");

        modalImg.src = imgSrc;
        modalTitle.innerText = title;
        modalDownload.href = `/api/download-figure/${figId}`;
        modal.classList.add("active");
    };

    const modalCloseBtn = document.getElementById("modalCloseBtn");
    const imageModal = document.getElementById("imageModal");

    if (modalCloseBtn) {
        modalCloseBtn.addEventListener("click", () => {
            imageModal.classList.remove("active");
        });
    }

    if (imageModal) {
        imageModal.addEventListener("click", (e) => {
            if (e.target === imageModal) {
                imageModal.classList.remove("active");
            }
        });
    }
});
