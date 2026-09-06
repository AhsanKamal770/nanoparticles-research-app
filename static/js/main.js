// Main Frontend Application Logic for Nanoparticle ML Portal

document.addEventListener("DOMContentLoaded", () => {
    let currentTarget = "SEM_Size";
    let actualVsPredChart = null;
    let featureImportanceChart = null;

    // DOM Elements
    const targetSelect = document.getElementById("targetSelect");
    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabContents = document.querySelectorAll(".tab-content");
    const btnTrainModels = document.getElementById("btnTrainModels");
    const trainingStatus = document.getElementById("trainingStatus");
    const metricsTable = document.getElementById("metricsTable");
    const bestModelCard = document.getElementById("bestModelCard");
    const bestModelName = document.getElementById("bestModelName");
    const dynamicFormFields = document.getElementById("dynamicFormFields");
    const predictionForm = document.getElementById("predictionForm");

    // Initialize
    initTabs();
    loadDataOverview();
    updateFormFields(currentTarget);

    // Event Listeners
    targetSelect.addEventListener("change", (e) => {
        currentTarget = e.target.value;
        loadDataOverview();
        updateFormFields(currentTarget);
        resetMetricsTable();

        const activeTab = document.querySelector(".tab-btn.active");
        if (activeTab && activeTab.getAttribute("data-tab") === "tab-visualizations") {
            loadVisualizations();
        }
    });

    btnTrainModels.addEventListener("click", () => {
        trainAndCompareModels();
    });

    predictionForm.addEventListener("submit", (e) => {
        e.preventDefault();
        runPrediction();
    });

    // 1. Tab Navigation Logic
    function initTabs() {
        tabBtns.forEach(btn => {
            btn.addEventListener("click", () => {
                const targetTab = btn.getAttribute("data-tab");
                tabBtns.forEach(b => b.classList.remove("active"));
                tabContents.forEach(c => c.classList.remove("active"));

                btn.classList.add("active");
                document.getElementById(targetTab).classList.add("active");

                if (targetTab === "tab-visualizations") {
                    loadVisualizations();
                }
            });
        });
    }

    // 2. Load Dataset Overview
    async function loadDataOverview() {
        try {
            const res = await fetch(`/api/data/inspect?target=${currentTarget}`);
            const data = await res.json();

            if (data.status === "success") {
                document.getElementById("stat-shape").innerText = `${data.summary.shape[0]} Experiments (${data.summary.shape[1]} Columns)`;
                document.getElementById("stat-nulls").innerText = Object.values(data.summary.null_counts).reduce((a, b) => a + b, 0);
                document.getElementById("stat-duplicates").innerText = data.summary.duplicate_count;
                document.getElementById("stat-target-name").innerText = data.target_info.label;

                // Render Preview Table
                renderPreviewTable(data.summary.head);
            }
        } catch (err) {
            console.error("Error loading data overview:", err);
        }
    }

    function renderPreviewTable(records) {
        if (!records || records.length === 0) return;
        const headRow = Object.keys(records[0]).map(k => `<th>${k}</th>`).join("");
        const bodyRows = records.map(r => {
            const cols = Object.values(r).map(v => `<td>${v}</td>`).join("");
            return `<tr>${cols}</tr>`;
        }).join("");

        document.querySelector("#previewTable thead").innerHTML = `<tr>${headRow}</tr>`;
        document.querySelector("#previewTable tbody").innerHTML = bodyRows;
    }

    // 3. Dynamic Form Field Building
    function updateFormFields(target) {
        let fields = [
            { id: "pH", label: "pH", unit: "dim", min: 4.0, max: 12.0, step: 0.1, val: 7.0 },
            { id: "Temperature", label: "Temperature", unit: "°C", min: 30, max: 100, step: 1, val: 65 },
            { id: "Time", label: "Reaction Time", unit: "h", min: 0.5, max: 10, step: 0.5, val: 2.5 },
            { id: "Zn_Concentration", label: "Zn Precursor Conc.", unit: "mM", min: 0.01, max: 1.0, step: 0.01, val: 0.10 },
            { id: "Plant_Extract", label: "Plant Extract Conc.", unit: "%", min: 1, max: 50, step: 0.5, val: 12.0 }
        ];

        if (target === "Degradation") {
            fields.push(
                { id: "SEM_Size", label: "SEM Particle Size", unit: "nm", min: 5, max: 100, step: 0.5, val: 25.0 },
                { id: "XRD_Size", label: "XRD Crystallite Size", unit: "nm", min: 5, max: 80, step: 0.5, val: 20.0 },
                { id: "UV_Bandgap", label: "UV-Vis Bandgap", unit: "eV", min: 2.8, max: 4.0, step: 0.01, val: 3.15 },
                { id: "DLS_Size", label: "DLS Size", unit: "nm", min: 10, max: 150, step: 0.5, val: 32.0 },
                { id: "PDI", label: "Polydispersity Index (PDI)", unit: "dim", min: 0.05, max: 0.8, step: 0.01, val: 0.22 },
                { id: "Zeta_Potential", label: "Zeta Potential", unit: "mV", min: -60, max: 10, step: 0.5, val: -24.0 }
            );
        }

        const html = fields.map(f => `
            <div class="form-group">
                <label for="inp_${f.id}">${f.label} (${f.unit}):</label>
                <input type="number" id="inp_${f.id}" name="${f.id}" class="form-control" 
                       min="${f.min}" max="${f.max}" step="${f.step}" value="${f.val}" required>
            </div>
        `).join("");

        dynamicFormFields.innerHTML = html;
        
        // Update prediction badge unit
        const units = { SEM_Size: "nm", XRD_Size: "nm", UV_Bandgap: "eV", Degradation: "%" };
        document.getElementById("predUnit").innerText = units[target] || "";
    }

    // 4. Train Models Function
    async function trainAndCompareModels() {
        trainingStatus.innerText = "⏳ Training Linear Regression, Decision Tree, Random Forest, & XGBoost models...";
        btnTrainModels.disabled = true;

        try {
            const res = await fetch("/api/train", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ target: currentTarget })
            });

            const data = await res.json();
            btnTrainModels.disabled = false;

            if (data.status === "success") {
                trainingStatus.innerText = `✅ Models trained successfully for target '${data.target_info.label}'. Best model: ${data.best_model}`;
                renderMetricsTable(data.metrics, data.best_model);
                bestModelCard.classList.remove("hidden");
                bestModelName.innerText = data.best_model;
            } else {
                trainingStatus.innerText = `❌ Error: ${data.message}`;
            }
        } catch (err) {
            btnTrainModels.disabled = false;
            trainingStatus.innerText = "❌ Failed to train models. Server error.";
        }
    }

    function renderMetricsTable(metrics, bestModel) {
        const rows = metrics.map(m => {
            const isBest = m.Model === bestModel;
            const cls = isBest ? "class='highlight-best'" : "";
            const badge = isBest ? "🏆 Best Model" : "Evaluated";
            return `
                <tr ${cls}>
                    <td><strong>${m.Model}</strong></td>
                    <td>${m["R2"]}</td>
                    <td>${m["RMSE"]}</td>
                    <td>${m["MAE"]}</td>
                    <td>${m["MAPE (%)"]}%</td>
                    <td><span class="badge">${badge}</span></td>
                </tr>
            `;
        }).join("");

        metricsTable.querySelector("tbody").innerHTML = rows;
    }

    function resetMetricsTable() {
        metricsTable.querySelector("tbody").innerHTML = `<tr><td colspan="6">Target changed. Click train to update metrics.</td></tr>`;
        bestModelCard.classList.add("hidden");
        trainingStatus.innerText = "Click 'Train & Compare All 4 Models' to run evaluation.";
    }

    // 5. Visualizations
    async function loadVisualizations() {
        try {
            const res = await fetch(`/api/visualizations?target=${currentTarget}`);
            const data = await res.json();

            if (data.status === "success") {
                renderActualVsPredChart(data.actual_vs_pred);
                renderFeatureImportanceChart(data.feature_importances);
            }
        } catch (err) {
            console.error("Error loading visualizations:", err);
        }
    }

    function renderActualVsPredChart(data) {
        const ctx = document.getElementById("actualVsPredChart").getContext("2d");
        if (actualVsPredChart) actualVsPredChart.destroy();

        const datasets = Object.keys(data.predictions).map((modelName, idx) => {
            const colors = ['#2b5c8f', '#e67e22', '#2e7d32', '#8e44ad'];
            const points = data.actual.map((act, i) => ({ x: act, y: data.predictions[modelName][i] }));
            return {
                label: modelName,
                data: points,
                backgroundColor: colors[idx % colors.length]
            };
        });

        actualVsPredChart = new Chart(ctx, {
            type: 'scatter',
            data: { datasets: datasets },
            options: {
                responsive: true,
                plugins: {
                    title: { display: true, text: 'Actual vs Predicted Property Values' }
                },
                scales: {
                    x: { title: { display: true, text: 'Actual Measured Value' } },
                    y: { title: { display: true, text: 'Model Predicted Value' } }
                }
            }
        });
    }

    function renderFeatureImportanceChart(importances) {
        const ctx = document.getElementById("featureImportanceChart").getContext("2d");
        if (featureImportanceChart) featureImportanceChart.destroy();

        const modelName = "Random Forest";
        const featData = importances[modelName] || {};
        const labels = Object.keys(featData);
        const values = Object.values(featData);

        featureImportanceChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: `Feature Relative Weight (${modelName})`,
                    data: values,
                    backgroundColor: '#2b5c8f'
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: { beginAtZero: true, title: { display: true, text: 'Importance Weight' } }
                }
            }
        });
    }

    // 6. Run Single Prediction
    async function runPrediction() {
        const formObj = {};
        const formData = new FormData(predictionForm);
        let hasError = false;
        let errorMsg = "";

        formData.forEach((val, key) => {
            if (val === "" || val === null || isNaN(parseFloat(val))) {
                hasError = true;
                errorMsg = `Please enter a valid numeric value for parameter '${key}'.`;
            } else {
                formObj[key] = parseFloat(val);
            }
        });

        if (hasError) {
            document.getElementById("predVal").innerText = "--";
            document.getElementById("insightSummary").innerHTML = `<span style="color: #c0392b; font-weight: bold;">⚠️ Input Error: ${errorMsg}</span>`;
            document.getElementById("insightMechanisms").innerHTML = "";
            return;
        }

        const selectedModel = document.getElementById("selectPredictModel").value;

        try {
            const res = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    target: currentTarget,
                    model: selectedModel,
                    inputs: formObj
                })
            });

            const data = await res.json();
            if (res.ok && data.status === "success") {
                document.getElementById("predVal").innerText = data.predicted_value;
                document.getElementById("predUnit").innerText = data.unit;

                // Scientific insights
                document.getElementById("insightSummary").innerHTML = data.interpretation.summary;
                const mechHtml = data.interpretation.mechanism_notes.map(m => `<li>${m}</li>`).join("");
                document.getElementById("insightMechanisms").innerHTML = mechHtml;
            } else {
                document.getElementById("predVal").innerText = "Error";
                document.getElementById("insightSummary").innerHTML = `<span style="color: #c0392b; font-weight: bold;">❌ Prediction Error: ${data.message || "Failed to calculate prediction."}</span>`;
                document.getElementById("insightMechanisms").innerHTML = "";
            }
        } catch (err) {
            console.error("Prediction error:", err);
            document.getElementById("predVal").innerText = "Error";
            document.getElementById("insightSummary").innerHTML = `<span style="color: #c0392b; font-weight: bold;">❌ Server connection error. Please try again.</span>`;
            document.getElementById("insightMechanisms").innerHTML = "";
        }
    }
});
