# Green-Synthesized ZnO Nanoparticle ML Project

A comprehensive machine learning research framework and web platform for predicting physicochemical and functional properties of green-synthesized Zinc Oxide (ZnO) nanoparticles from experimental synthesis parameters.

Designed according to requirements in [`requirements.md`](file:///d:/Projects/Research/Nanoparticles/requirements.md) and [`modify.md`](file:///d:/Projects/Research/Nanoparticles/modify.md).

---

## 🌟 Key Features

1. **Multi-Target Research Framework**:
   - **RQ1**: Synthesis Parameters → `SEM Particle Size (nm)`
   - **RQ2**: Synthesis Parameters → `XRD Crystallite Size (nm)`
   - **RQ3**: Synthesis Parameters → `UV-Vis Band Gap (eV)`
   - **RQ4**: Synthesis + Characterization Properties → `Photocatalytic Degradation (%)`
2. **Machine Learning Algorithms**:
   - Linear Regression (Baseline)
   - Decision Tree Regressor
   - Random Forest Regressor
   - XGBoost Regressor
3. **Data Leakage & Preprocessing Rules**:
   - Strict variable scoping per research target.
   - 80/20 train/test splitting.
   - Comprehensive model evaluation using R², RMSE, MAE, and MAPE.
4. **Simple HTML/CSS/JS Web Interface**:
   - Clean, lightweight, non-flashy design.
   - Interactive Data Overview & Preview Table.
   - One-click Model Training & Metrics Comparison Table (highlighting the top-performing model).
   - Scatter plots & Feature Importance charts via Chart.js.
   - Interactive Property Predictor with physical chemistry mechanism insights.

---

## 📁 Directory Structure

```text
Nanoparticles/
├── data/
│   ├── raw/
│   │   ├── zn_particle.xlsx         # Raw experimental dataset (150 runs)
│   │   └── zn_particle_sample.xlsx  # 3-experiment practice dataset
│   └── processed/                   # Processed train/test data splits
├── notebooks/
│   ├── 01_data_inspection.ipynb     # Dataset inspection & summary stats
│   ├── 02_preprocessing.ipynb       # Feature selection & splitting
│   ├── 03_linear_regression.ipynb   # Linear regression baseline model
│   ├── 04_decision_tree.ipynb       # Decision tree regressor
│   ├── 05_random_forest.ipynb       # Random forest regressor
│   └── 06_xgboost.ipynb             # XGBoost regressor
├── src/
│   ├── __init__.py
│   ├── preprocessing.py             # Data loading & leakage-free splitting
│   ├── models.py                    # ML model training & joblib persistence
│   └── evaluation.py                # R2, RMSE, MAE, MAPE & chemical explanations
├── results/
│   ├── figures/
│   ├── tables/
│   └── models/                      # Saved trained models
├── templates/
│   └── index.html                   # Simple HTML5 frontend
├── static/
│   ├── css/style.css                # Clean CSS styling
│   └── js/main.js                   # Tab navigation & REST API integration
├── generate_data.py                 # Synthetic dataset generator script
├── app.py                           # Flask backend web server
├── requirements.txt                 # Python dependency list
├── requirements.md                  # Project specification document
└── modify.md                        # Multi-target research extension specs
```

---

## 🚀 Quick Start Guide

### 1. Install Dependencies

Ensure Python 3.11+ is installed, then run:

```bash
python -m pip install -r requirements.txt
```

### 2. Generate Raw Datasets

Generate raw Excel (`zn_particle.xlsx`) and practice sample files:

```bash
python generate_data.py
```

### 3. Launch Web Application

Run the Flask web server:

```bash
python app.py
```

Open your browser and navigate to:
👉 **`http://127.0.0.1:5000`**

---

## 🔬 Model Evaluation Metrics

Models are evaluated using four scientific metrics:
- **R² Score**: Coefficient of determination (closer to 1.0 is better).
- **RMSE**: Root Mean Squared Error (lower is better).
- **MAE**: Mean Absolute Error (lower is better).
- **MAPE (%)**: Mean Absolute Percentage Error (lower is better).

---

## 🧪 Scientific & Chemical Interpretation

The portal incorporates physical chemistry kinetics principles:
- **Nucleation vs Growth**: Higher plant extract concentrations and alkaline pH supply capping phytochemicals and OH⁻ ions, accelerating nucleation and yielding smaller nanoparticle sizes.
- **Quantum Confinement**: Reductions in particle size below exciton Bohr radius lead to optical band gap widening (blue shift).
- **Photocatalytic Efficiency**: Higher specific surface area and negative Zeta Potential dispersion stability enhance photocatalytic degradation rates.
