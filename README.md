# Al₂O₃ Nanoparticle Photocatalytic Degradation ML Research System

A focused machine learning research platform for predicting and optimizing the **photocatalytic degradation efficiency (%)** of dye pollutants using **Aluminum Oxide ($\text{Al}_2\text{O}_3$ / AlO)** nanoparticles based on real experimental laboratory data.

---

## 🔬 Experimental System & Research Scope

The platform focuses strictly on **photocatalytic degradation kinetics** over $\text{Al}_2\text{O}_3$ nanoparticles trained on 102 real experimental laboratory runs (`AlO particles.xlsx`).

### 1. Controlled Input Parameters:
1. **Solution pH**: $5.0 - 9.0$
2. **Reaction Temperature ($^\circ\text{C}$)**: $20 - 60^\circ\text{C}$
3. **Irradiation / Reaction Time ($\text{min}$)**: $20 - 120\text{ min}$
4. **$\text{Al}_2\text{O}_3$ Nanoparticle Catalyst Dosage ($\text{mg}$)**: $2.0 - 10.0\text{ mg}$
5. **Initial Dye Concentration ($\text{M}$)**: $2.5 \times 10^{-5} - 4.5 \times 10^{-5}\text{ M}$

### 2. Predicted Research Target:
- **Photocatalytic Dye Degradation Efficiency ($\%$)**: Range $30.69\% - 92.51\%$

---

## 🏆 Machine Learning Benchmark Results

Evaluated on an 80/20 train/unseen experimental test split with shuffled 5-fold cross-validation:

| Model Architecture | Train $R^2$ | Test $R^2$ | Adjusted $R^2$ | Test RMSE (%) | Test MAE (%) | Test MAPE (%) | Pearson $r$ | 5-Fold CV $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🏆 **Support Vector Regressor (SVR)** | **0.9970** | **0.9616** | **0.9488** | **2.75%** | **1.92%** | **4.31%** | **0.9848** | **0.9768 ± 0.008** |
| 🧠 **Artificial Neural Network (MLP)** | 0.9608 | 0.9047 | 0.8730 | 4.34% | 2.92% | 6.17% | 0.9567 | 0.8650 ± 0.041 |
| ⚡ **Gradient Boosting Regressor** | 0.9909 | 0.8805 | 0.8407 | 4.86% | 3.23% | 7.21% | 0.9564 | 0.8965 ± 0.034 |
| 📐 **Polynomial Regression (Deg 2)** | 0.9181 | 0.8747 | 0.8330 | 4.97% | 3.81% | 7.96% | 0.9474 | 0.8844 ± 0.030 |
| 🚀 **XGBoost Regressor** | 0.9865 | 0.8416 | 0.7888 | 5.59% | 4.06% | 8.78% | 0.9278 | 0.8853 ± 0.047 |
| 🌲 **Random Forest Regressor** | 0.9401 | 0.7151 | 0.6201 | 7.50% | 5.78% | 12.12% | 0.8819 | 0.7377 ± 0.073 |
| 📏 **Linear Regression (Baseline)** | 0.7808 | 0.6948 | 0.5931 | 7.76% | 6.01% | 12.56% | 0.8551 | 0.6973 ± 0.077 |

---

## 📊 Publication-Grade Research Figures (300 DPI)

The framework automatically generates high-resolution publication-ready vector (SVG) and raster (300 DPI PNG) figures formatted for top environmental catalysis journals (Elsevier / ACS / Springer):

1. **Figure 1: Parity Plot (Real vs. Predicted)**
   - Train vs. Test scatter points, $y = x$ 1:1 parity line, $\pm 5\%$ and $\pm 10\%$ error bands, linear regression fit, and statistical parameter box.
2. **Figure 2: Residuals Diagnostics & Error Distribution**
   - Homoscedasticity scatter analysis ($y_{\text{act}} - y_{\text{pred}}$ vs $\hat{y}$) and Gaussian normal error probability density distribution.
3. **Figure 3: Comparative ML Performance Benchmark**
   - Side-by-side grouped evaluation of Train $R^2$, Test $R^2$, 5-Fold CV $R^2$, RMSE, and MAE.
4. **Figure 4: Parametric Sensitivity & Feature Importance**
   - Relative influence breakdown of Reaction Time ($49.4\%$), Dye Concentration ($17.0\%$), Catalyst Dosage ($14.9\%$), Solution pH ($13.7\%$), and Temperature ($5.0\%$).
5. **Figure 5: Experimental Degradation Kinetics Curves**
   - 4-panel comparison overlays: Real experimental data points vs. ML continuous kinetic curves across pH, Catalyst Dosage, Dye Concentration, and Temperature sweeps.
6. **Figure 6: 3D Response Surface & 2D Iso-Response Contours (RSM)**
   - 3D interactive response surfaces highlighting optimal operating conditions.

---

## 🚀 How to Run the Web Application

```bash
# 1. Start the Flask server
python app.py
```

Navigate to:
👉 **`http://127.0.0.1:5000`**

### Features Available in the Web Portal:
- **Interactive Predictor**: Adjust sliders for the 5 experimental parameters and compute predicted degradation with real-time radical kinetics mechanism analysis.
- **Research Graphs**: Interactive Chart.js parity plots, residual scatter, sensitivity bar charts, and model comparison.
- **Publication Figures Gallery**: Preview and download all six 300 DPI research figures.
- **Experimental Data Explorer**: Interactive table previewing the 102 laboratory runs with one-click full CSV export with model predictions and residuals.
