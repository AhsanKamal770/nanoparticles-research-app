import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import norm

# Research styling configuration
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.autolayout': True,
    'lines.linewidth': 2.0,
    'grid.alpha': 0.35,
    'grid.linestyle': '--'
})

def get_model_slug(model_name):
    """Generates clean filesystem slug from model name."""
    s = model_name.lower()
    s = re.sub(r'[\(\)\/\s]+', '_', s).strip('_')
    return s

def generate_plots_for_model(df, models_obj, X_train, y_train, X_test, y_test, metrics_df, model_name="Support Vector Regressor (SVR)", output_dir="results/figures", web_dir="static/generated_plots"):
    """
    Generates high-resolution (300 DPI) publication-quality research figures specifically for a chosen model.
    Saves to both results/figures/ (for research paper submission) and static/generated_plots/ (for web UI).
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(web_dir, exist_ok=True)
    
    slug = get_model_slug(model_name)
    generated_files = {}

    # 1. PARITY PLOT (Real vs Predicted)
    fig_parity = plot_parity_diagram(models_obj, X_train, y_train, X_test, y_test, metrics_df, model_name)
    save_fig(fig_parity, f"fig1_actual_vs_predicted_parity_{slug}", output_dir, web_dir, generated_files)

    # 2. RESIDUALS DIAGNOSTIC PLOT
    fig_residuals = plot_residual_diagnostics(models_obj, X_test, y_test, model_name)
    save_fig(fig_residuals, f"fig2_residual_analysis_{slug}", output_dir, web_dir, generated_files)

    # 3. COMPARATIVE MODEL PERFORMANCE (Universal)
    fig_models = plot_model_comparison(metrics_df)
    save_fig(fig_models, "fig3_model_metrics_comparison", output_dir, web_dir, generated_files)

    # 4. FEATURE IMPORTANCE & PARAMETRIC SENSITIVITY
    fig_feat = plot_feature_importance(models_obj, X_train, y_train, model_name)
    save_fig(fig_feat, f"fig4_feature_importance_sensitivity_{slug}", output_dir, web_dir, generated_files)

    # 5. EXPERIMENTAL KINETICS & PARAMETER VARIATION
    fig_kinetics = plot_experimental_kinetics(df, models_obj, model_name)
    save_fig(fig_kinetics, f"fig5_experimental_kinetics_sweeps_{slug}", output_dir, web_dir, generated_files)

    # 6. 3D RESPONSE SURFACES & 2D CONTOURS (RSM)
    fig_surface = plot_response_surfaces(models_obj, model_name)
    save_fig(fig_surface, f"fig6_response_surface_3d_{slug}", output_dir, web_dir, generated_files)

    return generated_files

def generate_all_research_plots(df, models_obj, X_train, y_train, X_test, y_test, metrics_df, best_model_name="Support Vector Regressor (SVR)", output_dir="results/figures", web_dir="static/generated_plots"):
    """
    Generates plots for all available models so they are instantly accessible.
    """
    all_generated = {}
    for m_name in models_obj.models.keys():
        plots = generate_plots_for_model(df, models_obj, X_train, y_train, X_test, y_test, metrics_df, m_name, output_dir, web_dir)
        all_generated[m_name] = plots
    return all_generated

def save_fig(fig, base_name, output_dir, web_dir, file_dict):
    """Saves figure in 300 DPI PNG and SVG formats."""
    png_path_res = os.path.join(output_dir, f"{base_name}.png")
    svg_path_res = os.path.join(output_dir, f"{base_name}.svg")
    png_path_web = os.path.join(web_dir, f"{base_name}.png")
    
    fig.savefig(png_path_res, dpi=300, bbox_inches='tight')
    fig.savefig(svg_path_res, format='svg', bbox_inches='tight')
    fig.savefig(png_path_web, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    file_dict[base_name] = {
        "png_res": png_path_res,
        "svg_res": svg_path_res,
        "web_url": f"/static/generated_plots/{base_name}.png"
    }

def plot_parity_diagram(models_obj, X_train, y_train, X_test, y_test, metrics_df, model_name):
    """
    High-end Parity Plot: Actual vs Predicted with 1:1 line, ±5%, ±10% error bands,
    and statistical summary box.
    """
    model = models_obj.models.get(model_name)
    if not model:
        model = list(models_obj.models.values())[0]

    y_train_pred = np.clip(model.predict(X_train), 0, 100)
    y_test_pred = np.clip(model.predict(X_test), 0, 100)
    
    # Calculate fit line on test
    slope, intercept = np.polyfit(y_test, y_test_pred, 1)

    # Metrics
    m_rows = metrics_df[metrics_df["Model"] == model_name]
    m_row = m_rows.iloc[0] if len(m_rows) > 0 else metrics_df.iloc[0]
    
    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    
    # Ideal line
    min_val = min(y_train.min(), y_test.min()) - 5
    max_val = max(y_train.max(), y_test.max()) + 5
    axis_line = np.linspace(min_val, max_val, 100)
    
    ax.plot(axis_line, axis_line, 'k-', lw=2.0, label='Ideal Parity (y = x)')
    ax.plot(axis_line, axis_line * 1.05, 'r--', lw=1.2, alpha=0.7, label='±5% Error Band')
    ax.plot(axis_line, axis_line * 0.95, 'r--', lw=1.2, alpha=0.7)
    ax.fill_between(axis_line, axis_line * 0.95, axis_line * 1.05, color='red', alpha=0.08)
    
    ax.plot(axis_line, axis_line * 1.10, 'g:', lw=1.0, alpha=0.6, label='±10% Error Band')
    ax.plot(axis_line, axis_line * 0.90, 'g:', lw=1.0, alpha=0.6)
    
    # Scatter points
    ax.scatter(y_train, y_train_pred, color='#1f77b4', edgecolors='k', s=65, alpha=0.75, label=f'Training Data (N={len(y_train)})')
    ax.scatter(y_test, y_test_pred, color='#e65100', marker='s', edgecolors='k', s=85, alpha=0.9, label=f'Testing Data (N={len(y_test)})')
    
    # Linear trendline on test
    trend_x = np.linspace(y_test.min(), y_test.max(), 50)
    ax.plot(trend_x, slope * trend_x + intercept, color='#b71c1c', lw=2.0, linestyle='-.', label=f'Linear Fit: y = {slope:.2f}x + {intercept:.2f}')
    
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect('equal')
    ax.set_xlabel('Experimental / Actual Degradation (%)', fontweight='bold')
    ax.set_ylabel('Model Predicted Degradation (%)', fontweight='bold')
    ax.set_title(f'Actual vs Predicted Parity Plot: {model_name}\n(Photocatalytic Degradation by Al₂O₃ Nanoparticles)', pad=12)
    ax.grid(True)
    
    # Stats Box
    stats_text = (
        f"Model: {model_name}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Train R²: {m_row['Train_R2']:.4f}\n"
        f"Test R²:  {m_row['Test_R2']:.4f}\n"
        f"Test RMSE: {m_row['Test_RMSE']:.2f}%\n"
        f"Test MAE:  {m_row['Test_MAE']:.2f}%\n"
        f"5-Fold CV R²: {m_row['CV_R2_Mean']:.4f} ± {m_row['CV_R2_Std']:.4f}"
    )
    ax.text(0.04, 0.62, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.6', facecolor='#f8f9fa', edgecolor='#6c757d', alpha=0.92))
            
    ax.legend(loc='lower right', framealpha=0.92)
    return fig

def plot_residual_diagnostics(models_obj, X_test, y_test, model_name):
    """
    Residuals vs Predicted Plot & Residual Distribution Histogram with Gaussian Fit.
    """
    model = models_obj.models.get(model_name) or list(models_obj.models.values())[0]
    y_pred = np.clip(model.predict(X_test), 0, 100)
    residuals = y_test.values - y_pred

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    
    # Subplot 1: Residuals vs Predicted
    ax1.scatter(y_pred, residuals, color='#00897b', edgecolors='k', s=70, alpha=0.85)
    ax1.axhline(0, color='black', linestyle='--', lw=1.8)
    std_res = np.std(residuals)
    if std_res > 0:
        ax1.axhline(2*std_res, color='red', linestyle=':', lw=1.2, label=f'+2σ ({2*std_res:.2f})')
        ax1.axhline(-2*std_res, color='red', linestyle=':', lw=1.2, label=f'-2σ ({-2*std_res:.2f})')
    ax1.set_xlabel('Predicted Degradation (%)', fontweight='bold')
    ax1.set_ylabel('Residual (Actual - Predicted) (%)', fontweight='bold')
    ax1.set_title('(a) Residuals vs. Model Predicted Values')
    ax1.grid(True)
    ax1.legend(loc='upper right')

    # Subplot 2: Histogram of Residuals + Normal Distribution
    n_bins = 10
    n, bins, patches = ax2.hist(residuals, bins=n_bins, density=True, color='#0288d1', edgecolor='black', alpha=0.7, label='Residuals Frequency')
    mu, std = norm.fit(residuals)
    xmin, xmax = ax2.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, max(std, 1e-4))
    ax2.plot(x, p, 'r-', lw=2.2, label=f'Normal Fit (μ={mu:.2f}, σ={std:.2f})')
    ax2.set_xlabel('Residual Error (%)', fontweight='bold')
    ax2.set_ylabel('Probability Density', fontweight='bold')
    ax2.set_title('(b) Error Distribution & Homoscedasticity')
    ax2.grid(True)
    ax2.legend(loc='upper right')

    fig.suptitle(f'Model Residuals Diagnostics: {model_name}', fontsize=14, fontweight='bold')
    return fig

def plot_model_comparison(metrics_df):
    """
    Comparison bar chart of R2 (Train, Test, CV) and RMSE across all evaluated models.
    """
    df_sorted = metrics_df.sort_values(by="Test_R2", ascending=True).copy()
    models = df_sorted["Model"].tolist()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.0))
    
    y = np.arange(len(models))
    height = 0.25

    # Chart 1: R2 Comparison
    ax1.barh(y + height, df_sorted["Train_R2"], height, label="Train R²", color="#1976d2", alpha=0.85)
    ax1.barh(y, df_sorted["Test_R2"], height, label="Test R²", color="#388e3c", alpha=0.9)
    ax1.barh(y - height, df_sorted["CV_R2_Mean"], height, label="5-Fold CV R²", color="#f57c00", alpha=0.85)
    
    ax1.set_yticks(y)
    ax1.set_yticklabels(models, fontweight='bold')
    ax1.set_xlabel("R² Score (Coefficient of Determination)", fontweight='bold')
    ax1.set_xlim(0.0, 1.05)
    ax1.set_title("(a) Accuracy Comparison (R² Scores)")
    ax1.grid(True, axis='x')
    ax1.legend(loc="lower right")

    # Chart 2: Test RMSE & MAE
    ax2.barh(y + height/2, df_sorted["Test_RMSE"], height, label="Test RMSE (%)", color="#d32f2f", alpha=0.85)
    ax2.barh(y - height/2, df_sorted["Test_MAE"], height, label="Test MAE (%)", color="#7b1fa2", alpha=0.85)
    
    ax2.set_yticks(y)
    ax2.set_yticklabels(["" for _ in models])  # Blank to share label
    ax2.set_xlabel("Error Metric (%)", fontweight='bold')
    ax2.set_title("(b) Prediction Error (RMSE & MAE)")
    ax2.grid(True, axis='x')
    ax2.legend(loc="upper right")

    fig.suptitle("Machine Learning Models Evaluation & Benchmark on Al₂O₃ Degradation Dataset", fontsize=14, fontweight='bold')
    return fig

def plot_feature_importance(models_obj, X_train, y_train, model_name=None):
    """
    Parametric sensitivity analysis and feature importance bar plot specifically for model_name.
    """
    importances = models_obj.get_feature_importances(X_train, y_train)
    
    feat_data = None
    if model_name and model_name in importances:
        feat_data = importances[model_name]
    if not feat_data:
        feat_data = importances.get("Gradient Boosting") or importances.get("Random Forest") or importances.get("Global_Consensus")
    
    name_map = {
        "pH": "Solution pH",
        "Temperature_C": "Reaction Temperature (°C)",
        "Time_min": "Irradiation Time (min)",
        "Al2O3_mg": "Al₂O₃ Catalyst Dosage (mg)",
        "Dye_Conc_M": "Initial Dye Concentration (M)"
    }
    
    labels = [name_map.get(k, k) for k in feat_data.keys()]
    values = [v * 100 for v in feat_data.values()]
    
    # Sort
    sorted_idx = np.argsort(values)
    sorted_labels = [labels[i] for i in sorted_idx]
    sorted_vals = [values[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    colors = plt.cm.viridis(np.linspace(0.3, 0.85, len(sorted_vals)))
    
    bars = ax.barh(sorted_labels, sorted_vals, color=colors, edgecolor='k', alpha=0.9, height=0.55)
    
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 1.0, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va='center', fontweight='bold', fontsize=10)

    title_model = f" ({model_name})" if model_name else ""
    ax.set_xlabel("Relative Importance / Influence (%)", fontweight='bold')
    ax.set_title(f"Feature Sensitivity Analysis{title_model}\n(Photocatalytic Degradation Factor Contribution)", pad=12)
    ax.set_xlim(0, max(sorted_vals) * 1.25)
    ax.grid(True, axis='x')
    return fig

def plot_experimental_kinetics(df, models_obj, model_name):
    """
    4-panel experimental degradation kinetics curves with real data points vs specific model fit:
    (a) pH effect over time
    (b) Catalyst dosage effect over time
    (c) Dye concentration effect over time
    (d) Temperature effect over time
    """
    model = models_obj.models.get(model_name) or list(models_obj.models.values())[0]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    time_continuous = np.linspace(20, 120, 50)
    
    # Panel (a): pH sweep (Temp=30, Al2O3=6, Dye=2.5e-5)
    ax_a = axes[0, 0]
    ph_values = [5, 6, 7, 8, 9]
    ph_colors = ['#8e24aa', '#1e88e5', '#43a047', '#fb8c00', '#e53935']
    for ph_val, col in zip(ph_values, ph_colors):
        sub_df = df[(df['pH'] == ph_val) & (df['Temperature_C'] == 30) & (df['Al2O3_mg'] == 6) & (np.isclose(df['Dye_Conc_M'], 2.5e-5))]
        if len(sub_df) > 0:
            ax_a.scatter(sub_df['Time_min'], sub_df['Degradation_Percent'], color=col, s=60, edgecolors='k', label=f'pH {ph_val} (Exp)', zorder=5)
            pred_inputs = pd.DataFrame({
                'pH': [ph_val]*50,
                'Temperature_C': [30]*50,
                'Time_min': time_continuous,
                'Al2O3_mg': [6]*50,
                'Dye_Conc_M': [2.5e-5]*50
            })
            preds = np.clip(model.predict(pred_inputs), 0, 100)
            ax_a.plot(time_continuous, preds, color=col, lw=1.8, linestyle='-', alpha=0.85)
    ax_a.set_title('(a) Effect of Solution pH vs Time (T=30°C, Al₂O₃=6mg, C₀=2.5×10⁻⁵M)', fontsize=11, fontweight='bold')
    ax_a.set_xlabel('Irradiation Time (min)', fontweight='bold')
    ax_a.set_ylabel('Dye Degradation (%)', fontweight='bold')
    ax_a.set_ylim(20, 100)
    ax_a.grid(True)
    ax_a.legend(loc='lower right', fontsize=8.5)

    # Panel (b): Catalyst Dosage sweep (pH=7, Temp=30, Dye=2.5e-5)
    ax_b = axes[0, 1]
    catalyst_values = [2, 4, 6, 8, 10]
    cat_colors = ['#3949ab', '#00acc1', '#43a047', '#fdd835', '#d81b60']
    for cat_val, col in zip(catalyst_values, cat_colors):
        sub_df = df[(df['pH'] == 7) & (df['Temperature_C'] == 30) & (df['Al2O3_mg'] == cat_val) & (np.isclose(df['Dye_Conc_M'], 2.5e-5))]
        if len(sub_df) > 0:
            ax_b.scatter(sub_df['Time_min'], sub_df['Degradation_Percent'], color=col, s=60, edgecolors='k', label=f'{cat_val} mg Al₂O₃ (Exp)', zorder=5)
            pred_inputs = pd.DataFrame({
                'pH': [7]*50,
                'Temperature_C': [30]*50,
                'Time_min': time_continuous,
                'Al2O3_mg': [cat_val]*50,
                'Dye_Conc_M': [2.5e-5]*50
            })
            preds = np.clip(model.predict(pred_inputs), 0, 100)
            ax_b.plot(time_continuous, preds, color=col, lw=1.8, linestyle='-', alpha=0.85)
    ax_b.set_title('(b) Effect of Al₂O₃ Catalyst Dosage vs Time (pH=7, T=30°C)', fontsize=11, fontweight='bold')
    ax_b.set_xlabel('Irradiation Time (min)', fontweight='bold')
    ax_b.set_ylabel('Dye Degradation (%)', fontweight='bold')
    ax_b.set_ylim(20, 100)
    ax_b.grid(True)
    ax_b.legend(loc='lower right', fontsize=8.5)

    # Panel (c): Initial Dye Concentration sweep (pH=7, Temp=30, Al2O3=6)
    ax_c = axes[1, 0]
    dye_values = [2.5e-5, 3.0e-5, 3.5e-5, 4.0e-5, 4.5e-5]
    dye_labels = ['2.5×10⁻⁵M', '3.0×10⁻⁵M', '3.5×10⁻⁵M', '4.0×10⁻⁵M', '4.5×10⁻⁵M']
    dye_colors = ['#1e88e5', '#00897b', '#7cb342', '#f4511e', '#6d4c41']
    for dye_val, lbl, col in zip(dye_values, dye_labels, dye_colors):
        sub_df = df[(df['pH'] == 7) & (df['Temperature_C'] == 30) & (df['Al2O3_mg'] == 6) & (np.isclose(df['Dye_Conc_M'], dye_val))]
        if len(sub_df) > 0:
            ax_c.scatter(sub_df['Time_min'], sub_df['Degradation_Percent'], color=col, s=60, edgecolors='k', label=f'{lbl} (Exp)', zorder=5)
            pred_inputs = pd.DataFrame({
                'pH': [7]*50,
                'Temperature_C': [30]*50,
                'Time_min': time_continuous,
                'Al2O3_mg': [6]*50,
                'Dye_Conc_M': [dye_val]*50
            })
            preds = np.clip(model.predict(pred_inputs), 0, 100)
            ax_c.plot(time_continuous, preds, color=col, lw=1.8, linestyle='-', alpha=0.85)
    ax_c.set_title('(c) Effect of Initial Dye Concentration vs Time (pH=7, Al₂O₃=6mg)', fontsize=11, fontweight='bold')
    ax_c.set_xlabel('Irradiation Time (min)', fontweight='bold')
    ax_c.set_ylabel('Dye Degradation (%)', fontweight='bold')
    ax_c.set_ylim(20, 100)
    ax_c.grid(True)
    ax_c.legend(loc='lower right', fontsize=8.5)

    # Panel (d): Temperature sweep (pH=7, Al2O3=6, Dye=2.5e-5)
    ax_d = axes[1, 1]
    temp_values = [20, 30, 40, 50, 60]
    temp_colors = ['#0288d1', '#43a047', '#fbc02d', '#f57c00', '#d32f2f']
    for t_val, col in zip(temp_values, temp_colors):
        sub_df = df[(df['pH'] == 7) & (df['Temperature_C'] == t_val) & (df['Al2O3_mg'] == 6) & (np.isclose(df['Dye_Conc_M'], 2.5e-5))]
        if len(sub_df) > 0:
            ax_d.scatter(sub_df['Time_min'], sub_df['Degradation_Percent'], color=col, s=60, edgecolors='k', label=f'{t_val}°C (Exp)', zorder=5)
            pred_inputs = pd.DataFrame({
                'pH': [7]*50,
                'Temperature_C': [t_val]*50,
                'Time_min': time_continuous,
                'Al2O3_mg': [6]*50,
                'Dye_Conc_M': [2.5e-5]*50
            })
            preds = np.clip(model.predict(pred_inputs), 0, 100)
            ax_d.plot(time_continuous, preds, color=col, lw=1.8, linestyle='-', alpha=0.85)
    ax_d.set_title('(d) Effect of Reaction Temperature vs Time (pH=7, Al₂O₃=6mg)', fontsize=11, fontweight='bold')
    ax_d.set_xlabel('Irradiation Time (min)', fontweight='bold')
    ax_d.set_ylabel('Dye Degradation (%)', fontweight='bold')
    ax_d.set_ylim(20, 100)
    ax_d.grid(True)
    ax_d.legend(loc='lower right', fontsize=8.5)

    fig.suptitle(f'Experimental Kinetics & Model Simulation ({model_name})\nPhotocatalytic Degradation over Al₂O₃ Nanoparticles', fontsize=14, fontweight='bold')
    return fig

def plot_response_surfaces(models_obj, model_name):
    """
    3D Surface and 2D Contour Response Surface Methodology (RSM) interaction plots for specific model.
    """
    from mpl_toolkits.mplot3d import Axes3D
    model = models_obj.models.get(model_name) or list(models_obj.models.values())[0]

    fig = plt.figure(figsize=(15, 6.5))
    
    # Grid for Time vs pH
    time_grid = np.linspace(20, 120, 30)
    ph_grid = np.linspace(5.0, 9.0, 30)
    T_mesh, PH_mesh = np.meshgrid(time_grid, ph_grid)
    
    flat_inputs = pd.DataFrame({
        'pH': PH_mesh.ravel(),
        'Temperature_C': [30]*len(PH_mesh.ravel()),
        'Time_min': T_mesh.ravel(),
        'Al2O3_mg': [6]*len(PH_mesh.ravel()),
        'Dye_Conc_M': [2.5e-5]*len(PH_mesh.ravel())
    })
    
    Z_degrad = model.predict(flat_inputs).reshape(PH_mesh.shape)
    Z_degrad = np.clip(Z_degrad, 0, 100)

    # Subplot 1: 3D Surface
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax1.plot_surface(PH_mesh, T_mesh, Z_degrad, cmap='viridis', edgecolor='none', alpha=0.9)
    ax1.set_xlabel('Solution pH', fontweight='bold', labelpad=8)
    ax1.set_ylabel('Time (min)', fontweight='bold', labelpad=8)
    ax1.set_zlabel('Degradation (%)', fontweight='bold', labelpad=8)
    ax1.set_title(f'(a) 3D Response Surface: {model_name}', pad=12, fontweight='bold')
    fig.colorbar(surf, ax=ax1, shrink=0.55, aspect=10, label='Degradation (%)')

    # Subplot 2: 2D Contour
    ax2 = fig.add_subplot(1, 2, 2)
    contour = ax2.contourf(PH_mesh, T_mesh, Z_degrad, levels=15, cmap='viridis')
    lines = ax2.contour(PH_mesh, T_mesh, Z_degrad, levels=10, colors='black', linewidths=0.7, alpha=0.6)
    ax2.clabel(lines, inline=True, fontsize=8, fmt='%.1f%%')
    ax2.set_xlabel('Solution pH', fontweight='bold')
    ax2.set_ylabel('Irradiation Time (min)', fontweight='bold')
    ax2.set_title('(b) 2D Iso-Response Contours (T=30°C, Al₂O₃=6mg)', pad=12, fontweight='bold')
    fig.colorbar(contour, ax=ax2, label='Predicted Degradation (%)')

    fig.suptitle(f'Response Surface Analysis (RSM) — {model_name}', fontsize=14, fontweight='bold')
    return fig
