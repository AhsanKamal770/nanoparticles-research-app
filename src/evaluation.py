import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def calculate_metrics(y_true, y_pred):
    """Calculates R2, RMSE, MAE, and MAPE metrics."""
    r2 = r2_score(y_true, y_pred)
    
    # Calculate RMSE (compatible with all scikit-learn versions)
    try:
        from sklearn.metrics import root_mean_squared_error
        rmse = root_mean_squared_error(y_true, y_pred)
    except ImportError:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
    mae = mean_absolute_error(y_true, y_pred)
    
    # Calculate MAPE safely (avoid division by zero)
    non_zero_mask = y_true != 0
    if np.any(non_zero_mask):
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
    else:
        mape = 0.0

    return {
        "R2": round(float(r2), 4),
        "RMSE": round(float(rmse), 4),
        "MAE": round(float(mae), 4),
        "MAPE (%)": round(float(mape), 2)
    }

def evaluate_all_models(y_true, predictions_dict):
    """
    Evaluates predictions from all models and returns comparison DataFrame and best model name.
    """
    metrics_list = []
    for model_name, y_pred in predictions_dict.items():
        m = calculate_metrics(y_true, y_pred)
        m["Model"] = model_name
        metrics_list.append(m)

    df_metrics = pd.DataFrame(metrics_list)[["Model", "R2", "RMSE", "MAE", "MAPE (%)"]]
    
    # Select best model based on highest R2 (and lowest RMSE)
    best_row = df_metrics.sort_values(by=["R2", "RMSE"], ascending=[False, True]).iloc[0]
    best_model_name = best_row["Model"]

    return df_metrics, best_model_name

def generate_scientific_interpretation(target_name, top_features, prediction_value, input_dict):
    """
    Generates chemical & physical interpretation of model predictions based on green synthesis kinetics.
    """
    interpretations = []
    
    if target_name in ["SEM_Size", "XRD_Size"]:
        interpretations.append(
            "<b>Nucleation vs Growth Kinetics:</b> Higher plant extract concentration and elevated pH supply active phytochemical capping agents (flavonoids/phenols) and OH⁻ ions, accelerating nucleation rate and inhibiting particle growth. This yields smaller nanoparticle and crystallite sizes."
        )
        if input_dict.get("Temperature", 60) > 75:
            interpretations.append(
                "<b>Thermal Agglomeration:</b> High reaction temperature (>75°C) increases thermal kinetic energy, leading to higher collision rates and slight crystal grain coalescence."
            )
    elif target_name == "UV_Bandgap":
        interpretations.append(
            "<b>Quantum Confinement Effect:</b> As nanoparticle size decreases below the exciton Bohr radius (~2-5 nm region), the optical bandgap shifts towards higher energy (blue shift) due to quantum size confinement."
        )
    elif target_name == "Degradation":
        interpretations.append(
            "<b>Photocatalytic Efficiency:</b> Smaller nanoparticle sizes provide a larger specific surface area and higher density of reactive surface sites. High negative Zeta Potential improves colloidal dispersion stability, preventing agglomeration during photocatalytic reaction under UV/visible irradiation."
        )

    top_feat_str = ", ".join([f"<b>{k}</b> ({v*100:.1f}% relative weight)" for k, v in top_features.items()])
    summary = f"Key features influencing this prediction: {top_feat_str}."
    
    return {
        "summary": summary,
        "mechanism_notes": interpretations
    }

if __name__ == "__main__":
    y_true = np.array([25.0, 28.0, 30.0, 22.0, 35.0])
    y_pred = np.array([24.5, 28.2, 29.5, 22.8, 34.1])
    m = calculate_metrics(y_true, y_pred)
    print("Test Metrics:", m)
