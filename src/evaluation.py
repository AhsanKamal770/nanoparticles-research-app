import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, KFold
from scipy.stats import pearsonr

def calculate_metrics(y_true, y_pred, n_features=5):
    """
    Calculates standard scientific regression metrics:
    R2, Adjusted R2, RMSE, MAE, MAPE, Pearson r.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    n = len(y_true)
    
    r2 = r2_score(y_true, y_pred)
    
    # Adjusted R2
    if n > n_features + 1:
        adj_r2 = 1.0 - (1.0 - r2) * ((n - 1) / (n - n_features - 1))
    else:
        adj_r2 = r2
        
    # Calculate RMSE
    try:
        from sklearn.metrics import root_mean_squared_error
        rmse = root_mean_squared_error(y_true, y_pred)
    except ImportError:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
    mae = mean_absolute_error(y_true, y_pred)
    
    # Calculate MAPE safely
    non_zero_mask = y_true != 0
    if np.any(non_zero_mask):
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100.0
    else:
        mape = 0.0
        
    # Pearson correlation r
    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        r, _ = pearsonr(y_true, y_pred)
    else:
        r = np.sqrt(max(0, r2))

    return {
        "R2": round(float(r2), 4),
        "Adj_R2": round(float(adj_r2), 4),
        "RMSE": round(float(rmse), 4),
        "MAE": round(float(mae), 4),
        "MAPE (%)": round(float(mape), 2),
        "Pearson_r": round(float(r), 4)
    }

def evaluate_all_models(models_dict, X_train, y_train, X_test, y_test, X_full=None, y_full=None, cv_folds=5):
    """
    Evaluates all trained ML models across Train, Test, and K-Fold Cross-Validation splits.
    Returns a structured DataFrame and best model recommendation.
    """
    metrics_list = []
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    for model_name, model in models_dict.items():
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        train_m = calculate_metrics(y_train, y_train_pred, n_features=X_train.shape[1])
        test_m = calculate_metrics(y_test, y_test_pred, n_features=X_test.shape[1])
        
        # 5-fold Cross-Validation on complete or training dataset
        if X_full is not None and y_full is not None:
            cv_r2_scores = cross_val_score(model, X_full, y_full, cv=cv, scoring='r2')
            try:
                cv_rmse_scores = -cross_val_score(model, X_full, y_full, cv=cv, scoring='neg_root_mean_squared_error')
            except Exception:
                cv_rmse_scores = np.sqrt(-cross_val_score(model, X_full, y_full, cv=cv, scoring='neg_mean_squared_error'))
        else:
            cv_r2_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')
            cv_rmse_scores = np.array([0.0])

        metrics_list.append({
            "Model": model_name,
            "Train_R2": train_m["R2"],
            "Test_R2": test_m["R2"],
            "Adj_R2": test_m["Adj_R2"],
            "Test_RMSE": test_m["RMSE"],
            "Test_MAE": test_m["MAE"],
            "Test_MAPE": test_m["MAPE (%)"],
            "Pearson_r": test_m["Pearson_r"],
            "CV_R2_Mean": round(float(cv_r2_scores.mean()), 4),
            "CV_R2_Std": round(float(cv_r2_scores.std()), 4),
            "CV_RMSE_Mean": round(float(cv_rmse_scores.mean()), 4)
        })

    df_metrics = pd.DataFrame(metrics_list)
    
    # Sort by Test R2 and CV R2
    df_metrics = df_metrics.sort_values(by=["Test_R2", "CV_R2_Mean", "Test_RMSE"], ascending=[False, False, True])
    best_model_name = df_metrics.iloc[0]["Model"]
    
    return df_metrics, best_model_name

def generate_photocatalytic_interpretation(prediction_value, input_dict, top_features=None):
    """
    Generates peer-review grade chemical & physical reaction mechanism analysis
    for Al2O3 nanoparticle photocatalytic dye degradation.
    """
    ph = float(input_dict.get("pH", 7.0))
    temp = float(input_dict.get("Temperature_C", 30.0))
    time = float(input_dict.get("Time_min", 60.0))
    catalyst = float(input_dict.get("Al2O3_mg", 6.0))
    dye = float(input_dict.get("Dye_Conc_M", 0.000025))

    mechanisms = []
    
    # 1. Reaction Time & Kinetic Order
    if time >= 100:
        mechanisms.append(
            f"<b>Extended Irradiation Time ({time:.0f} min):</b> Prolonged light exposure promotes extensive generation of photo-induced hydroxyl (•OH) and superoxide (•O₂⁻) radicals, shifting the Langmuir-Hinshelwood pseudo-first-order degradation toward near-complete mineralization."
        )
    elif time <= 40:
        mechanisms.append(
            f"<b>Initial Kinetic Phase ({time:.0f} min):</b> Early stage of photocatalysis dominated by active radical generation and initial dye adsorption on Al₂O₃ active crystalline facets."
        )

    # 2. pH & Surface Charge Mechanics
    if 6.5 <= ph <= 7.5:
        mechanisms.append(
            f"<b>Optimal Neutral pH ({ph:.1f}):</b> Near the point of zero charge (pH_pzc) of green Al₂O₃ nanoparticles (~7.0), electrostatic repulsion is minimized, maximizing dye molecule adsorption without surface passivation or excessive radical scavenging."
        )
    elif ph < 6.5:
        mechanisms.append(
            f"<b>Acidic Solution (pH {ph:.1f}):</b> Protonation creates a positively charged catalyst surface (Al-OH₂⁺). Depending on dye chromophore ionic charge, excessive competitive adsorption of H⁺ or dissolution effects can slightly moderate degradation efficiency."
        )
    else:
        mechanisms.append(
            f"<b>Alkaline Solution (pH {ph:.1f}):</b> Deprotonation creates negative surface charge (Al-O⁻). While OH⁻ ions facilitate •OH radical formation, high ionic strength and repulsion against anionic dye groups can reduce net reaction rates."
        )

    # 3. Catalyst Dosage & Light Penetration Effect
    if catalyst == 6.0:
        mechanisms.append(
            f"<b>Optimal Catalyst Dosage (6.0 mg):</b> Provides the optimal balance of specific surface active sites and photon harvesting efficiency without inducing suspension opacity."
        )
    elif catalyst > 6.0:
        mechanisms.append(
            f"<b>High Catalyst Dosage ({catalyst:.1f} mg) Screening Effect:</b> Excess catalyst particles increase solution turbidity, causing photon scattering and optical screening, which limits light penetration to inner reaction zones."
        )
    else:
        mechanisms.append(
            f"<b>Sub-optimal Catalyst Dosage ({catalyst:.1f} mg):</b> Lower catalyst loading restricts the total number of photogenerated electron-hole pairs available per unit volume."
        )

    # 4. Initial Dye Concentration Competition
    if dye > 3.5e-5:
        mechanisms.append(
            f"<b>High Dye Concentration ({dye*1e5:.1f} × 10⁻⁵ M):</b> High chromophore density absorbs incident light ('inner filter effect') and saturates catalyst active sites, decreasing radical generation efficiency."
        )
    else:
        mechanisms.append(
            f"<b>Favorable Substrate Ratio ({dye*1e5:.1f} × 10⁻⁵ M):</b> Active radical flux is in stoichiometric excess relative to dye molecules, facilitating rapid bond cleavage and decolourization."
        )

    # 5. Temperature Effect
    if 30.0 <= temp <= 40.0:
        mechanisms.append(
            f"<b>Optimal Reaction Temperature ({temp:.0f}°C):</b> Enhances molecular collision frequency and mass transfer coefficient of dye molecules towards the Al₂O₃ photocatalyst surface."
        )
    elif temp > 40.0:
        mechanisms.append(
            f"<b>Elevated Temperature ({temp:.0f}°C):</b> While accelerating collision kinetics, high thermal energy can reduce dissolved oxygen solubility and promote electron-hole thermal recombination."
        )

    feat_summary = ""
    if top_features:
        top_items = sorted(top_features.items(), key=lambda x: x[1], reverse=True)[:3]
        feat_str = ", ".join([f"<b>{k}</b> ({v*100:.1f}%)" for k, v in top_items])
        feat_summary = f"Dominant experimental drivers for this condition: {feat_str}."

    return {
        "predicted_degradation": f"{prediction_value:.2f}%",
        "feature_summary": feat_summary,
        "mechanism_notes": mechanisms
    }
