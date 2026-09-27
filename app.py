import os
import io
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS

from src.preprocessing import (
    load_dataset, inspect_dataset, get_train_test_data,
    INPUT_FEATURES, TARGET_COLUMN, FEATURE_METADATA, TARGET_METADATA
)
from src.models import AlONanoparticleMLModels
from src.evaluation import evaluate_all_models, generate_photocatalytic_interpretation
from src.research_plots import generate_plots_for_model, get_model_slug

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# Global model cache and state
STATE = {
    "df": None,
    "models_suite": None,
    "X_train": None,
    "X_test": None,
    "y_train": None,
    "y_test": None,
    "metrics_df": None,
    "best_model": "Support Vector Regressor (SVR)",
    "model_plots_cache": {}
}

def get_or_train_pipeline(force_retrain=False):
    """Initializes or retrieves trained ML pipeline and cached figures."""
    if STATE["models_suite"] is not None and not force_retrain:
        return STATE

    df = load_dataset()
    X_train, X_test, y_train, y_test, feats = get_train_test_data(df)
    X_full = df[feats]
    y_full = df[TARGET_COLUMN]

    models_suite = AlONanoparticleMLModels()
    metrics_path = "results/models/metrics_cache.joblib"
    loaded = False if force_retrain else models_suite.load_models()

    if loaded and os.path.exists(metrics_path) and not force_retrain:
        import joblib
        cached = joblib.load(metrics_path)
        metrics_df = cached["metrics_df"]
        best_model_name = cached["best_model_name"]
    else:
        models_suite.fit_all(X_train, y_train)
        models_suite.save_models()
        metrics_df, best_model_name = evaluate_all_models(
            models_suite.models, X_train, y_train, X_test, y_test, X_full, y_full
        )
        import joblib
        os.makedirs("results/models", exist_ok=True)
        joblib.dump({"metrics_df": metrics_df, "best_model_name": best_model_name}, metrics_path)

    STATE.update({
        "df": df,
        "models_suite": models_suite,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "metrics_df": metrics_df,
        "best_model": best_model_name,
        "model_plots_cache": {}
    })

    return STATE

def resolve_model_name(req_name, available_models, default_model):
    """Resolves requested model name with full name, alias, or case-insensitive matching."""
    if not req_name:
        return default_model
    if req_name in available_models:
        return req_name
    
    req_clean = req_name.strip().lower()
    for m in available_models:
        if m.lower() == req_clean:
            return m
            
    aliases = {
        "svr": "Support Vector Regressor (SVR)",
        "ann": "Artificial Neural Network (MLP)",
        "mlp": "Artificial Neural Network (MLP)",
        "gradient boosting": "Gradient Boosting",
        "gradient boost": "Gradient Boosting",
        "random forest": "Random Forest",
        "xgboost": "XGBoost",
        "polynomial": "Polynomial Regression (Deg 2)",
        "linear": "Linear Regression",
        "decision tree": "Decision Tree",
        "extra trees": "Extra Trees"
    }
    if req_clean in aliases and aliases[req_clean] in available_models:
        return aliases[req_clean]
        
    for m in available_models:
        if req_clean in m.lower():
            return m
            
    return default_model

def get_figures_for_model(model_name):
    """Generates or retrieves cached research figures for the specified model."""
    state = get_or_train_pipeline()
    available_models = list(state["models_suite"].models.keys())
    model_name = resolve_model_name(model_name, available_models, state["best_model"])

    if model_name not in state["model_plots_cache"]:
        plots = generate_plots_for_model(
            state["df"], state["models_suite"],
            state["X_train"], state["y_train"],
            state["X_test"], state["y_test"],
            state["metrics_df"],
            model_name=model_name
        )
        state["model_plots_cache"][model_name] = plots

    slug = get_model_slug(model_name)
    m_row = state["metrics_df"][state["metrics_df"]["Model"] == model_name]
    r2_val = m_row.iloc[0]["Test_R2"] if len(m_row) > 0 else 0.0
    rmse_val = m_row.iloc[0]["Test_RMSE"] if len(m_row) > 0 else 0.0

    figures_info = [
        {
            "id": f"fig1_actual_vs_predicted_parity_{slug}",
            "title": f"Figure 1: Real vs. Predicted Parity Plot ({model_name})",
            "subtitle": f"Experimental vs. Predicted Degradation (%) • Test R² = {r2_val:.4f}, RMSE = {rmse_val:.2f}%",
            "url": f"/static/generated_plots/fig1_actual_vs_predicted_parity_{slug}.png",
            "description": f"Demonstrates {model_name} fidelity on both training (N=81) and unseen test (N=21) samples with ±5% and ±10% confidence bounds."
        },
        {
            "id": f"fig2_residual_analysis_{slug}",
            "title": f"Figure 2: Residuals Diagnostics & Error Distribution ({model_name})",
            "subtitle": f"Homoscedasticity Analysis and Normal Gaussian Error Distribution for {model_name}",
            "url": f"/static/generated_plots/fig2_residual_analysis_{slug}.png",
            "description": f"Evaluates {model_name} prediction errors (Actual - Predicted) to verify homoscedasticity and zero-mean normality."
        },
        {
            "id": "fig3_model_metrics_comparison",
            "title": "Figure 3: Comparative ML Model Performance Benchmark (All Models)",
            "subtitle": "R² Scores (Train, Test, 5-Fold Cross-Validation) and RMSE/MAE Error Benchmark",
            "url": "/static/generated_plots/fig3_model_metrics_comparison.png",
            "description": "Comprehensive benchmark ranking all 9 machine learning regression algorithms evaluated on Al2O3 degradation data."
        },
        {
            "id": f"fig4_feature_importance_sensitivity_{slug}",
            "title": f"Figure 4: Parametric Sensitivity & Feature Importance ({model_name})",
            "subtitle": f"Relative Operational Factor Influence Breakdown ({model_name})",
            "url": f"/static/generated_plots/fig4_feature_importance_sensitivity_{slug}.png",
            "description": f"Quantifies the governing relative percentage impact of Time, Dye Concentration, Catalyst Dosage, pH, and Temperature using {model_name}."
        },
        {
            "id": f"fig5_experimental_kinetics_sweeps_{slug}",
            "title": f"Figure 5: Experimental Degradation Kinetics Curves ({model_name})",
            "subtitle": f"Experimental Data Points vs. {model_name} Continuous Simulated Degradation Kinetics",
            "url": f"/static/generated_plots/fig5_experimental_kinetics_sweeps_{slug}.png",
            "description": f"4-panel parametric sweeps: Solution pH, Catalyst Dosage, Dye Concentration, and Temperature vs Time with {model_name} continuous curves."
        },
        {
            "id": f"fig6_response_surface_3d_{slug}",
            "title": f"Figure 6: 3D Response Surface & 2D Iso-Response Contours ({model_name})",
            "subtitle": f"Interactive Coupling of Key Reaction Variables Simulated by {model_name}",
            "url": f"/static/generated_plots/fig6_response_surface_3d_{slug}.png",
            "description": f"3D response surface and 2D contour maps simulated using {model_name}, highlighting optimal operational coordinates."
        }
    ]

    return figures_info

@app.route("/")
def index():
    """Main Research & Modeling Portal UI."""
    return render_template("index.html")

@app.route("/api/metadata", methods=["GET"])
def api_metadata():
    """Returns metadata about experimental inputs and degradation target."""
    return jsonify({
        "status": "success",
        "features": FEATURE_METADATA,
        "target": TARGET_METADATA,
        "input_feature_keys": INPUT_FEATURES
    })

@app.route("/api/data/inspect", methods=["GET"])
def api_inspect_data():
    """Returns dataset summary statistics and preview rows."""
    try:
        df = load_dataset()
        summary = inspect_dataset(df)
        return jsonify({
            "status": "success",
            "summary": summary
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/train", methods=["POST"])
def api_train_models():
    """Trains/evaluates all ML models and resets figure cache."""
    try:
        state = get_or_train_pipeline(force_retrain=True)
        metrics_records = state["metrics_df"].to_dict(orient="records")

        return jsonify({
            "status": "success",
            "metrics": metrics_records,
            "best_model": state["best_model"]
        })
    except Exception as e:
        return jsonify({"status": "error", "message": f"Training failed: {str(e)}"}), 500

@app.route("/api/visualizations", methods=["GET"])
def api_visualizations():
    """Returns data payloads for interactive charts."""
    try:
        state = get_or_train_pipeline()
        X_test = state["X_test"]
        y_test = state["y_test"]
        X_train = state["X_train"]
        y_train = state["y_train"]
        models_suite = state["models_suite"]

        preds_test = models_suite.predict_all(X_test)
        preds_train = models_suite.predict_all(X_train)

        actual_test = [round(float(v), 2) for v in y_test.values]
        actual_train = [round(float(v), 2) for v in y_train.values]

        pred_dict = {
            m_name: [round(float(v), 2) for v in preds]
            for m_name, preds in preds_test.items()
        }

        best_model = state["best_model"]
        best_preds = preds_test[best_model]
        residuals = [round(float(act - pred), 2) for act, pred in zip(actual_test, best_preds)]

        importances = models_suite.get_feature_importances(X_train, y_train)

        return jsonify({
            "status": "success",
            "best_model": best_model,
            "actual_vs_pred": {
                "actual_test": actual_test,
                "actual_train": actual_train,
                "predictions_test": pred_dict,
                "predictions_train": {
                    m_name: [round(float(v), 2) for v in preds]
                    for m_name, preds in preds_train.items()
                }
            },
            "residuals": {
                "predicted": [round(float(v), 2) for v in best_preds],
                "residuals": residuals
            },
            "feature_importances": importances,
            "metrics": state["metrics_df"].to_dict(orient="records")
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/research-figures", methods=["GET"])
def api_research_figures():
    """Returns URLs and descriptions of 300 DPI publication plots for the requested model."""
    try:
        state = get_or_train_pipeline()
        req_model = request.args.get("model", state["best_model"])
        
        figures_info = get_figures_for_model(req_model)
        available_models = list(state["models_suite"].models.keys())

        return jsonify({
            "status": "success",
            "selected_model": req_model,
            "available_models": available_models,
            "figures": figures_info
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Predicts photocatalytic degradation % for the 5 exact experimental inputs."""
    data = request.get_json() or {}
    model_name = data.get("model", "Support Vector Regressor (SVR)")
    inputs = data.get("inputs", {})

    state = get_or_train_pipeline()
    models_suite = state["models_suite"]

    try:
        pred_val = models_suite.predict_single(inputs, model_name=model_name)
    except ValueError as ve:
        return jsonify({"status": "error", "message": str(ve)}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": f"Prediction error: {str(e)}"}), 400

    importances = models_suite.get_feature_importances(state["X_train"], state["y_train"])
    top_importances = importances.get(model_name) or importances.get("Gradient Boosting", {})

    interpretation = generate_photocatalytic_interpretation(
        prediction_value=pred_val,
        input_dict=inputs,
        top_features=top_importances
    )

    return jsonify({
        "status": "success",
        "model_used": model_name,
        "predicted_value": pred_val,
        "unit": "%",
        "interpretation": interpretation
    })

@app.route("/api/export-results", methods=["GET"])
def api_export_results():
    """Exports full dataset with predictions from all ML models and residuals as CSV."""
    state = get_or_train_pipeline()
    df = state["df"].copy()
    models_suite = state["models_suite"]

    feats = INPUT_FEATURES
    X_full = df[feats]
    
    all_preds = models_suite.predict_all(X_full)
    for m_name, preds in all_preds.items():
        clean_name = "Pred_" + m_name.replace(" ", "_").replace("(", "").replace(")", "")
        df[clean_name] = [round(float(v), 2) for v in preds]

    best_clean = "Pred_" + state["best_model"].replace(" ", "_").replace("(", "").replace(")", "")
    if best_clean in df.columns:
        df["Residual_Error_%"] = (df[TARGET_COLUMN] - df[best_clean]).round(2)
        df["Abs_Error_%"] = df["Residual_Error_%"].abs()

    output = io.BytesIO()
    df.to_csv(output, index=False, encoding='utf-8')
    output.seek(0)

    return send_file(
        output,
        mimetype="text/csv",
        as_attachment=True,
        download_name="Al2O3_Photocatalytic_Degradation_ML_Predictions.csv"
    )

@app.route("/api/download-figure/<fig_id>", methods=["GET"])
def api_download_figure(fig_id):
    """Downloads high-resolution 300 DPI PNG research figure."""
    # Look in results/figures
    possible_paths = [
        f"results/figures/{fig_id}.png",
        f"results/figures/{fig_id}",
        f"static/generated_plots/{fig_id}.png",
        f"static/generated_plots/{fig_id}"
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return send_file(
                p,
                mimetype="image/png",
                as_attachment=True,
                download_name=f"{fig_id}_300dpi.png"
            )

    return jsonify({"status": "error", "message": "Figure file not found."}), 404

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Al2O3 Photocatalytic Degradation ML Portal on port {port}")
    get_or_train_pipeline()
    app.run(host="0.0.0.0", port=port, debug=False)
