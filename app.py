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
from src.research_plots import generate_all_research_plots

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
    "generated_plots": {}
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

    generated_plots = generate_all_research_plots(
        df, models_suite, X_train, y_train, X_test, y_test, metrics_df, best_model_name
    )

    STATE.update({
        "df": df,
        "models_suite": models_suite,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "metrics_df": metrics_df,
        "best_model": best_model_name,
        "generated_plots": generated_plots
    })

    return STATE

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
    """Trains/evaluates all ML models and generates research figures."""
    try:
        state = get_or_train_pipeline(force_retrain=True)
        metrics_records = state["metrics_df"].to_dict(orient="records")

        return jsonify({
            "status": "success",
            "metrics": metrics_records,
            "best_model": state["best_model"],
            "plots": state["generated_plots"]
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

        # Predictions per model
        preds_test = models_suite.predict_all(X_test)
        preds_train = models_suite.predict_all(X_train)

        actual_test = [round(float(v), 2) for v in y_test.values]
        actual_train = [round(float(v), 2) for v in y_train.values]

        pred_dict = {
            m_name: [round(float(v), 2) for v in preds]
            for m_name, preds in preds_test.items()
        }

        # Calculate residuals for best model
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
    """Returns URLs and descriptions of all 300 DPI publication plots."""
    state = get_or_train_pipeline()
    
    figures_info = [
        {
            "id": "fig1_actual_vs_predicted_parity",
            "title": "Figure 1: Real vs. Predicted Parity Plot",
            "subtitle": "Actual Experimental vs. ML Predicted Degradation (%) with 1:1 Parity & Error Bands",
            "url": "/static/generated_plots/fig1_actual_vs_predicted_parity.png",
            "description": "Demonstrates model fidelity on both training (N=81) and unseen test (N=21) samples with ±5% and ±10% confidence bounds."
        },
        {
            "id": "fig2_residual_analysis",
            "title": "Figure 2: Residuals Diagnostics & Error Distribution",
            "subtitle": "Homoscedasticity Analysis and Normal Gaussian Error Distribution",
            "url": "/static/generated_plots/fig2_residual_analysis.png",
            "description": "Verifies that model errors are randomly distributed around zero without systematic bias or heteroscedasticity."
        },
        {
            "id": "fig3_model_metrics_comparison",
            "title": "Figure 3: Comparative ML Model Performance Benchmark",
            "subtitle": "R² Scores (Train, Test, 5-Fold Cross-Validation) and RMSE/MAE Errors",
            "url": "/static/generated_plots/fig3_model_metrics_comparison.png",
            "description": "Comprehensive comparative evaluation ranking 9 regression architectures on Al2O3 degradation data."
        },
        {
            "id": "fig4_feature_importance_sensitivity",
            "title": "Figure 4: Parametric Sensitivity & Feature Importance",
            "subtitle": "Relative Contribution of Experimental Operational Factors",
            "url": "/static/generated_plots/fig4_feature_importance_sensitivity.png",
            "description": "Quantifies the governing influence of Reaction Time, Solution pH, Catalyst Dosage, Dye Concentration, and Temperature."
        },
        {
            "id": "fig5_experimental_kinetics_sweeps",
            "title": "Figure 5: Experimental Degradation Kinetics Curves",
            "subtitle": "Parametric Sweeps (pH, Catalyst Dosage, Dye Conc, Temperature vs. Time)",
            "url": "/static/generated_plots/fig5_experimental_kinetics_sweeps.png",
            "description": "Overlays experimental measured points against continuous machine learning simulated degradation kinetics."
        },
        {
            "id": "fig6_response_surface_3d",
            "title": "Figure 6: 3D Response Surface & 2D Contour Maps (RSM)",
            "subtitle": "Interactive Coupling of Key Reaction Parameters",
            "url": "/static/generated_plots/fig6_response_surface_3d.png",
            "description": "Response surface methodology maps highlighting optimal operational regions for peak degradation efficiency."
        }
    ]

    return jsonify({
        "status": "success",
        "figures": figures_info
    })

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
    
    # Generate predictions for all models
    all_preds = models_suite.predict_all(X_full)
    for m_name, preds in all_preds.items():
        clean_name = "Pred_" + m_name.replace(" ", "_").replace("(", "").replace(")", "")
        df[clean_name] = [round(float(v), 2) for v in preds]

    # Best model residual
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
    state = get_or_train_pipeline()
    file_info = state["generated_plots"].get(fig_id)
    if not file_info or not os.path.exists(file_info["png_res"]):
        return jsonify({"status": "error", "message": "Figure not found."}), 404

    return send_file(
        file_info["png_res"],
        mimetype="image/png",
        as_attachment=True,
        download_name=f"{fig_id}_300dpi.png"
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Al2O3 Photocatalytic Degradation ML Portal on port {port}")
    # Pre-train pipeline on startup
    get_or_train_pipeline()
    app.run(host="0.0.0.0", port=port, debug=False)
