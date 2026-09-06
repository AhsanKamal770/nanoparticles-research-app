import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd

from src.preprocessing import load_dataset, inspect_dataset, get_train_test_data, TARGET_CONFIGS
from src.models import NanoparticleMLModels
from src.evaluation import evaluate_all_models, generate_scientific_interpretation

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# Global model cache per target
model_cache = {}

def get_or_train_models(target_col="SEM_Size"):
    """Fetches cached models or trains them if not already cached."""
    df = load_dataset()
    X_train, X_test, y_train, y_test, feats = get_train_test_data(df, target_col)
    
    ml_models = NanoparticleMLModels(target_name=target_col)
    loaded = ml_models.load_models()
    
    if not loaded:
        ml_models.fit_all(X_train, y_train)
        ml_models.save_models()

    preds = ml_models.predict_all(X_test)
    df_metrics, best_model_name = evaluate_all_models(y_test.values, preds)

    model_cache[target_col] = {
        "ml_models": ml_models,
        "X_test": X_test,
        "y_test": y_test,
        "preds": preds,
        "metrics_df": df_metrics,
        "best_model": best_model_name
    }
    return model_cache[target_col]

@app.route("/")
def index():
    """Serves the main web interface."""
    return render_template("index.html")

@app.route("/api/targets", methods=["GET"])
def get_targets():
    """Returns available research target questions and their parameters."""
    return jsonify({
        "status": "success",
        "targets": TARGET_CONFIGS
    })

@app.route("/api/data/inspect", methods=["GET"])
def api_inspect_data():
    """Returns dataset summary statistics, column types, missing values, and preview."""
    target = request.args.get("target", "SEM_Size")
    df = load_dataset()
    summary = inspect_dataset(df)
    target_info = TARGET_CONFIGS.get(target, TARGET_CONFIGS["SEM_Size"])
    
    return jsonify({
        "status": "success",
        "target": target,
        "target_info": target_info,
        "summary": summary
    })

@app.route("/api/train", methods=["POST"])
def api_train_models():
    """Trains all 4 models (Linear Regression, Decision Tree, Random Forest, XGBoost) for target."""
    data = request.get_json() or {}
    target = data.get("target", "SEM_Size")
    
    if target not in TARGET_CONFIGS:
        return jsonify({"status": "error", "message": f"Invalid target '{target}'"}), 400

    cache = get_or_train_models(target)
    metrics_records = cache["metrics_df"].to_dict(orient="records")

    return jsonify({
        "status": "success",
        "target": target,
        "target_info": TARGET_CONFIGS[target],
        "metrics": metrics_records,
        "best_model": cache["best_model"]
    })

@app.route("/api/visualizations", methods=["GET"])
def api_visualizations():
    """Returns data for Chart.js graphs (Actual vs Predicted, Feature Importances)."""
    target = request.args.get("target", "SEM_Size")
    cache = model_cache.get(target) or get_or_train_models(target)
    
    y_test_list = [round(float(v), 2) for v in cache["y_test"].values]
    pred_dict = {
        name: [round(float(v), 2) for v in preds]
        for name, preds in cache["preds"].items()
    }
    
    importances = cache["ml_models"].get_feature_importances(cache["X_test"])

    return jsonify({
        "status": "success",
        "target": target,
        "actual_vs_pred": {
            "actual": y_test_list,
            "predictions": pred_dict
        },
        "feature_importances": importances
    })

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Calculates property prediction for given experimental inputs and returns chemical analysis."""
    data = request.get_json() or {}
    target = data.get("target", "SEM_Size")
    model_name = data.get("model", "Random Forest")
    inputs = data.get("inputs", {})

    if target not in TARGET_CONFIGS:
        return jsonify({"status": "error", "message": f"Invalid target '{target}'"}), 400

    cache = model_cache.get(target) or get_or_train_models(target)
    ml_models = cache["ml_models"]
    
    try:
        pred_val = ml_models.predict_single(inputs, model_name)
    except ValueError as ve:
        return jsonify({"status": "error", "message": str(ve)}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": f"Prediction failed: {str(e)}"}), 400

    top_importances = ml_models.get_feature_importances(cache["X_test"]).get(model_name) or ml_models.get_feature_importances(cache["X_test"]).get("Random Forest", {})

    target_info = TARGET_CONFIGS[target]
    interpretation = generate_scientific_interpretation(
        target_name=target,
        top_features=top_importances,
        prediction_value=pred_val,
        input_dict=inputs
    )

    return jsonify({
        "status": "success",
        "target": target,
        "model_used": model_name,
        "predicted_value": pred_val,
        "unit": target_info["unit"],
        "interpretation": interpretation
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Nanoparticle ML Web Portal on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
