import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor

from src.preprocessing import INPUT_FEATURES

class AlONanoparticleMLModels:
    """
    Machine Learning Suite for Al2O3 Photocatalytic Degradation Prediction.
    Supports linear, non-linear, tree ensemble, kernel, and neural architectures.
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.feature_names = list(INPUT_FEATURES)
        self.models = {
            "Support Vector Regressor (SVR)": Pipeline([
                ("scaler", StandardScaler()),
                ("svr", SVR(C=100.0, epsilon=0.1, gamma="scale", kernel="rbf"))
            ]),
            "Gradient Boosting": GradientBoostingRegressor(
                n_estimators=150, learning_rate=0.08, max_depth=4, random_state=random_state
            ),
            "XGBoost": XGBRegressor(
                n_estimators=150, learning_rate=0.08, max_depth=4, random_state=random_state
            ),
            "Polynomial Regression (Deg 2)": Pipeline([
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("scaler", StandardScaler()),
                ("reg", Ridge(alpha=1.0))
            ]),
            "Random Forest": RandomForestRegressor(
                n_estimators=200, max_depth=8, min_samples_split=2, random_state=random_state
            ),
            "Extra Trees": ExtraTreesRegressor(
                n_estimators=200, max_depth=8, random_state=random_state
            ),
            "Artificial Neural Network (MLP)": Pipeline([
                ("scaler", StandardScaler()),
                ("mlp", MLPRegressor(
                    hidden_layer_sizes=(32, 16),
                    activation="relu",
                    solver="lbfgs",
                    max_iter=500,
                    random_state=random_state
                ))
            ]),
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(
                max_depth=6, random_state=random_state
            )
        }

    def fit_all(self, X_train, y_train):
        """Fits all models on training data."""
        self.feature_names = list(X_train.columns)
        for name, model in self.models.items():
            model.fit(X_train, y_train)

    def predict_all(self, X):
        """Returns predictions from all models."""
        predictions = {}
        for name, model in self.models.items():
            preds = model.predict(X)
            # Clip between 0% and 100% physically meaningful bounds
            predictions[name] = np.clip(preds, 0.0, 100.0)
        return predictions

    def predict_single(self, input_dict, model_name="Support Vector Regressor (SVR)"):
        """
        Predicts degradation % for a single set of experimental parameters.
        """
        if model_name not in self.models:
            model_name = "Support Vector Regressor (SVR)"
            
        # Verify all input features are present
        for feat in self.feature_names:
            if feat not in input_dict or input_dict[feat] is None:
                raise ValueError(f"Missing required parameter: {feat}")
                
        row_vals = [float(input_dict[f]) for f in self.feature_names]
        input_df = pd.DataFrame([row_vals], columns=self.feature_names)
        
        model = self.models[model_name]
        raw_pred = float(model.predict(input_df)[0])
        bounded_pred = max(0.0, min(100.0, raw_pred))
        
        return round(bounded_pred, 2)

    def get_feature_importances(self, X_sample, y_sample=None):
        """
        Computes feature relative importance for all models.
        Uses MDI for tree models and Permutation Importance for kernel / linear models.
        """
        importances = {}
        for name, model in self.models.items():
            if hasattr(model, "feature_importances_"):
                imp = model.feature_importances_
                norm_imp = imp / np.sum(imp) if np.sum(imp) > 0 else imp
                importances[name] = dict(zip(self.feature_names, [round(float(v), 4) for v in norm_imp]))
            elif hasattr(model, "named_steps") and hasattr(model.named_steps.get("svr") or model.named_steps.get("mlp"), "coef_"):
                pass
            elif y_sample is not None and len(y_sample) > 0:
                try:
                    r = permutation_importance(model, X_sample, y_sample, n_repeats=5, random_state=self.random_state)
                    imp = np.maximum(0, r.importances_mean)
                    norm_imp = imp / np.sum(imp) if np.sum(imp) > 0 else imp
                    importances[name] = dict(zip(self.feature_names, [round(float(v), 4) for v in norm_imp]))
                except Exception:
                    pass
                    
        # Fallback tree baseline
        if "Gradient Boosting" in importances:
            importances["Global_Consensus"] = importances["Gradient Boosting"]
        elif "Random Forest" in importances:
            importances["Global_Consensus"] = importances["Random Forest"]
            
        return importances

    def save_models(self, save_dir="results/models"):
        """Saves models and feature signatures to disk."""
        os.makedirs(save_dir, exist_ok=True)
        for name, model in self.models.items():
            safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_") + ".joblib"
            joblib.dump(model, os.path.join(save_dir, safe_name))
        joblib.dump(self.feature_names, os.path.join(save_dir, "feature_names.joblib"))

    def load_models(self, save_dir="results/models"):
        """Loads trained models from disk."""
        if not os.path.exists(save_dir):
            return False
        feat_path = os.path.join(save_dir, "feature_names.joblib")
        if not os.path.exists(feat_path):
            return False
            
        loaded = {}
        for name in self.models.keys():
            safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_") + ".joblib"
            path = os.path.join(save_dir, safe_name)
            if os.path.exists(path):
                loaded[name] = joblib.load(path)
            else:
                return False
                
        self.models = loaded
        self.feature_names = joblib.load(feat_path)
        return True
