import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

class NanoparticleMLModels:
    def __init__(self, target_name="SEM_Size", random_state=42):
        self.target_name = target_name
        self.random_state = random_state
        self.models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(random_state=random_state, max_depth=6),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=random_state, max_depth=8),
            "XGBoost": XGBRegressor(n_estimators=100, random_state=random_state, learning_rate=0.08, max_depth=4)
        }
        self.feature_names = []

    def fit_all(self, X_train, y_train):
        """Fits all 4 models on training data."""
        self.feature_names = list(X_train.columns)
        for name, model in self.models.items():
            model.fit(X_train, y_train)

    def predict_all(self, X):
        """Returns dictionary of predictions for each model."""
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X)
        return predictions

    def predict_single(self, input_dict, model_name="Random Forest"):
        """Predicts target for a single dictionary of input feature values."""
        if model_name not in self.models:
            model_name = "Random Forest"
        
        if not self.feature_names:
            raise ValueError("Models are not trained or feature names are missing.")

        missing_feats = [f for f in self.feature_names if f not in input_dict or input_dict[f] is None or pd.isna(input_dict[f])]
        if missing_feats:
            raise ValueError(f"Missing or invalid values for feature(s): {', '.join(missing_feats)}")

        input_df = pd.DataFrame([input_dict])[self.feature_names]
        model = self.models[model_name]
        pred_val = float(model.predict(input_df)[0])
        return round(pred_val, 2)

    def get_feature_importances(self, X):
        """Returns feature importances for tree-based models."""
        importances = {}
        for name in ["Random Forest", "XGBoost", "Decision Tree"]:
            model = self.models[name]
            if hasattr(model, "feature_importances_"):
                imp = model.feature_importances_
                importances[name] = dict(zip(self.feature_names, [round(float(v), 4) for v in imp]))
        return importances

    def save_models(self, save_dir="results/models"):
        """Saves trained models to disk."""
        target_dir = os.path.join(save_dir, self.target_name)
        os.makedirs(target_dir, exist_ok=True)
        for name, model in self.models.items():
            fname = name.lower().replace(" ", "_") + ".joblib"
            joblib.dump(model, os.path.join(target_dir, fname))
        joblib.dump(self.feature_names, os.path.join(target_dir, "features.joblib"))

    def load_models(self, save_dir="results/models"):
        """Loads trained models from disk."""
        target_dir = os.path.join(save_dir, self.target_name)
        if not os.path.exists(target_dir):
            return False
            
        feat_path = os.path.join(target_dir, "features.joblib")
        if not os.path.exists(feat_path):
            return False

        loaded_models = {}
        for name in self.models.keys():
            fname = name.lower().replace(" ", "_") + ".joblib"
            path = os.path.join(target_dir, fname)
            if not os.path.exists(path):
                return False
            loaded_models[name] = joblib.load(path)
            
        self.models = loaded_models
        self.feature_names = joblib.load(feat_path)
        return True

if __name__ == "__main__":
    from src.preprocessing import load_dataset, get_train_test_data
    df = load_dataset()
    X_tr, X_te, y_tr, y_te, feats = get_train_test_data(df, "SEM_Size")
    ml_models = NanoparticleMLModels("SEM_Size")
    ml_models.fit_all(X_tr, y_tr)
    preds = ml_models.predict_all(X_te)
    print("Models fitted successfully. Prediction keys:", list(preds.keys()))
