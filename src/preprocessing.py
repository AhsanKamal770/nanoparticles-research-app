import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

SYNTHESIS_FEATURES = ["pH", "Temperature", "Time", "Zn_Concentration", "Plant_Extract"]

CHARACTERIZATION_FEATURES = [
    "XRD_Size", "UV_Bandgap", "DLS_Size", "PDI", "Zeta_Potential", "SEM_Size"
]

TARGET_CONFIGS = {
    "SEM_Size": {
        "label": "SEM Particle Size (nm)",
        "unit": "nm",
        "features": SYNTHESIS_FEATURES,
        "rq": "RQ1: Synthesis Parameters -> Particle Size",
        "description": "Predict average nanoparticle size directly from green synthesis experimental conditions before characterization."
    },
    "XRD_Size": {
        "label": "XRD Crystallite Size (nm)",
        "unit": "nm",
        "features": SYNTHESIS_FEATURES,
        "rq": "RQ2: Synthesis Parameters -> Crystallite Size",
        "description": "Predict crystallite domain size obtained via XRD peak broadening from synthesis parameters."
    },
    "UV_Bandgap": {
        "label": "UV-Vis Band Gap (eV)",
        "unit": "eV",
        "features": SYNTHESIS_FEATURES,
        "rq": "RQ3: Synthesis Parameters -> Optical Band Gap",
        "description": "Predict optical band gap energy calculated from Tauc plot of UV-Vis spectra."
    },
    "Degradation": {
        "label": "Photocatalytic Degradation (%)",
        "unit": "%",
        "features": SYNTHESIS_FEATURES + CHARACTERIZATION_FEATURES,
        "rq": "RQ4: Synthesis & Characterization -> Photocatalytic Performance",
        "description": "Predict pollutant degradation efficiency after synthesis and full physicochemical characterization."
    }
}

def load_dataset(file_path="data/raw/zn_particle.xlsx"):
    """Reads Excel or CSV dataset."""
    if not os.path.exists(file_path):
        # Fallback to CSV if xlsx not present
        csv_path = file_path.replace(".xlsx", ".csv")
        if os.path.exists(csv_path):
            file_path = csv_path
        else:
            raise FileNotFoundError(f"Dataset file not found at {file_path}")

    if file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)
    return df

def inspect_dataset(df):
    """Returns dataset summary statistics dictionary."""
    summary = {
        "shape": list(df.shape),
        "columns": list(df.columns),
        "null_counts": df.isnull().sum().to_dict(),
        "duplicate_count": int(df.duplicated().sum()),
        "describe": df.describe().round(3).to_dict(),
        "head": df.head(10).to_dict(orient="records")
    }
    return summary

def get_train_test_data(df, target_col="SEM_Size", test_size=0.20, random_state=42):
    """
    Extracts features and target according to target configuration, preventing data leakage.
    Returns X_train, X_test, y_train, y_test, feature_names
    """
    if target_col not in TARGET_CONFIGS:
        raise ValueError(f"Unknown target column '{target_col}'. Allowed: {list(TARGET_CONFIGS.keys())}")
    
    config = TARGET_CONFIGS[target_col]
    features = [f for f in config["features"] if f in df.columns and f != target_col]
    
    X = df[features].copy()
    y = df[target_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test, features

if __name__ == "__main__":
    df = load_dataset()
    print("Dataset loaded successfully. Shape:", df.shape)
    info = inspect_dataset(df)
    print("Null counts:", info["null_counts"])
    print("Duplicate count:", info["duplicate_count"])
    X_tr, X_te, y_tr, y_te, feats = get_train_test_data(df, "SEM_Size")
    print(f"Train shape: {X_tr.shape}, Test shape: {X_te.shape}, Features: {feats}")
