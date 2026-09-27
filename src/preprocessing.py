import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold

# Input features strictly from the AlO dataset
INPUT_FEATURES = [
    "pH",
    "Temperature_C",
    "Time_min",
    "Al2O3_mg",
    "Dye_Conc_M"
]

TARGET_COLUMN = "Degradation_Percent"

FEATURE_METADATA = {
    "pH": {
        "label": "Solution pH",
        "unit": "pH",
        "symbol": "pH",
        "min": 5.0,
        "max": 9.0,
        "step": 0.5,
        "default": 7.0,
        "exp_values": [5, 6, 7, 8, 9],
        "description": "Acidity / alkalinity of dye solution influencing surface charge and zeta potential of Al2O3 nanoparticles."
    },
    "Temperature_C": {
        "label": "Reaction Temperature",
        "unit": "°C",
        "symbol": "T",
        "min": 20.0,
        "max": 60.0,
        "step": 5.0,
        "default": 30.0,
        "exp_values": [20, 30, 40, 50, 60],
        "description": "Thermal kinetic condition of photocatalytic degradation system."
    },
    "Time_min": {
        "label": "Reaction / Irradiation Time",
        "unit": "min",
        "symbol": "t",
        "min": 20.0,
        "max": 120.0,
        "step": 10.0,
        "default": 60.0,
        "exp_values": [20, 40, 60, 80, 100, 120],
        "description": "Duration of light exposure and reactive radical contact."
    },
    "Al2O3_mg": {
        "label": "Al₂O₃ Nanoparticle Dosage",
        "unit": "mg",
        "symbol": "m(Al₂O₃)",
        "min": 2.0,
        "max": 10.0,
        "step": 1.0,
        "default": 6.0,
        "exp_values": [2, 4, 6, 8, 10],
        "description": "Mass of aluminum oxide photocatalyst providing active surface adsorption and reactive radical generation sites."
    },
    "Dye_Conc_M": {
        "label": "Initial Dye Concentration",
        "unit": "M",
        "symbol": "C₀",
        "min": 0.000025,
        "max": 0.000045,
        "step": 0.000005,
        "default": 0.000025,
        "exp_values": [0.000025, 0.000030, 0.000035, 0.000040, 0.000045],
        "exp_display_map": {
            0.000025: "2.5 × 10⁻⁵ M",
            0.000030: "3.0 × 10⁻⁵ M",
            0.000035: "3.5 × 10⁻⁵ M",
            0.000040: "4.0 × 10⁻⁵ M",
            0.000045: "4.5 × 10⁻⁵ M"
        },
        "description": "Starting pollutant concentration competing for photogenerated electron-hole pairs."
    }
}

TARGET_METADATA = {
    "name": "Degradation_Percent",
    "label": "Photocatalytic Degradation Efficiency",
    "unit": "%",
    "symbol": "η (%)",
    "min": 0.0,
    "max": 100.0,
    "description": "Percentage removal / degradation of target dye pollutant catalyzed by Al2O3 nanoparticles under irradiation."
}

def parse_dye_concentration(val):
    """
    Parses dye concentration from scientific notation string or numeric format into float.
    Handles '2.5 × 10⁻⁵', '2.5 x 10^-5', '2.5e-5', or 0.000025.
    """
    if isinstance(val, (int, float)):
        return float(val)
    
    val_str = str(val).strip()
    superscripts = {
        '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
        '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9', '⁻': '-'
    }
    for k, s in superscripts.items():
        val_str = val_str.replace(k, s)
    
    val_str = val_str.replace('×', '*').replace('x', '*').replace('X', '*')
    
    if '* 10^' in val_str:
        parts = val_str.split('* 10^')
        return float(parts[0].strip()) * (10 ** float(parts[1].strip()))
    elif '* 10-' in val_str:
        parts = val_str.split('* 10-')
        return float(parts[0].strip()) * (10 ** -float(parts[1].strip()))
    elif '* 10' in val_str:
        parts = val_str.split('* 10')
        return float(parts[0].strip()) * (10 ** float(parts[1].strip()))
    
    try:
        return float(val_str)
    except Exception:
        # Fallback eval for pure math strings like 2.5*10**-5
        return float(eval(val_str))

def load_dataset(file_path=None):
    """
    Loads and standardizes the real Al2O3 photocatalytic degradation dataset.
    """
    possible_paths = [
        file_path,
        "AlO particles.xlsx",
        "data/raw/AlO particles.xlsx",
        "../AlO particles.xlsx"
    ]
    
    selected_path = None
    for p in possible_paths:
        if p and os.path.exists(p):
            selected_path = p
            break
            
    if not selected_path:
        raise FileNotFoundError("Could not find 'AlO particles.xlsx' in workspace.")
    
    df_raw = pd.read_excel(selected_path)
    
    # Map raw column names to standardized clean identifiers
    col_mapping = {}
    for col in df_raw.columns:
        c_clean = str(col).strip()
        if "pH" in c_clean:
            col_mapping[col] = "pH"
        elif "Temp" in c_clean:
            col_mapping[col] = "Temperature_C"
        elif "Time" in c_clean:
            col_mapping[col] = "Time_min"
        elif "Al2O3" in c_clean or "NP" in c_clean or "AlO" in c_clean:
            col_mapping[col] = "Al2O3_mg"
        elif "Dye Conc" in c_clean:
            col_mapping[col] = "Dye_Conc_Raw"
        elif "Degradation" in c_clean:
            col_mapping[col] = "Degradation_Percent"
            
    df = df_raw.rename(columns=col_mapping).copy()
    
    # Process Dye Conc
    if "Dye_Conc_Raw" in df.columns:
        df["Dye_Conc_M"] = df["Dye_Conc_Raw"].apply(parse_dye_concentration)
        df["Dye_Conc_Display"] = df["Dye_Conc_M"].map(FEATURE_METADATA["Dye_Conc_M"]["exp_display_map"])
    
    # Ensure numeric types
    for col in INPUT_FEATURES + [TARGET_COLUMN]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
    return df

def inspect_dataset(df):
    """
    Produces comprehensive summary statistics for the frontend and reporting.
    """
    numeric_cols = [c for c in INPUT_FEATURES + [TARGET_COLUMN] if c in df.columns]
    desc = df[numeric_cols].describe().round(4).to_dict()
    
    # Format head for web table display with scientific notation string
    preview_df = df.copy()
    if "Dye_Conc_Display" in preview_df.columns:
        preview_df["Dye_Conc_M"] = preview_df["Dye_Conc_Display"]
    
    display_cols = ["pH", "Temperature_C", "Time_min", "Al2O3_mg", "Dye_Conc_M", "Degradation_Percent"]
    display_cols = [c for c in display_cols if c in preview_df.columns]
    
    head_records = preview_df[display_cols].head(15).to_dict(orient="records")
    
    summary = {
        "shape": list(df.shape),
        "columns": display_cols,
        "null_counts": df[display_cols].isnull().sum().to_dict(),
        "duplicate_count": int(df.duplicated(subset=INPUT_FEATURES).sum()),
        "describe": desc,
        "head": head_records,
        "feature_metadata": FEATURE_METADATA,
        "target_metadata": TARGET_METADATA
    }
    return summary

def get_train_test_data(df, test_size=0.20, random_state=42):
    """
    Extracts strictly defined input features and target degradation %, splitting into train and test sets.
    """
    X = df[INPUT_FEATURES].copy()
    y = df[TARGET_COLUMN].copy()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test, INPUT_FEATURES

def get_cross_validation_splits(n_splits=5, random_state=42):
    """Returns standard 5-fold cross-validation object with shuffling."""
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
