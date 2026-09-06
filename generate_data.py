import os
import numpy as np
import pandas as pd

def generate_datasets():
    # Ensure directories exist
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/models", exist_ok=True)
    os.makedirs("notebooks", exist_ok=True)

    # 1. Generate 3-sample practice dataset from requirements.md / modify.md
    sample_df = pd.DataFrame([
        {
            "pH": 6.0, "Temperature": 60, "Time": 2.0, "Zn_Concentration": 0.10, "Plant_Extract": 10,
            "XRD_Size": 24, "UV_Bandgap": 3.10, "DLS_Size": 35, "PDI": 0.28, "Zeta_Potential": -18,
            "SEM_Size": 28, "Degradation": 70
        },
        {
            "pH": 6.5, "Temperature": 65, "Time": 2.5, "Zn_Concentration": 0.10, "Plant_Extract": 12,
            "XRD_Size": 22, "UV_Bandgap": 3.15, "DLS_Size": 31, "PDI": 0.24, "Zeta_Potential": -22,
            "SEM_Size": 25, "Degradation": 76
        },
        {
            "pH": 7.0, "Temperature": 70, "Time": 3.0, "Zn_Concentration": 0.15, "Plant_Extract": 15,
            "XRD_Size": 20, "UV_Bandgap": 3.20, "DLS_Size": 27, "PDI": 0.20, "Zeta_Potential": -27,
            "SEM_Size": 22, "Degradation": 83
        }
    ])
    
    sample_path = "data/raw/zn_particle_sample.xlsx"
    sample_df.to_excel(sample_path, index=False)
    print(f"Sample dataset saved to {sample_path}")

    # 2. Generate Realistic Experimental Dataset (150 synthesis runs)
    np.random.seed(42)
    n_samples = 150

    pH = np.round(np.random.uniform(5.5, 10.5, n_samples), 1)
    temp = np.round(np.random.uniform(40, 90, n_samples), 0)
    time = np.round(np.random.uniform(1.0, 5.0, n_samples), 1)
    zn_conc = np.round(np.random.uniform(0.05, 0.30, n_samples), 2)
    plant_extract = np.round(np.random.uniform(5, 25, n_samples), 1)

    # Physical chemistry kinetics relationships:
    # 1. Higher pH and Plant Extract -> increased nucleation -> smaller SEM particle size & XRD crystallite size
    # 2. Higher temp & time -> particle growth -> larger size
    # 3. Higher Zn Conc -> more precursor -> larger size
    base_sem = 45.0 - 1.8 * (pH - 6.0) - 0.75 * (plant_extract - 10.0) + 0.15 * (temp - 50.0) + 1.2 * (time - 2.0) + 15.0 * (zn_conc - 0.1)
    sem_size = np.clip(base_sem + np.random.normal(0, 2.0, n_samples), 12.0, 65.0)
    sem_size = np.round(sem_size, 1)

    # XRD crystallite size is typically slightly smaller than SEM particle size
    xrd_size = np.clip(sem_size * 0.85 + np.random.normal(0, 1.2, n_samples), 8.0, 50.0)
    xrd_size = np.round(xrd_size, 1)

    # UV-Vis Bandgap (quantum confinement: smaller particle size -> higher bandgap eV)
    uv_bandgap = np.clip(3.35 - 0.006 * (sem_size - 20.0) + np.random.normal(0, 0.02, n_samples), 3.00, 3.45)
    uv_bandgap = np.round(uv_bandgap, 2)

    # DLS size (hydrodynamic diameter is larger than SEM size due to solvation layer)
    dls_size = np.clip(sem_size * 1.35 + np.random.normal(0, 3.0, n_samples), 18.0, 90.0)
    dls_size = np.round(dls_size, 1)

    # PDI (Polydispersity index)
    pdi = np.clip(0.35 - 0.008 * plant_extract + np.random.normal(0, 0.03, n_samples), 0.12, 0.45)
    pdi = np.round(pdi, 2)

    # Zeta potential (negative magnitude increases with higher plant capping extract)
    zeta_potential = np.clip(-12.0 - 0.8 * plant_extract - 0.5 * (pH - 6.0) + np.random.normal(0, 2.0, n_samples), -40.0, -8.0)
    zeta_potential = np.round(zeta_potential, 1)

    # Photocatalytic Degradation (%): smaller particle size (higher surface area) & higher bandgap -> higher degradation
    base_deg = 65.0 - 0.6 * (sem_size - 25.0) + 15.0 * (uv_bandgap - 3.1) - 0.3 * (zeta_potential + 20.0) + 0.4 * (plant_extract - 10.0)
    degradation = np.clip(base_deg + np.random.normal(0, 3.0, n_samples), 45.0, 98.0)
    degradation = np.round(degradation, 1)

    df = pd.DataFrame({
        "pH": pH,
        "Temperature": temp,
        "Time": time,
        "Zn_Concentration": zn_conc,
        "Plant_Extract": plant_extract,
        "XRD_Size": xrd_size,
        "UV_Bandgap": uv_bandgap,
        "DLS_Size": dls_size,
        "PDI": pdi,
        "Zeta_Potential": zeta_potential,
        "SEM_Size": sem_size,
        "Degradation": degradation
    })

    main_path = "data/raw/zn_particle.xlsx"
    df.to_excel(main_path, index=False)
    print(f"Main dataset generated with {len(df)} samples at {main_path}")

    # Also save as CSV for quick loading
    csv_path = "data/raw/zn_particle.csv"
    df.to_csv(csv_path, index=False)
    print(f"CSV dataset saved at {csv_path}")

if __name__ == "__main__":
    generate_datasets()
