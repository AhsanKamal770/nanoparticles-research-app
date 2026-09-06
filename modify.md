# Machine Learning Requirements
## Green-Synthesized Nanoparticle Research Project

---

## 1. Project Objective

Develop machine learning models to study and predict the relationship between:

1. Green synthesis conditions
2. Nanoparticle physicochemical characteristics
3. Functional/performance properties

The project will combine experimental chemistry data with machine learning.

---

# 2. Research Framework

The overall research workflow is:

Green Synthesis
        ↓
Synthesis Parameters
        ↓
Nanoparticle Formation
        ↓
Characterization
        ↓
Physicochemical Properties
        ↓
Functional Performance

Machine learning will be used at different stages of this workflow.

---

# 3. Synthesis Parameters

The initial input variables may include:

- pH
- Temperature
- Reaction time
- Metal precursor concentration
- Plant extract concentration
- Plant extract type
- Extract preparation conditions
- Calcination temperature
- Calcination time
- Other experimentally controlled synthesis parameters

Units must be clearly defined and kept consistent.

---

# 4. Characterization Data

Characterization data should be incorporated into the dataset when experimentally available.

## 4.1 UV-Visible Spectroscopy

Possible variables:

- Absorption maximum (λmax)
- Absorbance
- Optical band gap (eV)
- Other experimentally derived optical parameters

Potential use:

- Characterization feature
- Prediction target

Example:

Synthesis conditions
        ↓
ML model
        ↓
Optical band gap

---

# 4.2 FTIR

Possible variables:

- Important peak positions (cm⁻¹)
- Peak intensities
- Selected functional-group indicators
- Spectral descriptors

Examples of potentially relevant regions/features:

- O–H
- C=O
- C–O
- Metal–oxygen region

The exact variables should be determined from the actual FTIR data.

FTIR spectra can be used in two ways:

### Option A — Extract selected spectral features

Example:

| OH peak | C=O peak | C-O peak | Zn-O peak |
|---|---|---|---|

### Option B — Use the complete spectrum

Wavenumber and intensity values can be transformed into numerical features using suitable preprocessing or dimensionality-reduction methods.

---

# 4.3 XRD

Possible variables:

- 2θ peak positions
- Peak intensity
- FWHM
- Crystallite size
- d-spacing
- Lattice parameters
- Phase information
- Crystallinity-related descriptors

For example:

| 2θ | FWHM | Crystallite size |
|---|---|---|

Crystallite size may be calculated from XRD data using an appropriate established method.

Possible ML target:

Crystallite size

Possible ML features:

- pH
- Temperature
- Reaction time
- precursor concentration
- plant extract concentration
- calcination conditions

---

# 4.4 SEM / TEM

Possible variables:

- Average particle size
- Particle-size distribution
- Morphology descriptors
- Shape descriptors
- Aspect ratio
- Agglomeration-related descriptors

If images are available, image-based machine learning can be considered later.

For the initial project, numerical descriptors should be preferred.

Example:

| Average size | Aspect ratio | Morphology category |
|---|---|---|

---

# 4.5 DLS

Possible variables:

- Hydrodynamic particle size
- Size distribution
- Polydispersity index (PDI)

These should be kept separate from SEM/TEM particle size because different characterization techniques measure different physical quantities.

Example:

| SEM size | DLS size | PDI |
|---:|---:|---:|

---

# 4.6 Zeta Potential

Possible variable:

- Zeta potential (mV)

Potential applications:

- Prediction from synthesis conditions
- Prediction of colloidal stability-related behavior
- Feature for predicting functional performance

---

# 4.7 Other Characterization

Additional experimentally measured properties can be added, depending on the nanoparticle system:

- BET surface area
- Pore volume
- Pore size
- Elemental composition
- EDS/EDX descriptors
- Thermal stability
- Surface chemistry
- Other relevant physicochemical properties

Only experimentally measured variables should be included.

---

# 5. Functional / Performance Data

Possible target variables include:

## Photocatalytic performance

- Degradation percentage
- Reaction rate constant
- Removal efficiency
- Reusability performance

## Antibacterial performance

- Zone of inhibition
- Percentage inhibition
- MIC or other experimentally measured activity indicators

## Other applications

Performance variables should be selected according to the actual research objective.

---

# 6. Example Complete Dataset

A possible dataset structure is:

| pH | Temp | Time | Zn Conc | Extract Conc | XRD Size | UV-Vis Bandgap | DLS Size | PDI | Zeta Potential | SEM Size | Degradation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 60 | 2 | 0.10 | 10 | 24 | 3.10 | 35 | 0.28 | -18 | 28 | 70 |
| 6.5 | 65 | 2.5 | 0.10 | 12 | 22 | 3.15 | 31 | 0.24 | -22 | 25 | 76 |
| 7 | 70 | 3 | 0.15 | 15 | 20 | 3.20 | 27 | 0.20 | -27 | 22 | 83 |

IMPORTANT:

The values above are ONLY an example of dataset structure.

They must NOT be used as real experimental data.

---

# 7. Possible ML Research Questions

The project can contain several separate research questions.

## Research Question 1

Can synthesis conditions predict nanoparticle particle size?

Features:

- pH
- Temperature
- Reaction time
- Precursor concentration
- Plant extract concentration

Target:

- Particle size

---

## Research Question 2

Can synthesis conditions predict crystallite size?

Target:

- XRD crystallite size

---

## Research Question 3

Can synthesis conditions predict optical band gap?

Target:

- Band gap

---

## Research Question 4

Can synthesis and characterization data predict photocatalytic activity?

Features:

- Synthesis parameters
- Particle size
- Crystallite size
- Band gap
- Zeta potential
- Other valid characterization descriptors

Target:

- Photocatalytic degradation or other selected performance metric

---

# 8. Important Prediction-Time Rule

Before selecting a variable as an ML feature, ask:

"Would this information be available at the time I intend to make the prediction?"

For example:

If the objective is:

Predict particle size BEFORE characterization

then:

XRD particle size cannot be used as an input feature.

However:

If the objective is:

Predict photocatalytic performance AFTER characterization

then experimentally measured particle size, band gap, zeta potential, etc. may potentially be used as features.

The scientific question determines the feature set.

---

# 9. Avoid Data Leakage

Data leakage must be carefully controlled.

Examples of problematic leakage:

- Using the target measurement itself as a feature
- Using information derived from the target
- Using test-set information during model training
- Preprocessing the entire dataset before splitting when that preprocessing learns from the data

All preprocessing that learns parameters from data should be fitted using the training data only.

---

# 10. ML Models

The initial model comparison will include:

1. Linear Regression
2. Decision Tree
3. Random Forest
4. XGBoost

Later models may include:

- Support Vector Regression
- Gradient Boosting
- Artificial Neural Networks

The goal is comparison, not assuming that one algorithm is automatically best.

---

# 11. Spectroscopic Data Extension

For FTIR/UV-Vis and other spectra, two approaches can be considered.

## Approach 1 — Extract meaningful descriptors

Example:

- Peak position
- Peak intensity
- Peak area
- FWHM

Advantages:

- Easier to interpret
- Smaller dataset
- Easier for a beginner

## Approach 2 — Use the complete spectrum

Example:

```text
Wavenumber  →  Intensity
400         →  0.12
401         →  0.14
402         →  0.15
...
4000        →  0.05