# Machine Learning Requirements

## Green-Synthesized ZnO Nanoparticle Prediction Project

### 1. Project Objective

Develop machine learning models to predict the physicochemical properties of green-synthesized ZnO nanoparticles from experimental synthesis conditions.

Initial target:

* Particle Size (nm)

Possible future targets:

* Bandgap (eV)
* Photocatalytic degradation (%)
* Other experimentally measured nanoparticle properties

---

## 2. Input Features

The initial dataset should contain the following experimental variables:

* pH
* Temperature
* Reaction Time
* Zn Precursor Concentration
* Plant Extract Concentration

Example:

| Feature                     | Example Unit                    |
| --------------------------- | ------------------------------- |
| pH                          | dimensionless                   |
| Temperature                 | °C                              |
| Reaction Time               | h                               |
| Zn Precursor Concentration  | mol/L or mM                     |
| Plant Extract Concentration | clearly defined consistent unit |

### Target Variable

Initial target:

* Particle Size (nm)

---

## 3. Dataset Requirements

The dataset should be stored in Excel or CSV format.

Recommended filename:

`zn_particle.xlsx`

Each row should represent one independent experiment.

Each column should represent one variable.

Example:

```text
pH,Temperature,Time,Zn_Concentration,Plant_Extract,Particle_Size
6.0,60,2.0,0.10,10,28
6.5,65,2.5,0.10,12,25
7.0,70,3.0,0.15,15,22
```

### Data quality requirements

* Consistent units
* Correct numerical values
* No accidental duplicate experiments
* Missing values identified
* Outliers investigated
* Experimental conditions clearly documented
* Target measurements obtained using a consistent measurement procedure where possible

---

## 4. Python Environment

Recommended Python version:

`Python 3.11+`

Recommended environment:

`Anaconda` or `Miniconda`

A virtual environment is recommended so that project packages remain isolated.

Example environment name:

`nanoparticle-ml`

---

## 5. Required Python Libraries

### Data handling

```text
pandas
openpyxl
numpy
```

Purpose:

* Read Excel files
* Organize datasets
* Perform numerical operations
* Inspect and clean data

---

### Visualization

```text
matplotlib
seaborn
```

Purpose:

* Scatter plots
* Histograms
* Correlation plots
* Prediction vs actual plots
* Residual plots
* Heatmaps

---

### Machine Learning

```text
scikit-learn
xgboost
```

Models to be studied:

1. Linear Regression
2. Decision Tree
3. Random Forest
4. XGBoost

Possible future models:

* Support Vector Regression (SVR)
* Gradient Boosting
* Artificial Neural Networks

---

### Model interpretation

```text
shap
```

Purpose:

* Explain model predictions
* Investigate feature contributions
* Understand how features influence predictions

SHAP analysis should be interpreted as model explanation, not automatically as proof of chemical causation.

---

## 6. Suggested requirements.txt

The project can use the following basic package list:

```text
numpy
pandas
openpyxl
matplotlib
seaborn
scikit-learn
xgboost
shap
jupyter
```

Packages can later be pinned to exact versions after the working environment has been tested.

---

## 7. Project Directory Structure

Recommended structure:

```text
nanoparticle_ml/
│
├── data/
│   ├── raw/
│   │   └── zn_particle.xlsx
│   │
│   └── processed/
│
├── notebooks/
│   ├── 01_data_inspection.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_linear_regression.ipynb
│   ├── 04_decision_tree.ipynb
│   ├── 05_random_forest.ipynb
│   └── 06_xgboost.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── models.py
│   └── evaluation.py
│
├── results/
│   ├── figures/
│   ├── tables/
│   └── models/
│
├── requirements.txt
└── README.md
```

---

## 8. Machine Learning Workflow

The project should follow this sequence:

```text
Experimental Data
        ↓
Excel / CSV Dataset
        ↓
Data Inspection
        ↓
Data Cleaning
        ↓
Missing Value Check
        ↓
Duplicate Check
        ↓
Outlier Investigation
        ↓
Feature Selection
        ↓
Train/Test Split
        ↓
Model Training
        ↓
Prediction
        ↓
Model Evaluation
        ↓
Model Comparison
        ↓
Feature Interpretation
        ↓
Chemical Interpretation
```

---

## 9. Data Inspection

The first Python analysis should include:

```python
df.head()
df.shape
df.columns
df.info()
df.describe()
df.isnull().sum()
df.duplicated().sum()
```

These commands should be used before model training.

---

## 10. Feature and Target Definition

For the initial project:

```python
X = df[
    [
        "pH",
        "Temperature",
        "Time",
        "Zn_Concentration",
        "Plant_Extract"
    ]
]

y = df["Particle_Size"]
```

Where:

* `X` = features
* `y` = target

---

## 11. Train/Test Split

The dataset should be divided into training and testing data.

Example:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
```

The test dataset must remain unseen during model training.

---

## 12. Models

### Model 1 — Linear Regression

Purpose:

* Establish a simple baseline
* Investigate approximately linear relationships

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
```

---

### Model 2 — Decision Tree

Purpose:

* Capture nonlinear relationships
* Generate interpretable decision rules

```python
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(random_state=42)
```

---

### Model 3 — Random Forest

Purpose:

* Combine multiple decision trees
* Improve robustness compared with a single tree

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
```

---

### Model 4 — XGBoost

Purpose:

* Capture nonlinear relationships
* Sequentially improve predictions
* Provide a strong tree-based model for comparison

```python
from xgboost import XGBRegressor

model = XGBRegressor(
    n_estimators=100,
    random_state=42
)
```

Hyperparameters should be tuned later rather than selecting them solely because they are common defaults.

---

## 13. Model Evaluation

The following metrics should be calculated:

### R²

Higher values generally indicate better predictive agreement with the observed target variation.

```python
from sklearn.metrics import r2_score

r2 = r2_score(y_test, predictions)
```

---

### RMSE

Lower values indicate smaller prediction errors.

```python
from sklearn.metrics import root_mean_squared_error

rmse = root_mean_squared_error(
    y_test,
    predictions
)
```

If the installed scikit-learn version does not provide this function, RMSE can alternatively be calculated from mean squared error.

---

### MAE

Lower values indicate smaller average absolute errors.

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(
    y_test,
    predictions
)
```

---

### MAPE

MAPE can be used when the target values are appropriate for percentage-error calculations.

Care must be taken when actual target values are zero or very close to zero.

---

## 14. Model Comparison

Create a comparison table:

| Model             | R² | RMSE | MAE | MAPE |
| ----------------- | -: | ---: | --: | ---: |
| Linear Regression |  — |    — |   — |    — |
| Decision Tree     |  — |    — |   — |    — |
| Random Forest     |  — |    — |   — |    — |
| XGBoost           |  — |    — |   — |    — |

The best model should be selected based on appropriate validation/testing results, not simply because it is the most advanced algorithm.

---

## 15. Required Visualizations

The project should eventually include:

### 1. Feature distributions

```text
Histogram / KDE
```

### 2. Correlation heatmap

```text
Features ↔ Target
```

### 3. Actual vs Predicted plot

```text
Actual Particle Size
        vs
Predicted Particle Size
```

### 4. Residual plot

```text
Prediction Error
        vs
Predicted Value
```

### 5. Feature importance

Especially for:

* Random Forest
* XGBoost

---

## 16. Model Interpretation

Feature importance can be investigated using:

* Random Forest feature importance
* XGBoost feature importance
* SHAP values

Example:

```text
pH                  High
Temperature         High
Reaction Time       Medium
Zn Concentration    Medium
Plant Extract       Low
```

These results should then be discussed in relation to the underlying chemistry.

Feature importance should not automatically be interpreted as causal influence.

---

## 17. Important Scientific Requirements

The ML model must not use information that would not be available when making the intended prediction.

Avoid data leakage.

For example, if the goal is to predict particle size before characterization, using a particle-size measurement derived from the same experiment as an input feature would be inappropriate.

The scientific question must determine which variables are legitimate features.

---

## 18. Experimental vs ML Interpretation

The project should distinguish between:

### Prediction

> How accurately can the model predict particle size?

and:

### Chemical interpretation

> Why might pH, temperature, precursor concentration, or plant extract concentration be associated with particle size?

ML prediction does not by itself establish a chemical mechanism or causality.

---

## 19. Minimum Research Output

The final project should produce:

1. Clean dataset
2. Data preprocessing workflow
3. Linear Regression model
4. Decision Tree model
5. Random Forest model
6. XGBoost model
7. Evaluation metrics
8. Model comparison table
9. Actual vs predicted plots
10. Feature importance analysis
11. Chemical interpretation
12. Reproducible Python code

---

## 20. Current Dataset Status

The current practice Excel file contains:

* 3 experiments
* 6 columns
* 5 candidate input features
* 1 target variable

Current target:

`Particle size`

The current dataset is suitable for **learning the Python/ML workflow**, but it is not large enough to support reliable research conclusions or meaningful model comparison.

A larger experimental dataset should be collected before treating the results as publishable ML evidence.

---

## 21. Learning Sequence

The project will be developed in the following order:

```text
Part 1
Read Excel Dataset
        ↓
Part 2
Inspect Dataset
        ↓
Part 3
Clean Dataset
        ↓
Part 4
Visualize Dataset
        ↓
Part 5
Train/Test Split
        ↓
Part 6
Linear Regression
        ↓
Part 7
Decision Tree
        ↓
Part 8
Random Forest
        ↓
Part 9
XGBoost
        ↓
Part 10
R² / RMSE / MAE / MAPE
        ↓
Part 11
Feature Importance + SHAP
        ↓
Part 12
Scientific Interpretation
        ↓
Research Paper
```

---

## 22. Final Goal

The final objective is not simply to obtain a high R² value.

The objective is to answer a meaningful chemistry question:

> **Can synthesis conditions be used to reliably predict nanoparticle properties, and can the resulting ML model provide scientifically useful insights into the relationship between synthesis conditions and nanoparticle behavior?**
