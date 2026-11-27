# 🔥 Calories Burnt Prediction (Regression)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![Best R2](https://img.shields.io/badge/Best%20R%C2%B2-0.9265%20(Random%20Forest)-brightgreen)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

A Machine Learning regression project developed for the **Machine Learning Course** at **Helwan National University — Faculty of Computers and Information Technology (Fall 2026/2027)**.

---

## 📌 Project Overview

This project predicts **hourly calories burned (`Calories`, in kcal)** using real-world wearable sensor signals and physiological measurements collected from FitBit fitness trackers. Rather than relying on a clean synthetic dataset or a fixed formula, the project focuses on end-to-end real-world data engineering: merging multi-granularity sensor logs, removing duplicates and target leakage, applying leak-free `Pipeline` preprocessing, comparing multiple regression algorithms, and tuning hyperparameters via `GridSearchCV`.

---

## 📊 Dataset & Data Engineering

The dataset is sourced from the real-world **FitBit Fitness Tracker Data** (`arashnic/fitbit` on Kaggle). **6 raw sensor files** were cleaned, aggregated to a unified **hourly granularity**, and merged into `cleaned_dataset.csv`.

### Raw Files Used
| File | Granularity | Purpose in Pipeline |
| :--- | :--- | :--- |
| `hourlyCalories_merged.csv` | Hourly | Core table & continuous target variable (`Calories`) |
| `hourlyIntensities_merged.csv` | Hourly | Activity intensity (`TotalIntensity`) |
| `hourlySteps_merged.csv` | Hourly | Step count per hour (`StepTotal`) |
| `heartrate_seconds_merged.csv` | Seconds (1.15M rows) | Aggregated to hourly `AvgHeartRate`, `MaxHeartRate`, `MinHeartRate` |
| `minuteSleep_merged.csv` | Minutes (198K rows) | Deduplicated & aggregated to hourly `SleepMinutes` |
| `weightLogInfo_merged.csv` | User Logs | User body metrics (`WeightKg`, `BMI`) |

### Deliberate Exclusions (Data Quality & Leakage Prevention)
* **`minuteMETsNarrow_merged.csv`:** Excluded to prevent **Data Leakage** (`METs` has a `0.959` correlation with `Calories` because FitBit derives calorie estimates directly from METs).
* **`minuteCaloriesNarrow_merged.csv` & `minuteStepsNarrow_merged.csv`:** Excluded as redundant (1.44M rows each; already represented accurately at the hourly level).
* **`dailyActivity_merged.csv`:** Excluded due to mismatched temporal granularity (457 daily rows vs. hourly sensor tracking).

### Final Merged Dataset (`cleaned_dataset.csv`)
* **Shape:** `2,532 rows × 11 columns` (`2,532 × 15` features after One-Hot Encoding)
* **Features (10 raw):** `StepTotal`, `TotalIntensity`, `AvgHeartRate`, `MaxHeartRate`, `MinHeartRate`, `SleepMinutes`, `WeightKg`, `BMI`, `Hour`, `DayOfWeek`
* **Target (1):** `Calories` (Continuous numerical value, range: `47` – `933 kcal/hour`, mean: `117.51 kcal/hour`)

---

## 🧹 Preprocessing Pipeline

To guarantee fair model comparison with **zero Data Leakage**, all notebooks use the exact same data split (`test_size=0.2, random_state=0`), unified encoding, and `scikit-learn` `Pipeline` fitted strictly on training folds:

1. **Deduplication:** Detected and removed **525 exact duplicate rows** in `minuteSleep_merged.csv` prior to hourly aggregation.
2. **Redundancy & Sparsity Filtering:** Dropped `AverageIntensity` (perfect collinearity with `TotalIntensity`) and `Fat` (`93.9%` missing values).
3. **Feature Engineering:** Extracted `Hour` (0–23) and `DayOfWeek` from timestamps; filled unmatched hourly sleep intervals with `0` (awake hours).
4. **Unified Categorical Encoding:** Applied One-Hot Encoding on `DayOfWeek` (`pd.get_dummies(..., columns=['DayOfWeek'], drop_first=True, dtype=int)`) across all models to avoid the Dummy Variable Trap, yielding **15 consistent features**.
5. **Missing Value Imputation:** Imputed **1,520 real missing values** in `WeightKg` and `BMI` using `SimpleImputer(strategy='median')` inside each model's `Pipeline`.
6. **Model-Aware Feature Scaling:** Applied `StandardScaler()` inside the `Pipeline` for scale-sensitive models (e.g., Linear Regression, SVR), and intentionally omitted it for tree-based models (Decision Tree, Random Forest) since threshold-based splits are invariant to monotonic feature scaling.

---

## 📈 Models & Evaluation Results

All models are evaluated on the exact same held-out test set (**80/20 split, `random_state=0`, 507 test samples**) using **$R^2$ Score**, **MAE**, **MSE**, and **RMSE**.

### Model Benchmark Comparison
| Model | Train R² | Test R² | MAE (kcal) | MSE | RMSE (kcal) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Multiple Linear Regression** *(Baseline)* | — | `0.8711` | `19.35` | `797.45` | `28.24` | ✅ Done |
| **Decision Tree Regressor** *(Default)* | `1.0000` | `0.8592` | `17.81` | `871.42` | `29.52` | ✅ Done |
| **Decision Tree Regressor** *(Tuned — GridSearchCV)* | `0.9473` | `0.8835` | `17.66` | `721.14` | `26.85` | ✅ Done |
| **Random Forest Regressor** *(Default)* | `0.9887` | `0.9209` | `13.27` | `489.64` | `22.13` | ✅ Done |
| **🏆 Random Forest Regressor** *(Tuned — GridSearchCV)* | **`0.9899`** | **`0.9265`** | **`13.08`** | **`454.84`** | **`21.33`** | ✅ **Best So Far** |
| Polynomial Regression | — | — | — | — | — | ⏳ Planned |
| Ridge / Lasso Regression | — | — | — | — | — | ⏳ Planned |
| Support Vector Regression (SVR) | — | — | — | — | — | ⏳ Planned |
| Gradient Boosting / XGBoost *(Bonus)* | — | — | — | — | — | ⏳ Planned |

---

## ⚙️ Hyperparameter Tuning (`GridSearchCV`, 5-Fold CV)

| Model | Best Hyperparameters Found | Best 5-Fold CV R² | Test R² Gain |
| :--- | :--- | :---: | :---: |
| **Decision Tree Regressor** | `max_depth=7`, `min_samples_split=2`, `min_samples_leaf=2` | `0.8672` | `+0.0243` (`0.8592` → `0.8835`) |
| **Random Forest Regressor** | `n_estimators=300`, `max_features=0.7`, `max_depth=None`, `min_samples_split=2`, `min_samples_leaf=1` | `0.9212` | `+0.0056` (`0.9209` → `0.9265`) |

### Key Modeling Insights
* **Overfitting Resolution in Decision Tree:** The unconstrained default tree grew to depth `29` (`1,626` leaves), memorizing the training set (`Train R² = 1.0000`) while underperforming the linear baseline (`Test R² = 0.8592`). Pruning via `GridSearchCV` (`max_depth=7, min_samples_leaf=2`) cut overfitting and raised `Test R²` to **`0.8835`**.
* **Variance Reduction in Random Forest:** Ensembling `300` trees with `max_features=0.7` captured non-linear physiological patterns (high-intensity workout spikes), reducing RMSE from `28.24` (Linear) and `26.85` (Tuned Tree) down to **`21.33 kcal/hour`** (`Test R² = 0.9265`, `OOB R² = 0.9267`).

---

## 📁 Repository Structure

```text
├── data/
│   └── cleaned_dataset.csv                    # Pre-merged hourly dataset (2,532 × 11)
├── notebooks/
│   ├── 01_multiple_linear_regression.ipynb    # Baseline model & scaling pipeline
│   ├── 02_decision_tree_regression.ipynb      # Default & GridSearchCV-tuned Decision Tree
│   ├── 03_random_forest_regression.ipynb      # Default & GridSearchCV-tuned Random Forest
│   └── ...                                    # Upcoming models (SVR, Polynomial, Boosting)
├── app/                                       # (Upcoming) Interactive prediction GUI
└── README.md
