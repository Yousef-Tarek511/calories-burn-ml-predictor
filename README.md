# calories-burn-ml-predictor
# 🔥 Calories Burnt Prediction (Regression)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

A Machine Learning regression project developed for the **Machine Learning Course** at **Helwan National University — Faculty of Computers and Information Technology (Fall 2026/2027)**.

---

## 📌 Project Overview

This project predicts **hourly calories burned (`Calories`, in kcal)** using real-world wearable sensor signals and physiological measurements collected from FitBit fitness trackers. Rather than relying on a clean synthetic dataset or a rigid formula, the project focuses on end-to-end real-world data engineering: merging multi-granularity sensor logs, removing duplicates and target leakage, handling missing values inside leak-free pipelines, comparing multiple regression algorithms, and tuning hyperparameters.

---

## 📊 Dataset & Data Engineering

The dataset is sourced from the real-world **FitBit Fitness Tracker Data** (`arashnic/fitbit` on Kaggle). Instead of training on a single pre-cleaned table, **6 raw sensor files** were cleaned, aggregated to a unified **hourly granularity**, and merged into `cleaned_dataset.csv`.

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
* **Shape:** `2,532 rows × 11 columns`
* **Features (10):** `StepTotal`, `TotalIntensity`, `AvgHeartRate`, `MaxHeartRate`, `MinHeartRate`, `SleepMinutes`, `WeightKg`, `BMI`, `Hour`, `DayOfWeek`
* **Target (1):** `Calories` (Continuous numerical value, range: `47` – `933 kcal/hour`)

---

## 🧹 Preprocessing Pipeline

To guarantee clean evaluation with **zero Data Leakage**, all statistical transformations are fitted strictly on training folds using `scikit-learn`'s `Pipeline`:

1. **Deduplication:** Detected and removed **525 exact duplicate rows** in `minuteSleep_merged.csv` prior to aggregation.
2. **Redundancy & Sparsity Filtering:** Dropped `AverageIntensity` (perfect linear collinearity with `TotalIntensity`) and `Fat` (`93.9%` missing values).
3. **Feature Engineering:** Extracted `Hour` (0–23) and `DayOfWeek` from timestamps; filled unmatched hourly sleep intervals with `0` (awake hours).
4. **Categorical Encoding:** Applied One-Hot Encoding (`pd.get_dummies(drop_first=True)`) on `DayOfWeek`.
5. **Missing Value Imputation:** Imputed **1,520 real missing values** in `WeightKg` and `BMI` using `SimpleImputer(strategy='median')` fitted on `X_train` only.
6. **Feature Scaling:** Standardized numerical features using `StandardScaler()` inside the `Pipeline`.

---

## 📈 Models & Evaluation Results

All models are evaluated using regression metrics: **$R^2$ Score**, **MAE**, **MSE**, and **RMSE**.

### Current Benchmark (80/20 Train-Test Split, `random_state=0`)
| Model | R² Score | MAE (kcal) | MSE | RMSE (kcal) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Multiple Linear Regression** *(Baseline)* | **0.8711** | **19.35** | **797.45** | **28.24** | ✅ Done |
| Polynomial Regression | — | — | — | — | ⏳ Planned |
| Ridge / Lasso Regression | — | — | — | — | ⏳ Planned |
| Support Vector Regression (SVR) | — | — | — | — | ⏳ Planned |
| Decision Tree Regressor | — | — | — | — | ⏳ Planned |
| Random Forest Regressor | — | — | — | — | ⏳ Planned |
| Gradient Boosting / XGBoost *(Bonus)* | — | — | — | — | ⏳ Planned |

### Baseline Stability Across Split Ratios (Multiple Linear Regression)
| Test Size | R² Score | MAE (kcal) | MSE | RMSE (kcal) |
| :---: | :---: | :---: | :---: | :---: |
| **10%** | `0.8536` | `21.01` | `1011.08` | `31.80` |
| **20%** | **`0.8711`** | `19.35` | `797.45` | `28.24` |
| **30%** | `0.8616` | `19.30` | `787.63` | `28.06` |
| **40%** | `0.8618` | `19.80` | `829.29` | `28.80` |

---

## 📁 Repository Structure

```text
├── data/
│   └── cleaned_dataset.csv                # Pre-merged hourly dataset (2,532 × 11)
├── notebooks/
│   ├── 01_multiple_linear_regression.ipynb # Baseline model & preprocessing pipeline
│   └── ...                                # Additional models & tuning notebooks
├── app/                                   # (Upcoming) Interactive prediction GUI
└── README.md
