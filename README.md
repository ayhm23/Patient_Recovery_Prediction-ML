# 🧠 Patient Recovery Prediction - ML

Predicting patient recovery outcomes using machine learning.

## 📘 Project Overview

This project develops and evaluates machine learning models to estimate a patient’s recovery progress based on treatment and lifestyle factors.  
It is a **supervised regression** problem where the target variable, **Recovery Index**, is a continuous score ranging from **10 to 100**, representing the patient’s overall recovery.

### Key Objectives
- Perform Exploratory Data Analysis (EDA) and preprocessing.
- Engineer features to capture non-linear interactions and relationships.
- Train and compare multiple regression models.
- Optimize the final model using **10-fold cross-validation**.
- Generate predictions suitable for Kaggle submission.
- Ensure full reproducibility using Scikit-learn Pipelines.

---

## 🧾 Dataset Description

**Dataset size:**  
- 10,000 patient records  
  - 8,000 for training  
  - 2,000 for testing

**Features:**
| Feature | Description |
|----------|-------------|
| Therapy Hours | Total number of hours spent in therapy sessions |
| Initial Health Score | Health assessment score recorded at first check-up |
| Lifestyle Activities | Whether patient engaged in healthy lifestyle activities (Yes/No) |
| Average Sleep Hours | Average hours of sleep per day |
| Follow-Up Sessions | Number of follow-up sessions attended |

**Target Variable:**  
- **Recovery Index (10–100)** — Continuous measure of overall recovery progress.  

---

## 🧹 EDA and Preprocessing

### Data Cleaning
- No missing or duplicate values found.
- `Lifestyle Activities` converted from categorical (Yes/No) to numeric (OneHotEncoder).
- `Id` column removed as it had no predictive value.

### Feature Scaling
- Applied `StandardScaler` to numerical features.
- Used `ColumnTransformer` inside `Pipeline` to prevent data leakage.

### Feature Engineering
Two phases of engineered features:

1. **Initial (11 features total):**
   - Interaction & summation terms like  
     `Therapy_x_Health`, `Therapy_plus_Health`, `Sleep_x_Health`, `Sleep_plus_Health`, etc.

2. **Advanced (15 features total):**
   - Added ratio & log features:  
     `Therapy_Health_ratio`, `Sleep_Health_ratio`, `log_Therapy`, `log_Health`.

These captured proportional and non-linear effects between health and therapy metrics.

---

## 🤖 Models Used

### Stage 1: Initial Model Comparison (11 features)
Models tested:
- **Linear models:** Linear Regression, Ridge, Lasso, ElasticNet, Bayesian Ridge  
- **Tree ensembles:** Random Forest, Decision Tree  
- **Boosting methods:** XGBoost, Gradient Boosting, AdaBoost, LightGBM  
- **Others:** SVR (RBF & Linear), KNN

**Best model:**  
- **Ridge Regression (α≈22.5)**  
- **Cross-val RMSE ≈ 2.0435, Kaggle Score = 2.066**

### Stage 2: Refined Model Selection (15 features)
Focused on:
- **Ridge Regression (α=1.0)**
- **ElasticNetCV (α≈0.0028, l1_ratio=0.9)**  
  Used automatic 10-fold cross-validation.

---

## 🔧 Hyperparameter Tuning

- **Ridge / Lasso / ElasticNet:** GridSearchCV & ElasticNetCV  
- **Random Forest:** Tuned `n_estimators`, `max_depth`, `min_samples_leaf`  
- **Boosting Models:** Tuned `n_estimators`, `learning_rate`, `max_depth`

**Scoring Metric:** Negative Root Mean Squared Error (`neg_root_mean_squared_error`)

---

## 📊 Results and Performance

| Stage | Model | CV / OOF RMSE | Kaggle Score |
|--------|--------|---------------|---------------|
| Initial (11 features) | Ridge | ~2.044 | 2.066 |
| Refined (15 features) | Ridge | ~2.0444 | 1.982 |
| Final (15 features) | ElasticNet | ~2.0449 (OOF) | **1.980** |

**Final Model:** ElasticNet (α=0.00276, l1_ratio=0.9)  
- Trained on full 15-feature dataset  
- Predictions clipped to [10, 100]  
- Saved as `.joblib` for deployment

---

## 🔍 Feature Importance

| Feature | Coefficient |
|----------|-------------|
| Therapy_plus_Health | 7.1272 |
| Initial Health Score | 5.2301 |
| Therapy Hours | 4.1095 |
| Sleep_plus_Health | 3.5845 |
| log_Health | 1.6685 |
| Therapy_x_Health | 1.2643 |
| Therapy_Health_ratio | 1.1996 |
| Follow-Up Sessions | 0.5463 |
| Lifestyle Activities_Yes | 0.2991 |
| Lifestyle Activities_No | -0.2991 |

*(Several less-informative features like `Average Sleep Hours`, `FollowUp_x_Health`, `log_Therapy` were automatically zeroed out by ElasticNet.)*

---

## 💻 Usage Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/ayhm23/Patient_Recovery_Prediction-ML.git
   cd Patient_Recovery_Prediction-ML

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
Run the pipeline

bash
Copy code
python src/0_data_loading.py
python src/1_eda.py
python src/2_preprocessing.py
python src/3_model_training.py
python src/4_model_evaluation.py
Saved outputs

Trained models → /models

Visualizations → /visualizations

Reports → /output

🧩 Reproducibility
Random seed: random_state=42

Cross-validation: StratifiedKFold (10 splits, binned on y)

Libraries:
pandas, numpy, scikit-learn, matplotlib, xgboost, joblib

All preprocessing handled in Scikit-learn Pipeline to avoid data leakage.

📈 Key Takeaways
Linear models outperformed complex ensembles for this dataset.

Feature engineering (ratios, logs) significantly boosted performance.

ElasticNet provided the optimal mix of interpretability, stability, and regularization.

Final Kaggle score 1.980, RMSE ≈ 2.04 (OOF).

🧑‍💻 Contributors
Archit Jaju (IMT2023128)

Sanyam Verma (IMT2023040)

📜 License & AcknowledgementsAC
License: Add license (e.g., MIT)

Acknowledgements: Dataset providers, Kaggle platform, Scikit-learn, XGBoost