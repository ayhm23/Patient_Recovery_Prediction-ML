# submit_elasticnet_only.py
import os, math, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from distutils.version import LooseVersion
from datetime import datetime
import joblib
import sklearn

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error

# Config
RND = 42
OUTDIR = "./output"   # current folder
MODEL_OUTDIR = "./models"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

def safe_to_csv(df, path):
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        base, ext = os.path.splitext(path)
        alt = f"{base}_{TIMESTAMP}{ext}"
        df.to_csv(alt, index=False)
        return alt

print("sklearn version:", sklearn.__version__)
print("Output folder:", os.path.abspath(OUTDIR))

# OneHotEncoder compatibility
if LooseVersion(sklearn.__version__) >= LooseVersion("1.2"):
    cat_ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
else:
    cat_ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

# --- Load and feature-engineer ---
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

def fe(df):
    df = df.copy()
    df['Therapy_x_Health'] = df['Therapy Hours'] * df['Initial Health Score']
    df['Therapy_plus_Health'] = df['Therapy Hours'] + df['Initial Health Score']
    df['Sleep_x_Health'] = df['Average Sleep Hours'] * df['Initial Health Score']
    df['FollowUp_x_Health'] = df['Follow-Up Sessions'] * df['Initial Health Score']
    df['Therapy_x_Sleep'] = df['Therapy Hours'] * df['Average Sleep Hours']
    df['Sleep_plus_Health'] = df['Average Sleep Hours'] + df['Initial Health Score']
    df['Therapy_Health_ratio'] = df['Therapy Hours'] / (df['Initial Health Score'] + 1)
    df['Sleep_Health_ratio'] = df['Average Sleep Hours'] / (df['Initial Health Score'] + 1)
    df['log_Therapy'] = np.log1p(df['Therapy Hours'])
    df['log_Health'] = np.log1p(df['Initial Health Score'])
    return df

train = fe(train)
test = fe(test)

# features
numerical_features = [
    'Therapy Hours', 'Initial Health Score', 'Average Sleep Hours', 'Follow-Up Sessions',
    'Therapy_x_Health','Therapy_plus_Health','Sleep_x_Health','FollowUp_x_Health',
    'Therapy_x_Sleep','Sleep_plus_Health','Therapy_Health_ratio','Sleep_Health_ratio',
    'log_Therapy','log_Health'
]
categorical_features = ['Lifestyle Activities']

X = train.drop(['Id','Recovery Index'], axis=1)
y = train['Recovery Index']
X_test = test.drop(['Id'], axis=1)

print("Train shape:", X.shape, "Test shape:", X_test.shape)
print("Numeric features:", numerical_features)
print("Categorical features:", categorical_features)

# Preprocessor: scale numeric, OHE categorical
preproc = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', cat_ohe, categorical_features)
])

# CV: Stratified on binned y (5 bins) for stability
y_binned = pd.qcut(y, q=5, labels=False, duplicates='drop')
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RND)

print("\nFitting ElasticNetCV (this may take a moment)...")
enet_cv = ElasticNetCV(alphas=np.logspace(-5, 1, 60),
                       l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.85, 0.9],
                       cv=cv, max_iter=5000, n_jobs=-1, random_state=RND)

pipeline = Pipeline([('preproc', preproc), ('enet', enet_cv)])
t0 = time.time()
pipeline.fit(X, y)
t1 = time.time()
alpha = pipeline.named_steps['enet'].alpha_
l1 = pipeline.named_steps['enet'].l1_ratio_
print(f"Fitted ElasticNetCV in {t1-t0:.1f}s  alpha={alpha:.6g}  l1_ratio={l1}")

# OOF predictions (unbiased)
print("\nComputing OOF predictions (cross_val_predict)...")
oof_preds = cross_val_predict(pipeline, X, y, cv=cv, n_jobs=-1)
oof_rmse = math.sqrt(mean_squared_error(y, oof_preds))
print("OOF RMSE (ElasticNetCV):", oof_rmse)

# Inspect coefficients (feature names)
pre = pipeline.named_steps['preproc']
# build categorical names after OHE
try:
    cat_names = list(pre.named_transformers_['cat'].get_feature_names_out(categorical_features))
except Exception:
    cat_encoder = pre.named_transformers_['cat']
    cat_names = []
    for i, cats in enumerate(cat_encoder.categories_):
        cat = categorical_features[i]
        cat_names += [f"{cat}_{c}" for c in cats]

feat_names = numerical_features + cat_names
coefs = pipeline.named_steps['enet'].coef_
coef_df = pd.DataFrame({'feature': feat_names, 'coef': coefs})
coef_df['abs_coef'] = coef_df['coef'].abs()
coef_df = coef_df.sort_values('abs_coef', ascending=False).reset_index(drop=True)
print("\nTop coefficients:")
print(coef_df.head(20).to_string(index=False))
coef_df.to_csv(os.path.join(OUTDIR, "elasticnet_coefficients.csv"), index=False)
print("Saved coefficients -> elasticnet_coefficients.csv")

# Fit final model on full train (pipeline already fitted, but ensure)
pipeline.fit(X, y)

# Predict test and create submission
preds_test = pipeline.predict(X_test)
# clip to plausible target range if known (your earlier data had min 10, max 100)
preds_test = np.clip(preds_test, 10, 100)

submission = pd.DataFrame({'Id': test['Id'], 'Recovery Index': preds_test})
submission_path = safe_to_csv = None
try:
    submission_path = os.path.join(OUTDIR, "submission_elasticnet_final.csv")
    submission.to_csv(submission_path, index=False)
except PermissionError:
    alt = os.path.join(OUTDIR, f"submission_elasticnet_final_{TIMESTAMP}.csv")
    submission.to_csv(alt, index=False)
    submission_path = alt

print("\nSaved submission ->", submission_path)

# Save pipeline for later use
model_path = os.path.join(MODEL_OUTDIR, f"elasticnet_pipeline_{TIMESTAMP}.joblib")
joblib.dump(pipeline, model_path)
print("Saved pipeline ->", model_path)

print("\nDone. Use submission_elasticnet_final.csv for submission.")
