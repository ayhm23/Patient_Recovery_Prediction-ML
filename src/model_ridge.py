# src/model_ridge.py
"""
Ridge model pipeline (RidgeCV). Uses shared feature engineering.
Saves: models/ridge_pipeline_*.joblib, output/submission_ridge_*.csv, output/oof_ridge_*.csv, output/coefficients_ridge_*.csv
"""
import os, time, math, joblib
from datetime import datetime
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import mean_squared_error

from feature_engineering import fe, get_feature_lists
from utils import make_preprocessor, safe_to_csv, get_feature_names_after_preproc

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTDIR = "./output"
MODELDIR = "./models"
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(MODELDIR, exist_ok=True)

def main():
    train = pd.read_csv("./data/train.csv")
    test = pd.read_csv("./data/test.csv")
    train = fe(train)
    test = fe(test)

    numerical_features, categorical_features = get_feature_lists()
    X = train.drop(['Id','Recovery Index'], axis=1)
    y = train['Recovery Index']
    X_test = test.drop(['Id'], axis=1)

    preproc = make_preprocessor(numerical_features, categorical_features)
    # RidgeCV will select best alpha from this list
    alphas = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
    ridge = RidgeCV(alphas=alphas, cv=5, scoring='neg_mean_squared_error')
    pipeline = Pipeline([('preproc', preproc), ('ridge', ridge)])

    # CV splits (stratify on binned target)
    y_binned = pd.qcut(y, q=5, labels=False, duplicates='drop')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("Fitting RidgeCV pipeline...")
    t0 = time.time()
    pipeline.fit(X, y)
    t1 = time.time()
    print(f"Fitted on full train in {t1-t0:.1f}s. chosen alpha: {pipeline.named_steps['ridge'].alpha_}")

    # OOF predictions
    oof_preds = cross_val_predict(pipeline, X, y, cv=cv, n_jobs=-1)
    oof_rmse = math.sqrt(mean_squared_error(y, oof_preds))
    print("OOF RMSE (Ridge):", oof_rmse)

    # Save OOF CSV
    oof_df = pd.DataFrame({'Id': train['Id'], 'Actual': y, 'Predicted': oof_preds})
    oof_path = os.path.join(OUTDIR, f"oof_ridge_{TIMESTAMP}.csv")
    safe_to_csv(oof_df, oof_path)

    # Coefficients
    pre = pipeline.named_steps['preproc']
    feat_names = get_feature_names_after_preproc(pre, numerical_features, categorical_features)
    coef_vals = pipeline.named_steps['ridge'].coef_
    coef_df = pd.DataFrame({'feature': feat_names[:len(coef_vals)], 'coef': coef_vals})
    coef_df['abs_coef'] = coef_df['coef'].abs()
    coef_df = coef_df.sort_values('abs_coef', ascending=False).reset_index(drop=True)
    coef_path = os.path.join(OUTDIR, f"coefficients_ridge_{TIMESTAMP}.csv")
    safe_to_csv(coef_df, coef_path)

    # Test predictions & submission
    preds_test = pipeline.predict(X_test)
    preds_test = np.clip(preds_test, 10, 100)
    sub = pd.DataFrame({'Id': test['Id'], 'Recovery Index': preds_test})
    sub_path = os.path.join(OUTDIR, f"submission_ridge_{TIMESTAMP}.csv")
    safe_to_csv(sub, sub_path)

    # Save pipeline
    model_path = os.path.join(MODELDIR, f"ridge_pipeline_{TIMESTAMP}.joblib")
    joblib.dump(pipeline, model_path)
    print("Saved ridge pipeline ->", model_path)
    print("Done.")

if __name__ == "__main__":
    main()
