# src/model_lasso.py

import os, time, math, joblib
from datetime import datetime
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LassoCV
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

    lasso = LassoCV(cv=5, n_jobs=-1, max_iter=10000)
    pipeline = Pipeline([('preproc', preproc), ('lasso', lasso)])

    # CV splits
    y_binned = pd.qcut(y, q=5, labels=False, duplicates='drop')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("Fitting LassoCV pipeline...")
    t0 = time.time()
    pipeline.fit(X, y)
    t1 = time.time()
    print(f"Fitted on full train in {t1-t0:.1f}s. chosen alpha: {pipeline.named_steps['lasso'].alpha_}")

    # OOF predictions
    oof_preds = cross_val_predict(pipeline, X, y, cv=cv.split(X, y_binned), n_jobs=-1)
    oof_rmse = math.sqrt(mean_squared_error(y, oof_preds))
    print("OOF RMSE (Lasso):", oof_rmse)

    # Save OOF CSV
    oof_df = pd.DataFrame({'Id': train['Id'], 'Actual': y, 'Predicted': oof_preds})
    oof_path = os.path.join(OUTDIR, f"oof_lasso.csv")
    safe_to_csv(oof_df, oof_path)

    # Coefficients
    pre = pipeline.named_steps['preproc']
    feat_names = get_feature_names_after_preproc(pre, numerical_features, categorical_features)
    coef_vals = pipeline.named_steps['lasso'].coef_
    coef_df = pd.DataFrame({'feature': feat_names[:len(coef_vals)], 'coef': coef_vals})
    coef_df['abs_coef'] = coef_df['coef'].abs()
    coef_df = coef_df.sort_values('abs_coef', ascending=False).reset_index(drop=True)
    coef_path = os.path.join(OUTDIR, f"coefficients_lasso.csv")
    safe_to_csv(coef_df, coef_path)

    # Test predictions & submission
    preds_test = pipeline.predict(X_test)
    preds_test = np.clip(preds_test, 10, 100)
    sub = pd.DataFrame({'Id': test['Id'], 'Recovery Index': preds_test})
    sub_path = os.path.join(OUTDIR, f"submission_lasso.csv")
    safe_to_csv(sub, sub_path)

    # Save pipeline
    model_path = os.path.join(MODELDIR, f"lasso_pipeline.joblib")
    joblib.dump(pipeline, model_path)
    print("Saved lasso pipeline ->", model_path)
    print("Done.")

if __name__ == "__main__":
    main()
