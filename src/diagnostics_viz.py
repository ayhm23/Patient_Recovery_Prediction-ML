import os
import glob
import math
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import mean_squared_error

sns.set(style="whitegrid")
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# ---------- Feature engineering (same as your pipeline FE) ----------
def fe(df):
    d = df.copy()
    d['Therapy_x_Health'] = d['Therapy Hours'] * d['Initial Health Score']
    d['Therapy_plus_Health'] = d['Therapy Hours'] + d['Initial Health Score']
    d['Sleep_x_Health'] = d['Average Sleep Hours'] * d['Initial Health Score']
    d['FollowUp_x_Health'] = d['Follow-Up Sessions'] * d['Initial Health Score']
    d['Therapy_x_Sleep'] = d['Therapy Hours'] * d['Average Sleep Hours']
    d['Sleep_plus_Health'] = d['Average Sleep Hours'] + d['Initial Health Score']
    d['Therapy_Health_ratio'] = d['Therapy Hours'] / (d['Initial Health Score'] + 1)
    d['Sleep_Health_ratio'] = d['Average Sleep Hours'] / (d['Initial Health Score'] + 1)
    d['log_Therapy'] = np.log1p(d['Therapy Hours'])
    d['log_Health'] = np.log1p(d['Initial Health Score'])
    return d

def safe_write(df, path):
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        base, ext = os.path.splitext(path)
        alt = f"{base}_{TIMESTAMP}{ext}"
        df.to_csv(alt, index=False)
        return alt

def find_latest_joblib(models_dir="./models"):
    pattern = os.path.join(models_dir, "*.joblib")
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

def get_feature_names_from_preproc(preproc, X):
    # try to obtain feature names from preprocessor if possible
    names = None
    try:
        # get numeric column names from X (fallback)
        numeric_cols = [c for c in X.columns if X[c].dtype.kind in 'biufc']
        pre = preproc
        cat = pre.named_transformers_.get('cat', None)
        if cat is not None:
            try:
                cat_names = list(cat.get_feature_names_out())
            except Exception:
                # fallback to categories_
                cat_names = []
                try:
                    for i, cats in enumerate(cat.categories_):
                        # no category feature names list available here, skip
                        pass
                except Exception:
                    cat_names = []
            # best effort: return numeric_cols + cat_names
            names = numeric_cols + cat_names
    except Exception:
        names = None
    if names is None:
        names = list(X.columns)
    return names

def main():
    models_dir = "./models"
    vis_dir = "./visualizations"
    os.makedirs(vis_dir, exist_ok=True)

    # find latest pipeline
    pipeline_path = find_latest_joblib(models_dir)
    if pipeline_path is None:
        print("No pipeline .joblib found in ./models/. Please run training script first.")
        return
    print("Using pipeline:", pipeline_path)
    pipeline = joblib.load(pipeline_path)

    # load data
    train_path = "./data/train.csv"
    test_path = "./data/test.csv"
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("train.csv or test.csv not found in ./data/. Place them there.")
        return

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # apply feature engineering if necessary
    if 'Therapy_x_Health' not in train.columns:
        print("Applying feature engineering to train/test")
        train = fe(train)
        test = fe(test)
    else:
        print("Engineered features already present - skipping FE")

    # prepare X, y, X_test
    if 'Recovery Index' not in train.columns:
        print("train.csv must contain 'Recovery Index' column.")
        return

    X = train.drop(['Id', 'Recovery Index'], axis=1)
    y = train['Recovery Index']
    X_test = test.drop(['Id'], axis=1)

    # compute OOF predictions
    n_splits = 10
    y_binned = pd.qcut(y, q=5, labels=False, duplicates='drop')
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    print("Computing OOF predictions (this may take a bit)...")
    t0 = time.time()
    oof_preds = cross_val_predict(pipeline, X, y, cv=cv, n_jobs=-1)
    t1 = time.time()
    oof_rmse = math.sqrt(mean_squared_error(y, oof_preds))
    print(f"OOF RMSE: {oof_rmse:.6f}  (time {t1-t0:.1f}s)")

    # save OOF CSV
    oof_df = pd.DataFrame({'Id': train['Id'], 'Actual': y, 'Predicted': oof_preds})
    oof_csv = os.path.join(vis_dir, f"oof_predictions_{TIMESTAMP}.csv")
    safe_write(oof_df, oof_csv)
    print("Saved OOF CSV ->", oof_csv)

    # diagnostic arrays
    residuals = y.values - oof_preds

    # 1) Pred vs Actual
    plt.figure(figsize=(6,6))
    plt.scatter(y, oof_preds, s=8, alpha=0.6)
    mn, mx = min(y.min(), oof_preds.min()), max(y.max(), oof_preds.max())
    plt.plot([mn,mx], [mn,mx], 'r--', linewidth=1)
    plt.xlabel("Actual Recovery Index")
    plt.ylabel("Predicted (OOF)")
    plt.title(f"Predicted vs Actual (OOF)  RMSE={oof_rmse:.4f}")
    p1 = os.path.join(vis_dir, "predicted_vs_actual.png")
    plt.tight_layout()
    plt.savefig(p1, dpi=150)
    plt.close()

    # 2) Residuals vs Predicted
    plt.figure(figsize=(8,5))
    plt.scatter(oof_preds, residuals, s=8, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Residuals vs Predicted (OOF)")
    p2 = os.path.join(vis_dir, "residuals_vs_predicted.png")
    plt.tight_layout()
    plt.savefig(p2, dpi=150)
    plt.close()

    # 3) Residual histogram
    plt.figure(figsize=(6,4))
    sns.histplot(residuals, bins=40, kde=True)
    plt.xlabel("Residual")
    plt.title("Residuals distribution (OOF)")
    p3 = os.path.join(vis_dir, "residual_histogram.png")
    plt.tight_layout()
    plt.savefig(p3, dpi=150)
    plt.close()

    # 4) Coefficients bar (if available)
    coef_plot = None
    try:
        last_step = list(pipeline.named_steps.keys())[-1]
        estimator = pipeline.named_steps[last_step]
        coef_vals = getattr(estimator, "coef_", None)
        if coef_vals is None:
            # maybe a wrapper (e.g., GridSearchCV) — try best_estimator_
            try:
                coef_vals = estimator.best_estimator_.named_steps[list(estimator.best_estimator_.named_steps.keys())[-1]].coef_
            except Exception:
                coef_vals = None
        if coef_vals is not None:
            # try to get feature names from preproc if exists
            feat_names = None
            try:
                preproc = pipeline.named_steps.get('preproc', None)
                if preproc is not None:
                    feat_names = get_feature_names_from_preproc(preproc, X)
            except Exception:
                feat_names = list(X.columns)[:len(coef_vals)]
            if feat_names is None or len(feat_names) != len(coef_vals):
                feat_names = list(X.columns)[:len(coef_vals)]
            coef_df = pd.DataFrame({'feature': feat_names, 'coef': coef_vals})
            coef_df['abs_coef'] = coef_df['coef'].abs()
            coef_df = coef_df.sort_values('abs_coef', ascending=False).reset_index(drop=True)
            topn = min(20, len(coef_df))
            plt.figure(figsize=(8, max(4, topn*0.25)))
            sns.barplot(x='abs_coef', y='feature', data=coef_df.head(topn), orient='h')
            plt.xlabel("Absolute Coefficient")
            plt.title("Top coefficients (absolute)")
            coef_plot = os.path.join(vis_dir, "coefficients_bar.png")
            plt.tight_layout()
            plt.savefig(coef_plot, dpi=150)
            plt.close()

            # save coefficients CSV
            coef_csv = os.path.join(vis_dir, f"coefficients_{TIMESTAMP}.csv")
            coef_df.to_csv(coef_csv, index=False)
            print("Saved coefficients CSV ->", coef_csv)
    except Exception as e:
        print("Could not extract coefficients:", e)

    # 5) Correlation matrix for numeric features
    try:
        numeric_cols = [c for c in X.columns if X[c].dtype.kind in 'biufc']
        corr = train[numeric_cols].corr()
        plt.figure(figsize=(9,7))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0)
        plt.title("Numerical feature correlation matrix")
        corr_path = os.path.join(vis_dir, "correlation_matrix.png")
        plt.tight_layout()
        plt.savefig(corr_path, dpi=150)
        plt.close()
    except Exception as e:
        print("Skipping correlation matrix:", e)

    # 6) RMSE by predicted quantile
    try:
        dfq = pd.DataFrame({'y': y, 'pred': oof_preds})
        dfq['pred_bin'] = pd.qcut(dfq['pred'], q=10, labels=False, duplicates='drop')
        rmse_by_bin = dfq.groupby('pred_bin').apply(lambda g: math.sqrt(((g['y']-g['pred'])**2).mean()))
        rmse_by_bin = rmse_by_bin.reset_index().rename(columns={0:'rmse'})
        rmse_csv = os.path.join(vis_dir, "rmse_by_pred_quantile.csv")
        rmse_by_bin.to_csv(rmse_csv, index=False)
        print("Saved RMSE by predicted quantile ->", rmse_csv)
    except Exception as e:
        print("Skipping rmse_by_pred_quantile:", e)

    # 7) Final test preds (submission saved in visualizations for convenience)
    try:
        preds_test = pipeline.predict(X_test)
        preds_test = np.clip(preds_test, 10, 100)
        sub = pd.DataFrame({'Id': test['Id'], 'Recovery Index': preds_test})
        sub_path = os.path.join(vis_dir, f"submission_from_pipeline_{TIMESTAMP}.csv")
        safe_write(sub, sub_path)
        print("Saved submission ->", sub_path)
    except Exception as e:
        print("Could not produce submission from pipeline:", e)

    # Print summary
    print("\nSaved visualizations to:", os.path.abspath(vis_dir))
    print(f" - predicted_vs_actual: {p1}")
    print(f" - residuals_vs_predicted: {p2}")
    print(f" - residual_histogram: {p3}")
    if coef_plot:
        print(f" - coefficients_bar: {coef_plot}")
    print(" - oof csv:", oof_csv)
    print(" - OOF RMSE:", oof_rmse)

if __name__ == "__main__":
    main()