# src/feature_engineering.py
import numpy as np
import pandas as pd

def fe(df):
    """Apply feature engineering to a copy of df and return it."""
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

def get_feature_lists():
    """Return (numerical_features, categorical_features) consistent with pipelines."""
    numerical_features = [
        'Therapy Hours', 'Initial Health Score', 'Average Sleep Hours', 'Follow-Up Sessions',
        'Therapy_x_Health','Therapy_plus_Health','Sleep_x_Health','FollowUp_x_Health',
        'Therapy_x_Sleep','Sleep_plus_Health','Therapy_Health_ratio','Sleep_Health_ratio',
        'log_Therapy','log_Health'
    ]
    categorical_features = ['Lifestyle Activities']
    return numerical_features, categorical_features

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
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