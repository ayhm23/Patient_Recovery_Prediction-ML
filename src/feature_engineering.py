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
