# src/utils.py
import os
from distutils.version import LooseVersion
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

def safe_to_csv(df, path, timestamp=None):
    """Write df to path; on PermissionError append timestamp to filename."""
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        base, ext = os.path.splitext(path)
        if timestamp is None:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        alt = f"{base}_{timestamp}{ext}"
        df.to_csv(alt, index=False)
        return alt

def make_preprocessor(numerical_features, categorical_features, sklearn_version=None):
    """
    Build ColumnTransformer: StandardScaler for numeric, OneHotEncoder for categorical.
    Returns fitted transformer? No — returns transformer instance (to be fitted in pipeline).
    """
    if sklearn_version is None:
        # Try to import sklearn and get its version
        try:
            import sklearn
            sklearn_version = sklearn.__version__
        except Exception:
            sklearn_version = "1.0"

    if LooseVersion(sklearn_version) >= LooseVersion("1.2"):
        cat_ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        cat_ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preproc = ColumnTransformer([
        ('num', StandardScaler(), numerical_features),
        ('cat', cat_ohe, categorical_features)
    ])
    return preproc

def get_feature_names_after_preproc(preproc, numerical_features, categorical_features):
    """
    Given a fitted ColumnTransformer 'preproc', return list of output feature names
    in the same order as the transformed matrix (approximate).
    Note: this works when the categorical transformer supports get_feature_names_out.
    """
    names = []
    # numeric features first (assumes StandardScaler doesn't change names)
    names.extend(numerical_features)
    # categorical features -> one-hot names
    try:
        cat = preproc.named_transformers_['cat']
        cat_names = list(cat.get_feature_names_out(categorical_features))
    except Exception:
        # fallback: try to read categories_ attribute
        cat_encoder = preproc.named_transformers_['cat']
        cat_names = []
        for i, cats in enumerate(cat_encoder.categories_):
            cat = categorical_features[i]
            cat_names += [f"{cat}_{c}" for c in cats]
    names.extend(cat_names)
    return names
