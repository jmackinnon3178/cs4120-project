import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def is_classification_target(y):
    # Heuristic: treat as classification if number of unique values is small relative to length
    if y is None:
        raise ValueError("y is not defined. Please define X and y before running the extended sections.")
    try:
        unique = pd.Series(y).dropna().unique()
    except Exception:
        unique = np.unique(y)
    n_unique = len(unique)
    if pd.api.types.is_numeric_dtype(pd.Series(y)) and n_unique > 20:
        return False
    return True

def make_cv(y, n_splits=5, random_state=42):
    if is_classification_target(y):
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
