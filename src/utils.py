import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from mlflow.models.signature import infer_signature
from sklearn.model_selection import cross_validate, GridSearchCV
import mlflow

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

def mlflow_log(name, run_name, metrics, pipeline, signature):
    with mlflow.start_run(run_name=run_name):
        mlflow.log_metrics(metrics)
        mlflow.log_params(pipeline.get_params())
        mlflow.sklearn.log_model(sk_model=pipeline, name=name, signature=signature)

def cross_val(X_train, y_train, pipelines, scoring, cv):
    rows = []

    for name, pipe in pipelines.items():
        cvres = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)
        row = {"model": name}
        row["pipeline"] = pipe
        for k, v in scoring.items():
            row[f"mean_{k}"] = np.mean(cvres[f"test_{k}"])
            row[f"std_{k}"] = np.std(cvres[f"test_{k}"])
        pipe.fit(X_train, y_train)
        signature = infer_signature(X_train, pipe.predict(X_train))
        row["signature"] = signature
        rows.append(row)

    return rows

def gs_cross_val(pipelines, X_train, y_train, scoring, cv_outer):
    rows = []
    for name, rest in pipelines.items():
        pipeline = rest["pipeline"]
        param_grid = rest["param_grid"]
        for k, v in scoring.items():
            gs = GridSearchCV(pipeline, param_grid=param_grid, cv=5, scoring=v, n_jobs=-1)
            scores = []
            for train_idx, test_idx in cv_outer.split(X_train, y_train):
                gs.fit(X_train.iloc[train_idx], pd.Series(y_train).iloc[train_idx])
                X_test = X_train.iloc[test_idx]
                y_test = pd.Series(y_train).iloc[test_idx]
                best_model = gs.best_estimator_
                scores.append(best_model.score(X_test, y_test))

            row = {"name": f"{name}_{k}_GSCV"}
            pipeline = gs.best_estimator_
            row["pipeline"] = pipeline
            row[f"mean_{k}"] = float(np.mean(scores))
            row[f"std_{k}"] = float(np.std(scores))
            row["params"] = gs.best_params_
            pipeline.fit(X_train, y_train)
            signature = infer_signature(X_train, pipeline.predict(X_train))
            row["signature"] = signature
            rows.append(row)

    return rows

def parse_cv_results(rows, mlflow_tracking):
    for row in rows:
        name = row["model"]
        pipeline = row["pipeline"]
        signature = row["signature"]
        metrics = {k: v for k,v in row.items() if k not in ["model", "pipeline", "signature"]}
        
        if mlflow_tracking:
            mlflow_log(name, f"cv-{name}", metrics, pipeline, signature)

        else:
            df = pd.DataFrame(rows)
            res = df.drop(columns=["pipeline", "signature"])
            print("mlflow tracking disabled")
            print(res)

def parse_gscv_results(rows, mlflow_tracking):
    for row in rows:
        name = row["name"]
        pipeline = row["pipeline"]
        params = row["params"]
        signature = row["signature"]
        metrics = {k: v for k,v in row.items() if k not in ["name", "pipeline", "params", "signature"]}

        if mlflow_tracking:
            mlflow_log(name, f"cv-{name}", metrics, pipeline, signature)

        else:
            print("mlfow tracking diabled")
            df = pd.DataFrame(rows)
            res = df.drop(columns=["pipeline", "params"])
            print(res)

