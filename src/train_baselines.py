import numpy as np
import pandas as pd
import mlflow
import sklearn
import data
from features import lr_preprocessor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, get_scorer_names
from utils import is_classification_target, make_cv
from sklearn.dummy import DummyRegressor

random_state = 1
mlflow.set_tracking_uri("http://127.0.0.1:8080")
baseline_experiment = mlflow.set_experiment("Baseline_Models")
# run in cs4120-project directory mlflow server --host 127.0.0.1 --port 8080 --artifacts-destination ./models

def train_linear_regression(random_state: int):
    d = data.Data()
    preprocessor = lr_preprocessor

    X_train, X_test, y_train, y_test = d.train_test_split(test_ratio=0.4, random_state=random_state)

    linear_regression = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    linear_model = TransformedTargetRegressor(
            regressor=linear_regression,
            transformer=StandardScaler()
    )

    models = {
        "Dummy mean": DummyRegressor(strategy="mean"),
        "Dummy median": DummyRegressor(strategy="median"),
        "LinearRegression": linear_model
    }

    scoring = {"r2": "r2", "mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"}

    cv = make_cv(y_train, n_splits=5, random_state=random_state)

    rows = []
    for name, est in models.items():
        cvres = cross_validate(est, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)
        row = {"model": name}
        for k, v in scoring.items():
            row[f"mean_{k}"] = np.mean(cvres[f"test_{k}"])
            row[f"std_{k}"] = np.std(cvres[f"test_{k}"])
        rows.append(row)

    metric_key = list(scoring.keys())[0]
    baseline_results = (
        pd.DataFrame(rows)
        .sort_values(by=[f"mean_{metric_key}"], ascending=False)
        .reset_index(drop=True)
    )

    print(baseline_results)

    for row in rows:
        name = row["model"]
        metrics = {k: v for k,v in row.items() if k != "model"}
        
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(sk_model=name, input_example=X_train.iloc[[1]], name=name)

if __name__ == '__main__':
    train_linear_regression(random_state)
