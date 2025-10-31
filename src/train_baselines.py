import numpy as np
import pandas as pd
import mlflow
import sklearn
import data
from features import lr_prep_stdscaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, get_scorer_names
from utils import is_classification_target, make_cv
from sklearn.dummy import DummyRegressor
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE
from sklearn.svm import LinearSVC, LinearSVR

random_state = 1
mlflow.set_tracking_uri("http://127.0.0.1:8080")
baseline_experiment = mlflow.set_experiment("Baseline_Models")
# run in cs4120-project directory mlflow server --host 127.0.0.1 --port 8080 --artifacts-destination ./models

def train_linear_regression(random_state: int):
    d = data.Data()
    preprocessor = lr_prep_stdscaler

    X_train, X_test, y_train, y_test = d.train_test_split(test_ratio=0.4, random_state=random_state)
    pipelines = {}

    
    pipelines["Dummy mean"] = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", DummyRegressor(strategy="mean"))
    ])

    pipelines["Dummy median"] = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", DummyRegressor(strategy="median"))
    ])

    pipelines["LinearRegression"] = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    pipelines["LinearRegressionSelectKBest"] = Pipeline([
        ("preprocessor", preprocessor),
        ("select", SelectKBest(score_func=f_regression, k=min(20, X_train.shape[1]))),
        ("regressor", LinearRegression())
    ])

    pipelines["LinearRegressionRFELinearSVR"] = Pipeline([
        ("preprocessor", preprocessor),
        ("rfe", RFE(estimator=LinearSVR(max_iter=8000), n_features_to_select=min(20, X_train.shape[1] // 2))),
        ("regressor", LinearRegression())
    ])


    scoring = {"r2": "r2", "mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"}

    cv = make_cv(y_train, n_splits=5, random_state=random_state)

    rows = []
    for name, est in pipelines.items():
        cvres = cross_validate(est, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)
        row = {"model": name}
        for k, v in scoring.items():
            row[f"mean_{k}"] = np.mean(cvres[f"test_{k}"])
            row[f"std_{k}"] = np.std(cvres[f"test_{k}"])
        rows.append(row)

    # metric_key = list(scoring.keys())[0]
    # baseline_results = (
    #     pd.DataFrame(rows)
    #     .sort_values(by=[f"mean_{metric_key}"], ascending=False)
    #     .reset_index(drop=True)
    # )
    #
    # print(baseline_results)

    for row in rows:
        name = row["model"]
        print(f"{name}======================================")
        model = pipelines[name]
        metrics = {k: v for k,v in row.items() if k != "model"}
        
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(sk_model=model, input_example=X_train.iloc[[1]], name=name)

if __name__ == '__main__':
    train_linear_regression(random_state)
