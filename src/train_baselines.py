import numpy as np
import pandas as pd
import mlflow
import data
from features import lr_prep_stdscaler, grade_to_pass_fail, dt_prep
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, GridSearchCV
from utils import make_cv
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE
from sklearn.svm import LinearSVC, LinearSVR
from mlflow.models.signature import infer_signature

mlflow_tracking = True
random_state = 1

if (mlflow_tracking):
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    baseline_experiment = mlflow.set_experiment("Baseline_Models")
# run in cs4120-project directory mlflow server --host 127.0.0.1 --port 8080 --artifacts-destination ./models

def cv_and_log(X_train, y_train, pipelines, scoring, cv):
    rows = []

    for name, pipe in pipelines.items():
        cvres = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)
        row = {"model": name}
        for k, v in scoring.items():
            row[f"mean_{k}"] = np.mean(cvres[f"test_{k}"])
            row[f"std_{k}"] = np.std(cvres[f"test_{k}"])
        rows.append(row)

    if (mlflow_tracking):
        for row in rows:
            name = row["model"]
            pipeline = pipelines[name]
            metrics = {k: v for k,v in row.items() if k != "model"}
            pipeline.fit(X_train, y_train)
            
            with mlflow.start_run(run_name=f"training-{name}"):
                mlflow.log_metrics(metrics)
                mlflow.log_params(pipeline.get_params())
                signature = infer_signature(X_train, pipeline.predict(X_train))
                mlflow.sklearn.log_model(sk_model=pipeline, name=name, signature=signature)

    else:
        metric_key = list(scoring.keys())[0]
        baseline_results = (
            pd.DataFrame(rows)
            .sort_values(by=[f"mean_{metric_key}"], ascending=False)
            .reset_index(drop=True)
        )
        print("mlflow tracking disabled")
        print(baseline_results)

def gscv_and_log(name, X_train, y_train, pipe, param_grid, scoring, cv_outer):
    rows = []

    for k, v in scoring.items():
        gs = GridSearchCV(pipe, param_grid=param_grid, cv=5, scoring=v, n_jobs=-1)
        scores = []
        for train_idx, test_idx in cv_outer.split(X_train, y_train):
            gs.fit(X_train.iloc[train_idx], pd.Series(y_train).iloc[train_idx])
            X_test = X_train.iloc[test_idx]
            y_test = pd.Series(y_train).iloc[test_idx]
            best_model = gs.best_estimator_
            scores.append(best_model.score(X_test, y_test))

        row = {"name": f"{name}_{k}_GSCV"}
        row["pipeline"] = gs.best_estimator_
        row["params"] = gs.best_params_
        row[f"mean_{k}"] = float(np.mean(scores))
        row[f"std_{k}"] = float(np.std(scores))
        rows.append(row)

    if (mlflow_tracking):
        for row in rows:
            name = row["name"]
            pipeline = row["pipeline"]
            params = row["params"]
            metrics = {k: v for k,v in row.items() if k not in ["name", "pipeline", "params"]}
            pipeline.fit(X_train, y_train)

            with mlflow.start_run(run_name=f"training-{name}"):
                mlflow.log_metrics(metrics)
                mlflow.log_params(params)
                signature = infer_signature(X_train, pipeline.predict(X_train))
                mlflow.sklearn.log_model(sk_model=pipeline, name=name, signature=signature)

    else:
        print("mlfow tracking diabled")
        print(rows)

def train_linear_regression():
    d = data.Data()
    X_train, X_test, y_train, y_test = d.train_test_split(test_ratio=0.4, random_state=random_state)

    pipelines = {}
    preprocessor = lr_prep_stdscaler
    
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
        ("rfe", RFE(estimator=LinearSVR(max_iter=10000), n_features_to_select=min(20, X_train.shape[1] // 2))),
        ("regressor", LinearRegression())
    ])

    scoring = {"train_r2": "r2", "train_mae": "neg_mean_absolute_error", "rmse_train": "neg_root_mean_squared_error"}
    cv = make_cv(y_train, n_splits=5, random_state=random_state)
    cv_and_log(X_train, y_train, pipelines, scoring, cv)

def train_dt_regression():
    d = data.Data()
    X_train, _, y_train, _ = d.train_test_split(test_ratio=0.4, random_state=random_state)

    preprocessor = dt_prep

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", DecisionTreeRegressor(random_state=random_state))
    ])
    param_grid = {
        "regressor__max_depth": list(range(2, 7)),
        "regressor__min_samples_split": list(range(2, 20)),
        "regressor__criterion": ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
    }

    scoring = {"train_mae": "neg_mean_absolute_error", "train_rmse": "neg_root_mean_squared_error"}
    cv_outer = make_cv(y_train, n_splits=5, random_state=random_state)
    gscv_and_log("DT_reg", X_train, y_train, pipe, param_grid, scoring, cv_outer)

def train_logistic_regression():
    d = data.Data()
    X_train, X_test, y_train, y_test = d.train_test_split(test_ratio=0.4, random_state=random_state)
    y_train_clf = grade_to_pass_fail(y_train)

    pipelines = {}
    preprocessor = lr_prep_stdscaler
    
    pipelines["Dummy_most_frequent"] = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", DummyClassifier(strategy="most_frequent"))
    ])
    pipelines["Dummy_stratified"] = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", DummyClassifier(strategy="stratified"))
    ])
    pipelines["LogisticRegression"] = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", LogisticRegression(max_iter=500))
    ])
    pipelines["LogisticRegression_Univariate_SelectKBest"] = Pipeline([
        ("preprocessor", preprocessor),
        ("select", SelectKBest(score_func=f_classif, k=min(20, X_train.shape[1]))),
        ("clf", LogisticRegression(max_iter=500))
    ])
    pipelines["LogisticRegression_RFE_LinearSVC"] = Pipeline([
        ("preprocessor", preprocessor),
        ("rfe", RFE(estimator=LinearSVC(dual=False, max_iter=5000), n_features_to_select=min(20, X_train.shape[1] // 2))),
        ("clf", LogisticRegression(max_iter=500))
    ])
    pipelines["LogisticRegression_L1"] = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", LogisticRegression(penalty="l1", solver="saga", C=1.0, max_iter=2000))
    ])
    pipelines["LogisticRegression_L2"] = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", LogisticRegression(penalty="l2", solver="saga", C=1.0, max_iter=2000))
    ])

    scoring = {"train_accuracy": "accuracy", "train_f1": "f1"}
    cv = make_cv(y_train_clf, n_splits=5, random_state=random_state)
    cv_and_log(X_train, y_train_clf, pipelines, scoring, cv)

if __name__ == '__main__':
    train_dt_regression()
