from inspect import signature
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
        row["pipeline"] = pipe
        for k, v in scoring.items():
            row[f"mean_{k}"] = np.mean(cvres[f"test_{k}"])
            row[f"std_{k}"] = np.std(cvres[f"test_{k}"])
        pipe.fit(X_train, y_train)
        signature = infer_signature(X_train, pipe.predict(X_train))
        row["signature"] = signature
        rows.append(row)

    # return rows
    parse_cv_results(rows, mlflow_tracking)

def mlflow_log(name, run_name, metrics, pipeline, signature):
    with mlflow.start_run(run_name=run_name):
        mlflow.log_metrics(metrics)
        mlflow.log_params(pipeline.get_params())
        mlflow.sklearn.log_model(sk_model=pipeline, name=name, signature=signature)

def parse_cv_results(rows, mlflow_tracking):
    for row in rows:
        name = row["model"]
        pipeline = row["pipeline"]
        metrics = {k: v for k,v in row.items() if k not in ["model", "pipeline", "signature"]}
        signature = row["signature"]
        
        if mlflow_tracking:
            mlflow_log(name, f"cv-{name}", metrics, pipeline, signature)

        else:
            df = pd.DataFrame(rows)
            res = df.drop(columns=["pipeline", "signature"])
            print("mlflow tracking disabled")
            print(res)

def gscv_and_log(pipelines, X_train, y_train, scoring, cv_outer):
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

    # return rows
    parse_gscv_results(rows, mlflow_tracking)

def parse_gscv_results(rows, mlflow_tracking):
    for row in rows:
        name = row["name"]
        pipeline = row["pipeline"]
        params = row["params"]
        metrics = {k: v for k,v in row.items() if k not in ["name", "pipeline", "params", "signature"]}
        signature = row["signature"]

        if mlflow_tracking:
            mlflow_log(name, f"cv-{name}", metrics, pipeline, signature)

        else:
            print("mlfow tracking diabled")
            df = pd.DataFrame(rows)
            res = df.drop(columns=["pipeline", "params"])
            print(res)


class regression_baselines:
    def __init__(self):
        self.data = data.Data()
        self.X_train, self.X_test, self.y_train, self.y_test = self.data.train_test_split(test_ratio=0.4, random_state=random_state)
        self.scoring = {"train_mae": "neg_mean_absolute_error", "train_rmse": "neg_root_mean_squared_error"}
        self.cv = make_cv(self.y_train, n_splits=5, random_state=random_state)
        self.pipelines = {}
        self.gscv_pipelines = {}
        
        self.pipelines["LinearRegression"] = Pipeline([
            ("preprocessor", lr_prep_stdscaler),
            ("regressor", LinearRegression())
        ])
        self.pipelines["LinearRegressionSelectKBest"] = Pipeline([
            ("preprocessor", lr_prep_stdscaler),
            ("select", SelectKBest(score_func=f_regression, k=min(20, self.X_train.shape[1]))),
            ("regressor", LinearRegression())
        ])
        self.pipelines["LinearRegressionRFELinearSVR"] = Pipeline([
            ("preprocessor", lr_prep_stdscaler),
            ("rfe", RFE(estimator=LinearSVR(max_iter=10000), n_features_to_select=min(20, self.X_train.shape[1] // 2))),
            ("regressor", LinearRegression())
        ])

        self.gscv_pipelines["DT_reg"] = {
            "pipeline":Pipeline([
                ("preprocessor", dt_prep),
                ("regressor", DecisionTreeRegressor(random_state=random_state))
            ]),
            "param_grid": {
                "regressor__max_depth": list(range(2, 7)),
                "regressor__min_samples_split": list(range(2, 20)),
                "regressor__criterion": ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
            }
        }

    def cv_regression_baselines(self):
        cv_and_log(self.X_train, self.y_train, self.pipelines, self.scoring, self.cv)
        gscv_and_log(self.gscv_pipelines, self.X_train, self.y_train, self.scoring, self.cv)




def cv_clf_baselines():
    d = data.Data()
    X_train, _, y_train, _ = d.train_test_split(test_ratio=0.4, random_state=random_state)
    y_train_clf = grade_to_pass_fail(y_train)

    preprocessor = dt_prep

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", DecisionTreeClassifier(random_state=random_state))
    ])
    param_grid = {
        "clf__max_depth": list(range(2, 7)),
        "clf__min_samples_split": list(range(2, 20)),
        "clf__criterion": ['gini', 'entropy', 'log_loss']
    }

    scoring = {"train_accuracy": "accuracy", "train_f1": "f1"}
    cv_outer = make_cv(y_train_clf, n_splits=5, random_state=random_state)
    gscv_and_log("DT_clf", X_train, y_train_clf, pipe, param_grid, scoring, cv_outer)

    pipelines = {}
    preprocessor = lr_prep_stdscaler
    
    pipelines["LogisticRegression"] = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", LogisticRegression(max_iter=500, random_state=random_state))
    ])
    pipelines["LogisticRegression_Univariate_SelectKBest"] = Pipeline([
        ("preprocessor", preprocessor),
        ("select", SelectKBest(score_func=f_classif, k=min(20, X_train.shape[1]))),
        ("clf", LogisticRegression(max_iter=500, random_state=random_state))
    ])
    pipelines["LogisticRegression_RFE_LinearSVC"] = Pipeline([
        ("preprocessor", preprocessor),
        ("rfe", RFE(estimator=LinearSVC(dual=False, max_iter=5000), n_features_to_select=min(20, X_train.shape[1] // 2))),
        ("clf", LogisticRegression(max_iter=500, random_state=random_state))
    ])

    scoring = {"train_accuracy": "accuracy", "train_f1": "f1"}
    cv = make_cv(y_train_clf, n_splits=5, random_state=random_state)
    cv_and_log(X_train, y_train_clf, pipelines, scoring, cv)

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", LogisticRegression(max_iter=2000, random_state=random_state))
    ])

    param_grid = {
        "clf__C": [0.01, 0.1, 1.0, 10.0],
        "clf__penalty": ["l2"],
        "clf__solver": ["lbfgs", "liblinear", "newton-cg", "sag", "saga"]
    }

    scoring = {"train_accuracy": "accuracy", "train_f1": "f1"}
    cv_outer = make_cv(y_train_clf, n_splits=5, random_state=random_state)
    gscv_and_log("LogisticRegression", X_train, y_train_clf, pipe, param_grid, scoring, cv_outer)


if __name__ == '__main__':
    rb = regression_baselines()
    rb.cv_regression_baselines()
