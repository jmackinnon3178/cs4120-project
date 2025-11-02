from sklearn import metrics
from train_baselines import regression_baselines, classification_baselines
from utils import mlflow_log, parse_results
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, accuracy_score
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature

random_state = 1
reg = regression_baselines()
clf = classification_baselines()

def reg_test_metrics():
    reg.train_baseline_models()

    rows = []
    for name, pipeline in reg.pipelines.items():
        y_pred = pipeline.predict(reg.X_test)

        row = {"model": name}
        row["pipeline"] = pipeline
        row["mae_t"] = mean_absolute_error(reg.y_test, y_pred)
        row["rmse_t"] = np.sqrt(mean_squared_error(reg.y_test, y_pred))
        row["signature"] = infer_signature(reg.X_test, y_pred)
        rows.append(row)

    return rows

def clf_test_metrics():
    clf.train_baseline_models()

    rows = []
    for name, pipeline in clf.pipelines.items():
        y_pred = pipeline.predict(clf.X_test)
        
        row = {"model": name}
        row["pipeline"] = pipeline
        row["f1_t"] = f1_score(clf.y_test_clf, y_pred)
        row["accuracy_t"] = accuracy_score(clf.y_test_clf, y_pred)
        row["signature"] = infer_signature(clf.X_test, y_pred)
        rows.append(row)

    return rows

def reg_cv_metrics():
    rows = reg.cv_regression_baselines(False)
    return rows

def cv_and_test_metrics(cv_rows, test_rows):
    cv_df = (pd.DataFrame(cv_rows)).drop(columns=["pipeline", "signature"])
    test_df = (pd.DataFrame(test_rows)).drop(columns=["pipeline", "signature"])
        
    comb_df = cv_df.combine_first(test_df)
    comb_df["mean_mae_cv"] = comb_df["mean_mae_cv"].abs()
    comb_df["mean_rmse_cv"] = comb_df["mean_mae_cv"].abs()
    comb_df = comb_df[["model", "mae_t", "mean_mae_cv", "std_mae_cv", "rmse_t", "mean_rmse_cv", "std_rmse_cv"]]
    comb_df = comb_df.sort_values(by="mae_t")
    comb_df = comb_df.reset_index(drop=True)
    return comb_df


if __name__ == '__main__':
    print(cv_and_test_metrics(reg_cv_metrics(), reg_test_metrics()))

    # print(parse_results(reg_test_metrics(), True, cv=False))
    # print(parse_results(clf_test_metrics(), True, cv=False))
