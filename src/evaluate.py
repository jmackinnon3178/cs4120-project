from sklearn import metrics
from train_baselines import regression_baselines, classification_baselines
from utils import mlflow_log, parse_cv_results
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, accuracy_score
import numpy as np
from mlflow.models.signature import infer_signature

random_state = 1
reg = regression_baselines()
clf = classification_baselines()

def reg_test_metrics():
    reg.train_baseline_models()

    rows = []
    for name, pipeline in reg.pipelines.items():
        y_pred = pipeline.predict(reg.X_test)
        mae = mean_absolute_error(reg.y_test, y_pred)
        mse = mean_squared_error(reg.y_test, y_pred)
        rmse = np.sqrt(mse)

        row = {"model": name}
        row["pipeline"] = pipeline
        row["mae_t"] = mae
        row["rmse_t"] = rmse
        signature = infer_signature(reg.X_test, y_pred)
        row["signature"] = signature
        rows.append(row)

    return rows

def clf_test_metrics():
    clf.train_baseline_models()

    rows = []
    for name, pipeline in clf.pipelines.items():
        y_pred = pipeline.predict(clf.X_test)
        f1 = f1_score(clf.y_test_clf, y_pred)
        accuracy = accuracy_score(clf.y_test_clf, y_pred)

        row = {"model": name}
        row["pipeline"] = pipeline
        row["f1_t"] = f1
        row["accuracy_t"] = accuracy
        signature = infer_signature(clf.X_test, y_pred)
        row["signature"] = signature
        rows.append(row)

    return rows

if __name__ == '__main__':
    print(parse_cv_results(reg_test_metrics(), False))
    print(parse_cv_results(clf_test_metrics(), False))
