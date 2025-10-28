import numpy as np
import mlflow
import sklearn
import features
import data
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

random_state = 1
mlflow.set_tracking_uri("http://127.0.0.1:8080")
# run in cs4120-project directory
# mlflow server --host 127.0.0.1 --port 8080 --artifacts-destination ./models

def train_linear_regression(random_state: int):
    baseline_experiment = mlflow.set_experiment("Baseline_Models")
    artifact_path = "lr_baseline"
    d = data.Data()
    f = features.Features()
    preprocessor = f.preprocessor

    X_train, X_test, X_val, y_train, y_test, y_val = d.split_data(train_ratio=0.6, test_ratio=0.2, val_ratio=0.2, random_state=random_state)


    linear_regression = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    linear_model = TransformedTargetRegressor(
            regressor=linear_regression,
            transformer=StandardScaler()
    )

    linear_model.fit(X_train, y_train)
    y_pred = linear_model.predict(X_val)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    metrics = {"mae": mae, "rmse": rmse}

    with mlflow.start_run(run_name="test") as run:
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(sk_model=linear_model, input_example=X_val, name=artifact_path)

if __name__ == '__main__':
    train_linear_regression(random_state)
