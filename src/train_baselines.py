import numpy as np
import mlflow
import sklearn
import features
import data
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = data.Data()
X_train, X_test, X_val, y_train, y_test, y_val = data.split_data(0.6, 0.2, 0.2, 1)

features = features.Features()
preprocessor = features.preprocessor

mlflow.set_tracking_uri("http://127.0.0.1:8080")
baseline_experiment = mlflow.set_experiment("Baseline_Models")

linear_reg = LinearRegression()
artifact_path = "lr_baseline"

X_train_enc = preprocessor.fit_transform(X_train)

X_test_enc = preprocessor.fit_transform(X_test)

linear_reg.fit(X_train_enc, y_train)
y_pred = linear_reg.predict(X_test_enc)


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

metrics = {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}

with mlflow.start_run(run_name="test") as run:
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(sk_model=LinearRegression, input_example=X_test, name=artifact_path)
