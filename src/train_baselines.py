import mlflow
import data
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE
from sklearn.svm import LinearSVC, LinearSVR
from features import lr_prep_stdscaler, grade_to_pass_fail, dt_prep
from utils import make_cv, cross_val, gs_cross_val, parse_results, parse_gscv_results

mlflow_tracking = True
run_gscv = False
random_state = 1

if (mlflow_tracking):
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    baseline_experiment = mlflow.set_experiment("Baseline_Models")
# run in cs4120-project directory mlflow server --host 127.0.0.1 --port 8080 --artifacts-destination ./models


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
        self.pipelines["DecisionTreeRegressor"] = Pipeline([
            ("preprocessor", dt_prep),
            ("regressor", DecisionTreeRegressor(criterion='absolute_error', max_depth=3, random_state=random_state))
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

    def cv_regression_baselines(self, run_gscv):
        parse_results(cross_val(self.X_train, self.y_train, self.pipelines, self.scoring, self.cv), True)
        if run_gscv:
            parse_gscv_results(gs_cross_val(self.gscv_pipelines, self.X_train, self.y_train, self.scoring, self.cv), True)

    def train_baseline_models(self):
        for _, pipeline in self.pipelines.items():
            pipeline.fit(self.X_train, self.y_train)


class classification_baselines:
    def __init__(self):
        self.data = data.Data()
        self.X_train, self.X_test, self.y_train, self.y_test = self.data.train_test_split(test_ratio=0.4, random_state=random_state)
        self.y_train_clf = grade_to_pass_fail(self.y_train)
        self.y_test_clf = grade_to_pass_fail(self.y_test)
        self.scoring = {"train_accuracy": "accuracy", "train_f1": "f1"}
        self.cv = make_cv(self.y_train_clf, n_splits=5, random_state=random_state)
        self.pipelines = {}
        self.gscv_pipelines = {}

        self.pipelines["LogisticRegression"] = Pipeline([
            ("preprocessor", lr_prep_stdscaler),
            ("clf", LogisticRegression(max_iter=500, random_state=random_state))
        ])
        self.pipelines["LogisticRegression_Univariate_SelectKBest"] = Pipeline([
            ("preprocessor", lr_prep_stdscaler),
            ("select", SelectKBest(score_func=f_classif, k=min(20, self.X_train.shape[1]))),
            ("clf", LogisticRegression(max_iter=500, random_state=random_state))
        ])
        self.pipelines["LogisticRegression_RFE_LinearSVC"] = Pipeline([
            ("preprocessor", lr_prep_stdscaler),
            ("rfe", RFE(estimator=LinearSVC(dual=False, max_iter=5000), n_features_to_select=min(20, self.X_train.shape[1] // 2))),
            ("clf", LogisticRegression(max_iter=500, random_state=random_state))
        ])
        self.pipelines["DecisionTreeClassifier"] = Pipeline([
            ("preprocessor", dt_prep),
            ("clf", DecisionTreeClassifier(criterion='entropy', max_depth=2,random_state=random_state))
        ])

        self.gscv_pipelines["DT_clf"] = {
            "pipeline": Pipeline([
                ("preprocessor", dt_prep),
                ("clf", DecisionTreeClassifier(random_state=random_state))
            ]),
            "param_grid": {
                "clf__max_depth": list(range(2, 7)),
                "clf__min_samples_split": list(range(2, 20)),
                "clf__criterion": ['gini', 'entropy', 'log_loss']
            }
        }
        self.gscv_pipelines["LogisticRegression"] = {
            "pipeline": Pipeline([
                ("preprocessor", lr_prep_stdscaler),
                ("clf", LogisticRegression(max_iter=2000, random_state=random_state))
            ]),
            "param_grid": {
                "clf__C": [0.01, 0.1, 1.0, 10.0],
                "clf__penalty": ["l2"],
                "clf__solver": ["lbfgs", "liblinear", "newton-cg", "sag", "saga"]
            }
        }

    def cv_classification_baselines(self, run_gscv):
        parse_results(cross_val(self.X_train, self.y_train_clf, self.pipelines, self.scoring, self.cv), True)
        if run_gscv:
            parse_gscv_results(gs_cross_val(self.gscv_pipelines, self.X_train, self.y_train_clf, self.scoring, self.cv), True)

    def train_baseline_models(self):
        for _, pipeline in self.pipelines.items():
            pipeline.fit(self.X_train, self.y_train_clf)

if __name__ == '__main__':
    rb = regression_baselines()
    rb.cv_regression_baselines(run_gscv)
    cb = classification_baselines()
    cb.cv_classification_baselines(run_gscv)
