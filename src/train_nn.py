from sklearn.neural_network import MLPClassifier
import data
from features import lr_prep_stdscaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score
from utils import cross_val, make_cv, parse_results

random_state = 1


scoring = {"accuracy": "accuracy", "f1": "f1"}
clf_data = data.Data()
X_train, X_test, y_train, y_test = clf_data.train_test_split(test_ratio=0.4, random_state=random_state, clf=True)
cv = make_cv(y_train, random_state=random_state)

pipelines = {
        "clf_pipeline": Pipeline([
            ("preprocessor", lr_prep_stdscaler),
            ("clf", MLPClassifier(hidden_layer_sizes=(64, 32),random_state=random_state, max_iter=1000))
        ])
}

cv_rows = cross_val(X_train, y_train, pipelines, scoring, cv)
print(parse_results(cv_rows, False))






