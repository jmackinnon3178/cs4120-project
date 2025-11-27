from sklearn.neural_network import MLPClassifier, MLPRegressor
import data
from features import lr_prep_stdscaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score
from utils import cross_val, make_cv, parse_results
import tensorflow as tf
from keras import layers, metrics, models, optimizers, callbacks
import optuna

random_state = 1


clf_data = data.Data()

def nn_clf_baseline():
    scoring = {"accuracy": "accuracy", "f1": "f1"}
    clf_data = data.Data()
    X_train, X_test, y_train, y_test = clf_data.train_test_split(test_ratio=0.4, random_state=random_state, clf=True)
    cv = make_cv(y_train, random_state=random_state)

    pipelines = {
            "clf_pipeline_32_16": Pipeline([
                ("preprocessor", lr_prep_stdscaler),
                ("clf", MLPClassifier(hidden_layer_sizes=(32, 16),random_state=random_state, max_iter=1000))
            ]),
            "clf_pipeline_64_32": Pipeline([
                ("preprocessor", lr_prep_stdscaler),
                ("clf", MLPClassifier(hidden_layer_sizes=(64, 32),random_state=random_state, max_iter=1000))
            ])
    }

    cv_rows = cross_val(X_train, y_train, pipelines, scoring, cv)
    print(parse_results(cv_rows, False))

def nn_clf():
    clf_data = data.Data()
    # X_train, X_test, y_train, y_test = clf_data.train_test_split(test_ratio=0.4, random_state=random_state, clf=True)
    X_train, X_test, X_val, y_train, y_test, y_val = clf_data.train_test_val_split(train_ratio=0.6, test_ratio=0.2, val_ratio=0.2, random_state=random_state, clf=True)

    X_train = lr_prep_stdscaler.fit_transform(X_train)
    X_val = lr_prep_stdscaler.transform(X_val)

    model = models.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])

    print(model.summary())

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'f1_score']
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=10,
        validation_data=(X_val, y_val)
    )

def objective(trial):
    X_train, X_test, X_val, y_train, y_test, y_val = clf_data.train_test_val_split(train_ratio=0.6, test_ratio=0.2, val_ratio=0.2, random_state=random_state, clf=True)

    X_train = lr_prep_stdscaler.fit_transform(X_train)
    X_val = lr_prep_stdscaler.transform(X_val)

    model = models.Sequential()
    model.add(layers.Input(shape=(X_train.shape[1],)))
    model.add(
        layers.Dense(
            units=trial.suggest_int("l1_units", 32, 64, step=8),
            activation=trial.suggest_categorical("activation", ["relu", "linear", "tanh", "sigmoid"])
        )
    )
    model.add(layers.Dropout(
            rate=trial.suggest_float("rate", 0.20, 0.35, step=0.05)
        )
    )
    model.add(
        layers.Dense(
            units=trial.suggest_int("l2_units", 16, 32, step=8),
            activation=trial.suggest_categorical("activation", ["relu", "linear", "tanh", "sigmoid"])
        )
    )
    model.add(layers.Dropout(
            rate=trial.suggest_float("rate", 0.20, 0.35, step=0.05)
        )
    )
    model.add(layers.Dense(1, activation='sigmoid'))

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)


    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', 'f1_score']
    )

    model.fit(
        X_train,
        y_train,
        epochs=10,
        validation_data=(X_val, y_val)
    )

    score = model.evaluate(X_val, y_val)
    return score[1]

def nn_reg_baseline():
    scoring = {"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"}
    reg_data = data.Data()
    X_train, X_test, y_train, y_test = reg_data.train_test_split(test_ratio=0.4, random_state=random_state, clf=False)
    cv = make_cv(y_train, random_state=random_state)

    pipelines = {
            "reg_pipeline_32_16": Pipeline([
                ("preprocessor", lr_prep_stdscaler),
                ("regressor", MLPRegressor(hidden_layer_sizes=(32, 16),random_state=random_state, max_iter=10000))
            ]),
            "reg_pipeline_64_32": Pipeline([
                ("preprocessor", lr_prep_stdscaler),
                ("regressor", MLPRegressor(hidden_layer_sizes=(64, 32),random_state=random_state, max_iter=10000))
            ])
    }

    cv_rows = cross_val(X_train, y_train, pipelines, scoring, cv)
    print(parse_results(cv_rows, False))

    
def nn_reg():
    reg_data = data.Data()
    # X_train, X_test, y_train, y_test = reg_data.train_test_split(test_ratio=0.4, random_state=random_state, clf=False)
    X_train, X_test, X_val, y_train, y_test, y_val = reg_data.train_test_val_split(train_ratio=0.6, test_ratio=0.2, val_ratio=0.2, random_state=random_state, clf=False)

    X_train = lr_prep_stdscaler.fit_transform(X_train)
    X_val = lr_prep_stdscaler.transform(X_val)

    model = models.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1)
    ])

    print(model.summary())

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', metrics.RootMeanSquaredError()]
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=100,
        validation_data=(X_val, y_val)
    )

if __name__ == '__main__':
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    # nn_clf()
    # nn_clf_baseline()
    # nn_reg()
    # nn_reg_baseline()

# best clf
# Best trial:
#   Value: 0.9538461565971375
#   Params:
#     l1_units: 56
#     activation: relu
#     rate: 0.2
#     l2_units: 24
#     learning_rate: 0.052707119953160395
