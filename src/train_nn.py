from sklearn.neural_network import MLPClassifier, MLPRegressor
import data
from features import lr_prep_stdscaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score
from utils import cross_val, make_cv, parse_results
import tensorflow as tf
from keras import layers, metrics, models, optimizers, callbacks, utils, losses
import optuna

utils.set_random_seed(37)
random_state = 1

clf_data = data.Data()
reg_data = data.Data()

def nn_clf_final():
    X_train, X_test, X_val, y_train, y_test, y_val = clf_data.train_test_val_split(train_ratio=0.6, test_ratio=0.2, val_ratio=0.2, random_state=random_state, clf=True)

    X_train = lr_prep_stdscaler.fit_transform(X_train)
    X_val = lr_prep_stdscaler.transform(X_val)

    model = models.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(48, activation='relu'),
        layers.Dropout(0.25),
        layers.Dense(32, activation='sigmoid'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.010485843639257204),
        loss='binary_crossentropy',
        metrics=['accuracy', 'f1_score']
    )

    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=32,
        callbacks=[early_stop],
        validation_data=(X_val, y_val)
    )

def clf_optuna():
    def objective(trial):
        X_train, X_test, X_val, y_train, y_test, y_val = clf_data.train_test_val_split(train_ratio=0.6, test_ratio=0.2, val_ratio=0.2, random_state=random_state, clf=True)

        X_train = lr_prep_stdscaler.fit_transform(X_train)
        X_val = lr_prep_stdscaler.transform(X_val)

        model = models.Sequential()
        model.add(layers.Input(shape=(X_train.shape[1],)))
        model.add(
            layers.Dense(
                units=trial.suggest_int("l1_units", 32, 64, step=8),
                activation=trial.suggest_categorical("l1_activation", ["relu", "sigmoid"])
            )
        )
        model.add(layers.Dropout(
                rate=trial.suggest_float("l1_drop_rate", 0.20, 0.35, step=0.05)
            )
        )
        model.add(
            layers.Dense(
                units=trial.suggest_int("l2_units", 16, 32, step=8),
                activation=trial.suggest_categorical("l2_activation", ["relu", "sigmoid"])
            )
        )
        model.add(layers.Dropout(
                rate=trial.suggest_float("l2_drop_rate", 0.20, 0.35, step=0.05)
            )
        )
        model.add(layers.Dense(1, activation='sigmoid'))

        opt = trial.suggest_categorical("optimizer", ["Adam", "RMSprop"])
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3
        )

        match opt:
            case "Adam":
                optimizer = optimizers.Adam(learning_rate=learning_rate)
            case "RMSprop":
                optimizer = optimizers.RMSprop(learning_rate=learning_rate)
            case _:
                optimizer = optimizers.Adam(learning_rate=learning_rate)

        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'f1_score']
        )

        model.fit(
            X_train,
            y_train,
            epochs=32,
            callbacks=[early_stop],
            validation_data=(X_val, y_val)
        )

        score = model.evaluate(X_val, y_val)
        return score[1]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=120, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

def nn_reg_final():
    X_train, X_test, X_val, y_train, y_test, y_val = reg_data.train_test_val_split(train_ratio=0.6, test_ratio=0.2, val_ratio=0.2, random_state=random_state, clf=False)

    X_train = lr_prep_stdscaler.fit_transform(X_train)
    X_val = lr_prep_stdscaler.transform(X_val)

    model = models.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(40, activation='sigmoid'),
        layers.Dropout(0.25),
        layers.Dense(24, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])

    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3
    )

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.012745953868477105),
        loss=losses.MeanSquaredError(),
        metrics=[metrics.MeanAbsoluteError(), metrics.RootMeanSquaredError()]
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=32,
        callbacks=[early_stop],
        validation_data=(X_val, y_val)
    )

def reg_optuna():
    def objective(trial):
        X_train, X_test, X_val, y_train, y_test, y_val = reg_data.train_test_val_split(train_ratio=0.6, test_ratio=0.2, val_ratio=0.2, random_state=random_state, clf=False)

        X_train = lr_prep_stdscaler.fit_transform(X_train)
        X_val = lr_prep_stdscaler.transform(X_val)

        model = models.Sequential()
        model.add(layers.Input(shape=(X_train.shape[1],)))
        model.add(
            layers.Dense(
                units=trial.suggest_int("l1_units", 32, 64, step=8),
                activation=trial.suggest_categorical("l1_activation", ["relu", "sigmoid"])
            )
        )
        model.add(layers.Dropout(
                rate=trial.suggest_float("l1_drop_rate", 0.20, 0.35, step=0.05)
            )
        )
        model.add(
            layers.Dense(
                units=trial.suggest_int("l2_units", 16, 32, step=8),
                activation=trial.suggest_categorical("l2_activation", ["relu", "sigmoid"])
            )
        )
        model.add(layers.Dropout(
                rate=trial.suggest_float("l2_drop_rate", 0.20, 0.35, step=0.05)
            )
        )
        model.add(layers.Dense(1))

        opt = trial.suggest_categorical("optimizer", ["Adam", "RMSprop"])
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3
        )

        match opt:
            case "Adam":
                optimizer = optimizers.Adam(learning_rate=learning_rate)
            case "RMSprop":
                optimizer = optimizers.RMSprop(learning_rate=learning_rate)
            case _:
                optimizer = optimizers.Adam(learning_rate=learning_rate)

        model.compile(
            optimizer=optimizer,
            loss=losses.MeanSquaredError(),
            metrics=[metrics.MeanAbsoluteError(), metrics.RootMeanSquaredError()]
        )


        model.fit(
            X_train,
            y_train,
            epochs=32,
            callbacks=[early_stop],
            validation_data=(X_val, y_val)
        )

        score = model.evaluate(X_val, y_val)
        return score[1]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=120, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == '__main__':
    nn_clf_final()
    nn_reg_final()
    # clf_optuna()
    # reg_optuna()

