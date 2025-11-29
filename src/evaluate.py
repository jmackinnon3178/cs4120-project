from sklearn import metrics
from train_baselines import regression_baselines, classification_baselines
from utils import mlflow_log, parse_results
from features import grade_to_pass_fail, lr_prep_stdscaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, accuracy_score
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
import data
import seaborn as sns
import matplotlib.pyplot as plt
from features import feature_cols_numeric
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras import utils, models
from train_nn import nn_clf_final, nn_reg_final
from sklearn.inspection import permutation_importance
import pickle


utils.set_random_seed(37)
random_state = 1
reg = regression_baselines()
clf = classification_baselines()
data = data.Data()
X_train_clf, X_test_clf, X_val_clf, y_train_clf, y_test_clf, y_val_clf = data.train_test_val_split(train_ratio=0.6, test_ratio=0.2, val_ratio=0.2, random_state=random_state, clf=True)
X_train_reg, X_test_reg, X_val_reg, y_train_reg, y_test_reg, y_val_reg = data.train_test_val_split(train_ratio=0.6, test_ratio=0.2, val_ratio=0.2, random_state=random_state, clf=False)

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
        row["f1_t"] = f1_score(clf.y_test, y_pred)
        row["accuracy_t"] = accuracy_score(clf.y_test, y_pred)
        row["signature"] = infer_signature(clf.X_test, y_pred)
        rows.append(row)

    return rows

def reg_cv_metrics():
    rows = reg.cv_regression_baselines(False)
    return rows

def clf_cv_metrics():
    rows = clf.cv_classification_baselines(False)
    return rows

def cv_and_test_metrics(cv_rows, test_rows, task, to_md=False):
    cv_df = (pd.DataFrame(cv_rows)).drop(columns=["pipeline", "signature"])
    test_df = (pd.DataFrame(test_rows)).drop(columns=["pipeline", "signature"])
        
    comb_df = cv_df.combine_first(test_df)

    if task == "reg":
        comb_df["mean_mae_cv"] = comb_df["mean_mae_cv"].abs()
        comb_df["mean_rmse_cv"] = comb_df["mean_mae_cv"].abs()
        comb_df = comb_df[["model", "mae_t", "mean_mae_cv", "std_mae_cv", "rmse_t", "mean_rmse_cv", "std_rmse_cv"]]
        comb_df = comb_df.sort_values(by="mae_t")
    else:
        comb_df["mean_accuracy_cv"] = comb_df["mean_accuracy_cv"].abs()
        comb_df["mean_f1_cv"] = comb_df["mean_f1_cv"].abs()
        comb_df = comb_df[["model", "accuracy_t", "mean_accuracy_cv", "std_accuracy_cv", "f1_t", "mean_f1_cv", "std_f1_cv"]]
        comb_df = comb_df.sort_values(by="f1_t", ascending=False)

    comb_df = comb_df.reset_index(drop=True)

    if to_md:
        with open(f"./notebooks/{task}_metrics.md", "w") as f:
            f.write(comb_df.to_markdown())

    return comb_df

def eda_plots():
    numcols = feature_cols_numeric
    numcols.append("G3")
    d = data.Data()
    y = d.y
    num_df = d.data[numcols]

    plt.figure(figsize=(6, 4))
    corr = num_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title("Correlation Heatmap of Numeric Features")
    plt.savefig("./notebooks/heatmap.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6, 4))
    y_labeled = grade_to_pass_fail(y)
    pass_fail_labels = {0: "fail", 1: "pass"}
    y_labeled_named = y_labeled.map(pass_fail_labels)
    ax = sns.countplot(x=y_labeled_named, palette='pastel')
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height + 0.5,
                int(height), ha="center", va="bottom", fontsize=10)
    plt.title('Target Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.savefig("./notebooks/target_boxplot.png", bbox_inches="tight")
    plt.close()

def residual_plot():
    pipeline = reg.pipelines["DecisionTreeRegressor"]
    pipeline.fit(reg.X_train, reg.y_train)
    y_pred = pipeline.predict(reg.X_test)
    residuals = reg.y_test - y_pred

    plt.figure(figsize=(6,4))
    plt.scatter(y_pred, residuals, color='teal', alpha=0.7)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title("DecisionTreeRegressor Residuals vs Predicted")
    plt.savefig("./notebooks/residuals_vs_predicted.png", bbox_inches="tight")
    plt.close()

def plot_confusion_matrix():
    pipeline = clf.pipelines["LogisticRegression"]
    pipeline.fit(clf.X_train, clf.y_train)
    y_pred = pipeline.predict(clf.X_test)

    cm = confusion_matrix(clf.y_test, y_pred, labels=[0,1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fail', 'Pass'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("LogisticRegression Confusion Matrix")
    plt.savefig("./notebooks/confusion_matrix.png", bbox_inches="tight")
    plt.close()

def plot_clf_nn():
    history = nn_clf_final(X_train_clf, X_val_clf, y_train_clf, y_val_clf)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.title("Classification NN Learning Curve")
    plt.savefig("./notebooks/clf_nn_learning_curve.png", bbox_inches="tight")
    plt.close()

def plot_reg_nn():
    history = nn_reg_final(X_train_reg, X_val_reg, y_train_reg, y_val_reg)
    plt.plot(history.history['mean_absolute_error'], label='mean_absolute_error')
    plt.plot(history.history['val_mean_absolute_error'], label='val_mean_absolute_error')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend(loc='best')
    plt.title("Regression NN Learning Curve")
    plt.savefig("./notebooks/reg_nn_learning_curve.png", bbox_inches="tight")
    plt.close()

def perm_imp():
    model = clf.pipelines["LogisticRegression"]
    model.fit(clf.X_train, clf.y_train)
    res = permutation_importance(model, clf.X_test, clf.y_test, n_repeats=30, random_state=random_state, scoring='accuracy')

    feat_names = data.dataset.variables.name[:32]

    for i in res.importances_mean.argsort()[::-1]:
        print(f"{feat_names[i]:<8}"
            f"{res.importances_mean[i]:.3f}"
            f" +/- {res.importances_std[i]:.3f}")

def plot_perm_imp():
    model = clf.pipelines["LogisticRegression"]
    model.fit(clf.X_train, clf.y_train)
    res = permutation_importance(model, clf.X_test, clf.y_test, n_repeats=30, random_state=random_state, scoring='accuracy')

    feat_names = data.dataset.variables.name[:32]

    res_sorted = res.importances_mean.argsort()[::-1]
    plt.figure(figsize=(8,6))
    plt.barh(
        feat_names[res_sorted],
        res.importances_mean[res_sorted]
    )
    plt.title("Permutation Importance (Test set)")
    plt.xlabel("Mean decrease in accuracy")
    plt.gca().invert_yaxis()
    plt.savefig("./notebooks/permutation_importance.png", bbox_inches="tight")
    plt.close()
    
def nn_clf_metrics():
    model = models.load_model("./models/nn_clf.keras")
    X_test = lr_prep_stdscaler.fit_transform(X_test_clf)
    score = model.evaluate(X_test, y_test_clf)
    with open("./models/nn_clf_hist.pkl", "rb") as f:
        history = pickle.load(f)

    test_acc = score[1]
    test_f1 = score[2]
    val_acc = history["val_accuracy"][-1]
    val_f1 = history["val_f1_score"][-1]

    return {"model": "Classification MLP NN", "accuracy_t": test_acc, "accuracy_val": val_acc, "f1_t": test_f1, "f1_val": val_f1}

def nn_reg_metrics():
    model = models.load_model("./models/nn_reg.keras")
    X_test = lr_prep_stdscaler.fit_transform(X_test_reg)
    score = model.evaluate(X_test, y_test_reg)
    with open("./models/nn_reg_hist.pkl", "rb") as f:
        history = pickle.load(f)

    test_mae = score[1]
    test_rmse = score[2]
    val_mae = history["val_mean_absolute_error"][-1]
    val_rmse = history["val_root_mean_squared_error"][-1]

    return {"model": "Regression MLP NN", "mae_t": test_mae, "rmse_t": test_rmse, "mae_val": val_mae, "rmse_val": val_rmse}

def clf_comp_table(to_md=False, to_tex=False):
    classic_df = cv_and_test_metrics(clf_cv_metrics(), clf_test_metrics(), "clf", to_md=False).head(1).drop(columns=["std_accuracy_cv", "std_f1_cv"])
    classic_df.columns = ["model", "accuracy_t", "accuracy_val", "f1_t", "f1_val"]
    nn_df = pd.DataFrame(nn_clf_metrics().items()).set_index(0).T
    nn_df.columns = ["model", "accuracy_t", "accuracy_val", "f1_t", "f1_val"]

    comb_df = pd.concat([classic_df, nn_df], ignore_index=True)

    if to_md:
        with open(f"./notebooks/clf_nn_classic_metrics.md", "w") as f:
            f.write(comb_df.to_markdown())
    elif to_tex:
        with open(f"./notebooks/clf_nn_classic_metrics.tex", "w") as f:
            f.write(comb_df.to_latex())
    else:
        return comb_df

def reg_comp_table(to_md=False, to_tex=False):
    classic_df = cv_and_test_metrics(reg_cv_metrics(), reg_test_metrics(), "reg", to_md=False).head(1).drop(columns=["std_mae_cv", "std_rmse_cv"])
    classic_df.columns = ["model", "mae_t", "mae_val", "rmse_t", "rmse_val"]
    nn_df = pd.DataFrame(nn_reg_metrics().items()).set_index(0).T
    nn_df.columns = ["model", "mae_t", "mae_val", "rmse_t", "rmse_val"]

    comb_df = pd.concat([classic_df, nn_df], ignore_index=True)

    if to_md:
        with open(f"./notebooks/reg_nn_classic_metrics.md", "w") as f:
            f.write(comb_df.to_markdown())
    elif to_tex:
        with open(f"./notebooks/reg_nn_classic_metrics.tex", "w") as f:
            f.write(comb_df.to_latex())
    else:
        return comb_df

if __name__ == '__main__':
    clf_comp_table(to_tex=True)
    reg_comp_table(to_tex=True)
    # print(clf_comp_table())
    # print(reg_comp_table())
    # nn_clf_metrics()
    # nn_reg_metrics()
    # print(cv_and_test_metrics(reg_cv_metrics(), reg_test_metrics(), "reg", to_md=False).iloc[0])
    # print(cv_and_test_metrics(clf_cv_metrics(), clf_test_metrics(), "clf", to_md=False).iloc[0])
    # plot_perm_imp()
    # plot_clf_nn()
    # plot_reg_nn()
    # eda_plots()
    # residual_plot()
    # plot_confusion_matrix()

