from sklearn import metrics
from train_baselines import regression_baselines, classification_baselines
from utils import mlflow_log, parse_results
from features import grade_to_pass_fail
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, accuracy_score
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
import data
import seaborn as sns
import matplotlib.pyplot as plt
from features import feature_cols_numeric
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

if __name__ == '__main__':
    print(cv_and_test_metrics(reg_cv_metrics(), reg_test_metrics(), "reg", to_md=False))
    print(cv_and_test_metrics(clf_cv_metrics(), clf_test_metrics(), "clf", to_md=False))
    # eda_plots()
    # residual_plot()
    # plot_confusion_matrix()

