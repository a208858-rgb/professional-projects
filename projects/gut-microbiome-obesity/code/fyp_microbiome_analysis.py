"""Genus-level cohort analysis for the FYP microbiome classification study."""

import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.feature_selection import RFECV, SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
plt.rcParams.update(
    {"font.family": "Arial", "axes.spines.top": False, "axes.spines.right": False}
)

# ======================= CONFIG =======================
COHORT = "Overall"  # "Overall" | "Malay" | "Chinese"
N_NORMAL = {"Overall": 250, "Malay": None, "Chinese": None}[COHORT]
FIRST_GENUS = "Bifidobacterium"
SEED, TEST_SIZE, N_FOLDS = 42, 0.30, 5
SCALE_BEFORE_SPLIT = True  # Set False to fit the scaler on training data only.
RUN_RFE_SBE = False  # Enable to run feature selection after the baseline analysis.
OUT = f"./model_{COHORT}_output/"
os.makedirs(OUT, exist_ok=True)
COLORS = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9"]

# ======================= A. LOAD =======================
df = pd.read_csv(f"cohort_{COHORT.lower()}.csv", index_col=0)
if FIRST_GENUS not in df.columns:
    raise ValueError(f"Genus column {FIRST_GENUS!r} was not found in the cohort CSV")
genus_start = df.columns.get_loc(FIRST_GENUS)
meta_cols = df.columns[:genus_start].tolist()
genus_cols = df.columns[genus_start:].tolist()
print(COHORT, df.shape, "| genus:", len(genus_cols))

# ======================= B. EDA =======================
print(df["Label"].value_counts().rename({0: "Normal", 1: "Obese/OW"}))
for column in ("Ethnicity", "Gender", "Age_Group"):
    if column in df:
        print(df[column].value_counts().to_dict())

genus = df[genus_cols].fillna(0)
print(
    f"Sparsity: {(genus == 0).values.mean() * 100:.1f}% zeros | "
    f"genera absent in >50% samples: {int(((genus == 0).mean() > 0.5).sum())}"
)
relative = genus.div(genus.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
alpha = pd.DataFrame(
    {
        "Shannon": -(relative * np.log(relative.where(relative > 0))).sum(axis=1),
        "Observed genera": (genus > 0).sum(axis=1),
        "Group": df["Label"].map({0: "Normal", 1: "Obese/OW"}),
    }
)
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for axis, measure in zip(axes, ("Shannon", "Observed genera")):
    sns.boxplot(data=alpha, x="Group", y=measure, ax=axis, palette=["#0072B2", "#D55E00"])
    normal_values = alpha.loc[alpha.Group == "Normal", measure]
    obese_values = alpha.loc[alpha.Group == "Obese/OW", measure]
    p_value = mannwhitneyu(normal_values, obese_values).pvalue
    axis.set_title(f"{measure} (Mann-Whitney p = {p_value:.3f})")
fig.savefig(OUT + "EDA_alpha.png", dpi=600, bbox_inches="tight")
plt.close(fig)

top_genera = genus.mean().nlargest(20).index
heatmap_data = genus[top_genera].groupby(
    df["Label"].map({0: "Normal", 1: "Obese/OW"})
).mean()
sns.heatmap(heatmap_data, cmap="YlOrRd")
plt.savefig(OUT + "EDA_heatmap.png", dpi=600, bbox_inches="tight")
plt.close()

pca_scores = PCA(n_components=2, random_state=SEED).fit_transform(
    StandardScaler().fit_transform(genus)
)
plt.scatter(pca_scores[:, 0], pca_scores[:, 1],
            c=df["Label"].map({0: "#0072B2", 1: "#D55E00"}), s=18)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig(OUT + "EDA_PCA.png", dpi=600, bbox_inches="tight")
plt.close()

# ============ C. UNDERSAMPLE -> MIN-MAX -> 70/30 SPLIT ============
normal, obese = df[df.Label == 0], df[df.Label == 1]
if N_NORMAL is not None:
    if N_NORMAL > len(normal):
        raise ValueError(f"Requested {N_NORMAL} normal samples, but only {len(normal)} are available")
    normal = normal.sample(n=N_NORMAL, random_state=SEED, replace=False)
balanced = pd.concat([normal, obese]).sample(frac=1, random_state=SEED)
X = balanced[genus_cols].fillna(0).values
y = balanced["Label"].values
positions = np.arange(len(balanced))

if SCALE_BEFORE_SPLIT:
    X = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test, i_train, i_test = train_test_split(
        X, y, positions, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
else:
    X_train, X_test, y_train, y_test, i_train, i_test = train_test_split(
        X, y, positions, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    scaler = MinMaxScaler().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

print(
    f"Balanced {len(balanced)} | train {len(y_train)} | test {len(y_test)} | "
    f"obese% train {y_train.mean():.2f} test {y_test.mean():.2f}"
)
for name, indices, values in (("training", i_train, X_train), ("test", i_test, X_test)):
    metadata = balanced.iloc[indices][meta_cols]
    output = pd.concat(
        [metadata, pd.DataFrame(values, columns=genus_cols, index=metadata.index)], axis=1
    )
    output.to_csv(f"{name}_data_{COHORT}.csv")

# ============ D. SIX MODELS: 5-FOLD CV + TEST SET ============
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


def build_models():
    base_models = [
        ("rf", RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1)),
        ("svm", SVC(kernel="rbf", C=1.0, probability=True, class_weight="balanced", random_state=SEED)),
        ("lr", LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=SEED)),
        ("xgb", XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                               eval_metric="logloss", random_state=SEED, verbosity=0)),
        ("nb", GaussianNB()),
    ]
    models = dict(zip(
        ["Random Forest", "SVM", "Logistic Regression", "XGBoost", "Naive Bayes"],
        [model for _, model in base_models],
    ))
    models["Stacked Ensemble"] = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(max_iter=1000, random_state=SEED),
        cv=5,
        stack_method="predict_proba",
        n_jobs=-1,
    )
    return models


def evaluate(X_train_data, X_test_data, tag):
    rows, fitted, roc_data = [], {}, {}
    for name, model in build_models().items():
        cv_result = cross_validate(
            model, X_train_data, y_train, cv=cv,
            scoring=["accuracy", "roc_auc"], n_jobs=-1,
        )
        model.fit(X_train_data, y_train)
        probabilities = model.predict_proba(X_test_data)[:, 1]
        predictions = model.predict(X_test_data)
        auc = roc_auc_score(y_test, probabilities)
        rows.append(
            dict(
                Model=name,
                CV_Acc=cv_result["test_accuracy"].mean(),
                CV_AUC=cv_result["test_roc_auc"].mean(),
                Test_Acc=accuracy_score(y_test, predictions),
                Precision=precision_score(y_test, predictions, zero_division=0),
                Recall=recall_score(y_test, predictions, zero_division=0),
                F1=f1_score(y_test, predictions, zero_division=0),
                Test_AUC=auc,
            )
        )
        fitted[name] = model
        roc_data[name] = (*roc_curve(y_test, probabilities)[:2], auc)
        model_file = f"{OUT}{tag}_{name.lower().replace(' ', '_')}.joblib"
        joblib.dump(model, model_file)

    results = pd.DataFrame(rows).set_index("Model").round(4).sort_values("Test_AUC", ascending=False)
    results.to_csv(f"{OUT}{tag}_results.csv")
    print(f"\n[{tag}]\n{results}")
    return fitted, roc_data


def plot_roc(roc_data, tag):
    fig, axis = plt.subplots(figsize=(7, 6))
    ordered = sorted(roc_data.items(), key=lambda item: -item[1][2])
    for (name, (fpr, tpr, auc)), color in zip(ordered, COLORS):
        axis.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {auc:.3f})")
    axis.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.500)")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title(f"ROC Curves: {COHORT} Cohort ({tag})", loc="left", fontweight="bold")
    axis.legend(frameon=False, loc="lower right")
    fig.savefig(f"{OUT}ROC_{tag}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_rf_importance(rf_model, feature_names, tag):
    importance = pd.Series(rf_model.feature_importances_, index=feature_names).nlargest(20)[::-1]
    fig, axis = plt.subplots(figsize=(8, 7))
    axis.barh(importance.index, importance.values, color="#0072B2", edgecolor="black", lw=0.5)
    axis.set_xlabel("Feature importance (Gini)")
    axis.set_title(f"Top 20 RF importance: {COHORT} ({tag})", loc="left", fontweight="bold")
    fig.savefig(f"{OUT}RF_importance_{tag}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_shap(model, X_test_data, feature_names, tag):
    shap_values = shap.TreeExplainer(model).shap_values(X_test_data)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif np.ndim(shap_values) == 3:
        shap_values = shap_values[:, :, 1]
    for plot_kind in ("dot", "bar"):
        shap.summary_plot(
            shap_values, X_test_data, feature_names=feature_names,
            plot_type=plot_kind, max_display=20, show=False,
        )
        plt.title(f"SHAP {plot_kind}: {COHORT} ({tag})", loc="left", fontweight="bold")
        plt.savefig(f"{OUT}SHAP_{plot_kind}_{tag}.png", dpi=600, bbox_inches="tight")
        plt.close()
    pd.DataFrame(shap_values, columns=feature_names).to_csv(
        f"{OUT}shap_values_{tag}.csv", index=False
    )


fitted, roc_data = evaluate(X_train, X_test, "baseline")
plot_roc(roc_data, "baseline")
plot_rf_importance(fitted["Random Forest"], genus_cols, "baseline")
plot_shap(fitted["Random Forest"], X_test, genus_cols, "baseline")

# ============ G. RFE -> SBE -> CONSENSUS (optional) ============
if RUN_RFE_SBE:
    rf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1
    )
    rfe = RFECV(
        rf, step=1, cv=cv, scoring="roc_auc", min_features_to_select=5, n_jobs=-1
    ).fit(X_train, y_train)
    shortlist = [genus for genus, keep in zip(genus_cols, rfe.support_) if keep]
    fast_rf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=SEED, n_jobs=-1
    )
    sbe = SequentialFeatureSelector(
        fast_rf, n_features_to_select="auto", tol=0.0, direction="backward",
        scoring="roc_auc", cv=cv, n_jobs=-1,
    ).fit(X_train[:, rfe.support_], y_train)
    consensus = [genus for genus, keep in zip(shortlist, sbe.get_support()) if keep]
    pd.Series(consensus).to_csv(f"{OUT}consensus_genera.csv", index=False, header=["genus"])
    print(f"RFE shortlist: {len(shortlist)} | consensus: {len(consensus)}\n", consensus)
    selected_indices = [genus_cols.index(genus) for genus in consensus]
    fitted_consensus, roc_consensus = evaluate(
        X_train[:, selected_indices], X_test[:, selected_indices], "consensus"
    )
    plot_roc(roc_consensus, "consensus")
    plot_shap(
        fitted_consensus["Random Forest"],
        X_test[:, selected_indices], consensus, "consensus",
    )
