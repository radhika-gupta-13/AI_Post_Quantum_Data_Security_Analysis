import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

# ================================================================
# 1. FILE PATHS
# ================================================================
# Change DATA_PATH if your CSV is stored in a different location.
DATA_PATH = "/content/AI_Post_Quantum_Data_Security_1000_Synthetic.csv"
OUTPUT_DIR = "/content/Research_Paper_Figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================================================
# 2. COLORFUL CHART THEME
# ================================================================
# Bright, varied palettes while retaining readable chart labels.
sns.set_theme(style="whitegrid", context="notebook")
PALETTE = "husl"
RISK_PALETTE = {
    "Low": "#2ECC71",
    "Medium": "#F1C40F",
    "High": "#E74C3C",
    "Critical": "#8E44AD"
}
PQC_PALETTE = "bright"

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 300,
    "font.family": "serif",
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "medium",
    "figure.figsize": (9, 5.5),
    "axes.edgecolor": "#555555",
    "grid.alpha": 0.25
})

def save_figure(filename):
    """Save the current chart in high-resolution PNG format."""
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, filename),
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )
    plt.show()
    plt.close()

def tidy_label(value):
    return str(value).replace("_", " ").title()

# ================================================================
# 3. LOAD AND CLEAN DATA
# ================================================================
# keep_default_na=False preserves the literal string "None".
# Empty cells and common missing markers are handled separately.
df = pd.read_csv(DATA_PATH, keep_default_na=False)

df.columns = [
    re.sub(
        r"_+", "_",
        re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower())
    ).strip("_")
    for c in df.columns
]

df = df.drop_duplicates().reset_index(drop=True)

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace(
            {"": np.nan, "nan": np.nan, "NaN": np.nan,
             "NA": np.nan, "N/A": np.nan, "null": np.nan,
             "NULL": np.nan}
        )

# Preserve "None" as a valid PQC adoption category.
if "pqc_adoption" in df.columns:
    df["pqc_adoption"] = df["pqc_adoption"].replace(
        {"none": "None", "NONE": "None"}
    )

print("=" * 70)
print("DATASET OVERVIEW")
print("=" * 70)
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print("\nColumn names:\n", df.columns.tolist())
print("\nMissing values:\n", df.isnull().sum())
print("\nData types:\n", df.dtypes)
print("\nFirst five rows:\n", df.head().to_string(index=False))

if len(df) != 500:
    print(
        f"\nNOTE: This file contains {len(df)} rows, while the current "
        "paper says 500 synthetic profiles. Verify which dataset was "
        "used for the reported results before editing the paper."
    )

df.describe(include="all").T.to_csv(
    os.path.join(OUTPUT_DIR, "descriptive_statistics.csv")
)
df.isnull().sum().rename("missing_count").to_csv(
    os.path.join(OUTPUT_DIR, "missing_value_summary.csv")
)

# Exact column names from the supplied dataset.
ai_col = "ai_usage_level"
sensitivity_col = "data_sensitivity"
volume_col = "data_volume"
privacy_col = "privacy_risk"
ai_risk_col = "ai_security_risk"
access_col = "access_control_strength"
preservation_col = "privacy_preservation_level"
encryption_col = "encryption_strength"
pqc_col = "pqc_adoption"
agility_col = "cryptographic_agility"
monitor_col = "security_monitoring"
legacy_col = "legacy_system_dependency"
incident_col = "incident_probability"
overall_col = "overall_risk_level"

required_columns = [
    ai_col, sensitivity_col, privacy_col, ai_risk_col,
    access_col, preservation_col, encryption_col, pqc_col,
    agility_col, monitor_col, legacy_col, incident_col, overall_col
]
missing_columns = [c for c in required_columns if c not in df.columns]
if missing_columns:
    raise ValueError(
        "Expected columns not found: " + ", ".join(missing_columns)
        + "\nPlease check the CSV header."
    )

print("\nDetected research variables:")
for col in required_columns:
    print(f"  {tidy_label(col):35s}: {col}")

# ================================================================
# 4. DESCRIPTIVE ANALYSIS
# ================================================================
print("\n" + "=" * 70)
print("DESCRIPTIVE ANALYSIS")
print("=" * 70)

numeric_df = df.select_dtypes(include=np.number)
numeric_df.describe().T.to_csv(
    os.path.join(OUTPUT_DIR, "numeric_summary.csv")
)

# 4.1 Overall risk category distribution
risk_order = ["Low", "Medium", "High", "Critical"]
risk_counts = df[overall_col].value_counts().reindex(
    risk_order, fill_value=0
)
risk_counts = risk_counts[risk_counts > 0]

plt.figure(figsize=(8.5, 5.5))
colors = sns.color_palette("Set2", n_colors=len(risk_counts))
ax = sns.barplot(
    x=risk_counts.index, y=risk_counts.values,
    hue=risk_counts.index, palette=colors, legend=False
)
for container in ax.containers:
    ax.bar_label(container, fmt="%.0f", padding=3, fontsize=10)
plt.title("Distribution of Overall Security Risk")
plt.xlabel("Overall Risk Level")
plt.ylabel("Number of Organizational Profiles")
save_figure("01_overall_risk_distribution.png")

plt.figure(figsize=(7, 7))
plt.pie(
    risk_counts.values,
    labels=risk_counts.index,
    autopct="%1.1f%%",
    startangle=90,
    colors=sns.color_palette("Set3", n_colors=len(risk_counts)),
    wedgeprops={"edgecolor": "white", "linewidth": 2},
    textprops={"fontsize": 11}
)
plt.title("Percentage Distribution of Overall Risk")
save_figure("01b_overall_risk_pie.png")

# 4.2 AI usage vs. AI security risk
ai_order = ["Low", "Medium", "High"]
ai_summary = (
    df.groupby(ai_col, observed=True)[ai_risk_col]
    .agg(["mean", "std", "count"])
    .reindex(ai_order)
)
print("\nAI usage and AI security risk:\n", ai_summary)
ai_summary.to_csv(os.path.join(OUTPUT_DIR, "ai_usage_risk_summary.csv"))

plt.figure(figsize=(8.5, 5.5))
ax = sns.barplot(
    data=df, x=ai_col, y=ai_risk_col,
    order=ai_order, hue=ai_col, palette="viridis",
    errorbar=("ci", 95), legend=False
)
for container in ax.containers:
    ax.bar_label(container, fmt="%.1f", padding=3, fontsize=9)
plt.title("AI Usage Level vs. AI Security Risk")
plt.xlabel("AI Usage Level")
plt.ylabel("Mean AI Security Risk")
save_figure("02_ai_usage_vs_security_risk.png")

# 4.3 Data sensitivity vs. privacy risk
sensitivity_order = ["Low", "Medium", "High", "Critical"]
sensitivity_summary = (
    df.groupby(sensitivity_col, observed=True)[privacy_col]
    .agg(["mean", "std", "count"])
    .reindex(sensitivity_order)
)
print("\nData sensitivity and privacy risk:\n", sensitivity_summary)
sensitivity_summary.to_csv(
    os.path.join(OUTPUT_DIR, "sensitivity_privacy_summary.csv")
)

plt.figure(figsize=(9, 5.5))
ax = sns.barplot(
    data=df, x=sensitivity_col, y=privacy_col,
    order=sensitivity_order, hue=sensitivity_col,
    palette="magma", errorbar=("ci", 95), legend=False
)
for container in ax.containers:
    ax.bar_label(container, fmt="%.1f", padding=3, fontsize=9)
plt.title("Data Sensitivity vs. Privacy Risk")
plt.xlabel("Data Sensitivity Level")
plt.ylabel("Mean Privacy Risk")
save_figure("03_sensitivity_vs_privacy_risk.png")

# 4.4 Data volume distribution
plt.figure(figsize=(9, 5.5))
sns.histplot(
    data=df, x=volume_col, bins=25, kde=True,
    color="#00A8E8", edgecolor="white", alpha=0.85
)
plt.title("Distribution of Organizational Data Volume")
plt.xlabel("Data Volume")
plt.ylabel("Number of Profiles")
save_figure("03b_data_volume_distribution.png")

# ================================================================
# 5. CORRELATION ANALYSIS
# ================================================================
print("\n" + "=" * 70)
print("CORRELATION ANALYSIS")
print("=" * 70)

# Standard Pearson correlation for numeric variables.
numeric_df = df.select_dtypes(include=np.number)
if len(numeric_df.columns) >= 2:
    corr = numeric_df.corr(method="pearson")
    corr.to_csv(os.path.join(OUTPUT_DIR, "correlation_matrix.csv"))

    plt.figure(figsize=(13, 10))
    sns.heatmap(
        corr, cmap="Spectral", center=0, vmin=-1, vmax=1,
        annot=True, fmt=".2f", square=True,
        linewidths=0.5, linecolor="white",
        annot_kws={"size": 8},
        cbar_kws={"shrink": 0.8, "label": "Pearson correlation"}
    )
    plt.title("Correlation Matrix of Numeric Security Variables")
    save_figure("04_correlation_heatmap.png")

# Ordinal encoding is used only for association analysis.
# It does not overwrite the original categorical columns.
ordinal_maps = {
    ai_col: {"Low": 1, "Medium": 2, "High": 3},
    sensitivity_col: {
        "Low": 1, "Medium": 2, "High": 3, "Critical": 4
    }
}

def get_numeric_series(col):
    """Use numeric values directly or a documented ordinal mapping."""
    if col in ordinal_maps:
        return df[col].map(ordinal_maps[col]).astype(float)
    return pd.to_numeric(df[col], errors="coerce")

def print_correlation(label, col_a, col_b):
    if col_a not in df.columns or col_b not in df.columns:
        print(f"{label}: required column not found.")
        return np.nan

    pair = pd.DataFrame({
        "a": get_numeric_series(col_a),
        "b": get_numeric_series(col_b)
    }).dropna()

    if len(pair) < 3 or pair["a"].nunique() < 2 or pair["b"].nunique() < 2:
        print(f"{label}: correlation unavailable for these data.")
        return np.nan

    r = pair["a"].corr(pair["b"])
    print(f"{label}: r = {r:.3f}; n = {len(pair)}")
    return r

correlation_results = {
    "AI usage vs. AI security risk":
        print_correlation("H1: AI usage vs. AI security risk", ai_col, ai_risk_col),
    "Data sensitivity vs. privacy risk":
        print_correlation("H1: Data sensitivity vs. privacy risk", sensitivity_col, privacy_col),
    "Access control vs. incident probability":
        print_correlation("H2: Access control vs. incident probability", access_col, incident_col),
    "Legacy dependency vs. incident probability":
        print_correlation("Legacy dependency vs. incident probability", legacy_col, incident_col),
    "Cryptographic agility vs. legacy dependency":
        print_correlation("H3: Cryptographic agility vs. legacy dependency", agility_col, legacy_col)
}
pd.Series(correlation_results, name="pearson_r").to_csv(
    os.path.join(OUTPUT_DIR, "hypothesis_correlations.csv")
)

# Colorful plot of selected hypothesis-related correlations.
valid_corr = pd.Series(correlation_results).dropna()
if not valid_corr.empty:
    plt.figure(figsize=(10, 6))
    colors = [
        "#00B894" if v >= 0 else "#FF7675"
        for v in valid_corr.values
    ]
    bars = plt.barh(valid_corr.index, valid_corr.values, color=colors)
    plt.axvline(0, color="#333333", linewidth=1)
    plt.xlim(-1, 1)
    plt.xlabel("Pearson correlation coefficient (r)")
    plt.title("Selected Relationships Relevant to Research Hypotheses")
    for bar, val in zip(bars, valid_corr.values):
        plt.text(
            val + (0.025 if val >= 0 else -0.025),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=9
        )
    save_figure("04b_hypothesis_correlations.png")

# ================================================================
# 6. POST-QUANTUM CRYPTOGRAPHY ANALYSIS
# ================================================================
print("\n" + "=" * 70)
print("POST-QUANTUM CRYPTOGRAPHY ANALYSIS")
print("=" * 70)

# Blank PQC cells remain missing. Literal "None" is a valid category.
# We report how many records have no recorded category rather than
# silently treating every missing cell as "None".
pqc_df = df.dropna(subset=[pqc_col]).copy()
print(f"\nProfiles with recorded PQC category: {len(pqc_df)}")
print(f"Profiles with missing PQC category: {df[pqc_col].isna().sum()}")
print("\nPQC adoption counts:\n", df[pqc_col].value_counts(dropna=False))

pqc_order = ["None", "Planned", "Partial", "Implemented"]
observed_pqc = [
    x for x in pqc_order if x in pqc_df[pqc_col].unique()
]
observed_pqc += [
    x for x in pqc_df[pqc_col].dropna().unique()
    if x not in observed_pqc
]

# 6.1 Numeric incident probability
incident_summary = (
    pqc_df.groupby(pqc_col, observed=True)[incident_col]
    .agg(["mean", "std", "count"])
    .reindex(observed_pqc)
)
print("\nPQC adoption vs. incident probability:\n", incident_summary)
incident_summary.to_csv(
    os.path.join(OUTPUT_DIR, "pqc_incident_summary.csv")
)

plt.figure(figsize=(9, 5.5))
ax = sns.barplot(
    data=pqc_df, x=pqc_col, y=incident_col,
    order=observed_pqc, hue=pqc_col,
    palette="bright", errorbar=("ci", 95), legend=False
)
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", padding=3, fontsize=9)
plt.title("PQC Adoption vs. Incident Probability")
plt.xlabel("PQC Adoption Level")
plt.ylabel("Mean Incident Probability")
plt.xticks(rotation=20, ha="right")
save_figure("05_pqc_vs_incident_probability.png")

# 6.2 Overall risk is categorical: use counts and row percentages,
# not mean(), which would cause a TypeError.
risk_counts = pd.crosstab(
    pqc_df[pqc_col], pqc_df[overall_col]
).reindex(observed_pqc)

risk_percentages = pd.crosstab(
    pqc_df[pqc_col], pqc_df[overall_col], normalize="index"
).reindex(observed_pqc) * 100

print("\nOverall risk counts by PQC adoption:\n", risk_counts)
print("\nOverall risk percentages within each PQC group:\n",
      risk_percentages.round(2))

risk_counts.to_csv(
    os.path.join(OUTPUT_DIR, "pqc_overall_risk_counts.csv")
)
risk_percentages.to_csv(
    os.path.join(OUTPUT_DIR, "pqc_overall_risk_percentages.csv")
)

risk_columns = [
    x for x in ["Low", "Medium", "High", "Critical"]
    if x in risk_percentages.columns
]
risk_colors = {
    "Low": "#00C853",
    "Medium": "#FFD600",
    "High": "#FF6D00",
    "Critical": "#AA00FF"
}

ax = risk_percentages[risk_columns].plot(
    kind="bar", stacked=True, figsize=(10, 6),
    color=[risk_colors[x] for x in risk_columns],
    edgecolor="white", linewidth=0.8
)
ax.set_title("Overall Risk Distribution by PQC Adoption")
ax.set_xlabel("PQC Adoption Level")
ax.set_ylabel("Organizations within PQC group (%)")
ax.set_ylim(0, 100)
ax.legend(title="Overall Risk Level", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.xticks(rotation=20, ha="right")
save_figure("05b_pqc_vs_overall_risk.png")

# ================================================================
# 7. SECURITY CONTROL ANALYSIS
# ================================================================
# These are associations in synthetic data, not proof of causation.
control_plots = [
    (access_col, incident_col,
     "Access Control Strength vs. Incident Probability",
     "06_access_control_vs_incidents.png", "#6C5CE7"),
    (monitor_col, incident_col,
     "Security Monitoring vs. Incident Probability",
     "07_monitoring_vs_incidents.png", "#00CEC9"),
    (encryption_col, incident_col,
     "Encryption Strength vs. Incident Probability",
     "08_encryption_vs_incidents.png", "#E84393"),
    (preservation_col, privacy_col,
     "Privacy Preservation vs. Privacy Risk",
     "09_privacy_preservation_vs_risk.png", "#F39C12"),
    (legacy_col, incident_col,
     "Legacy System Dependency vs. Incident Probability",
     "09b_legacy_vs_incidents.png", "#D63031"),
]

for control, outcome, title, filename, color in control_plots:
    if control in df.columns and outcome in df.columns:
        plt.figure(figsize=(8.5, 5.5))
        x = pd.to_numeric(df[control], errors="coerce")
        y = pd.to_numeric(df[outcome], errors="coerce")
        valid = pd.DataFrame({"x": x, "y": y}).dropna()

        if len(valid) >= 3 and valid["x"].nunique() > 1:
            sns.regplot(
                data=valid, x="x", y="y",
                scatter_kws={"alpha": 0.55, "s": 35, "color": color},
                line_kws={"linewidth": 2.5, "color": "#2D3436"},
                color=color
            )
            plt.xlabel(tidy_label(control))
            plt.ylabel(tidy_label(outcome))
            plt.title(title)
            save_figure(filename)

# ================================================================
# 8. MACHINE-LEARNING TARGET AND PREDICTORS
# ================================================================
# Target: existing categorical overall_risk_level.
# The target itself is excluded from predictors. Direct risk components
# and incident probability are excluded to reduce target leakage.
target_col = overall_col
y_raw = df[target_col].copy()
valid_target = y_raw.notna()
work = df.loc[valid_target].copy()
y = y_raw.loc[valid_target].astype(str).str.strip()

if y.nunique() < 2:
    raise ValueError("The target has fewer than two risk classes.")

leakage_terms = [
    "overall_risk_level",
    "overall_risk",
    "incident_probability",
    "ai_security_risk",
    "privacy_risk"
]
exclude = {
    c for c in work.columns
    if c == target_col or any(term == c for term in leakage_terms)
}
exclude.add("organization_id")

X = work.drop(columns=list(exclude), errors="ignore")
X = X.dropna(axis=1, how="all")
X = X.loc[:, X.nunique(dropna=False) > 1]

if X.shape[1] == 0:
    raise ValueError("No usable predictor columns remain.")

numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = [
    c for c in X.columns if c not in numeric_features
]

print("\n" + "=" * 70)
print("MACHINE-LEARNING SETUP")
print("=" * 70)
print("Target:", target_col)
print("\nTarget class counts:\n", y.value_counts())
print("\nPredictor columns:", X.columns.tolist())
print("Numeric predictors:", numeric_features)
print("Categorical predictors:", categorical_features)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_transformer, numeric_features),
    ("categorical", categorical_transformer, categorical_features)
])

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
}

# ================================================================
# 9. MODEL EVALUATION
# ================================================================
print("\n" + "=" * 70)
print("MACHINE-LEARNING MODEL COMPARISON")
print("=" * 70)

results = []
fitted_models = {}
predictions = {}
probabilities = {}
class_labels = np.array(sorted(y_train.unique()))

for name, estimator in models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator)
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    prob = pipe.predict_proba(X_test)

    fitted_models[name] = pipe
    predictions[name] = pred
    probabilities[name] = prob

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision_macro": precision_score(
            y_test, pred, average="macro", zero_division=0
        ),
        "Recall_macro": recall_score(
            y_test, pred, average="macro", zero_division=0
        ),
        "F1_macro": f1_score(
            y_test, pred, average="macro", zero_division=0
        )
    }

    try:
        if len(class_labels) == 2:
            metrics["ROC_AUC"] = roc_auc_score(
                y_test, prob[:, 1],
                labels=class_labels
            )
        else:
            metrics["ROC_AUC"] = roc_auc_score(
                y_test, prob,
                labels=pipe.named_steps["model"].classes_,
                multi_class="ovr",
                average="macro"
            )
    except ValueError:
        metrics["ROC_AUC"] = np.nan

    results.append(metrics)

    print(f"\n{name}")
    print(classification_report(y_test, pred, zero_division=0))

results_df = pd.DataFrame(results).set_index("Model")
print("\nMetric comparison:\n", results_df.round(4))
results_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"))

# Colorful grouped bar chart of model metrics.
metric_cols = [
    "Accuracy", "Precision_macro", "Recall_macro", "F1_macro", "ROC_AUC"
]
plot_metrics = results_df[metric_cols]
ax = plot_metrics.plot(
    kind="bar", figsize=(12, 6),
    color=["#00B894", "#0984E3", "#6C5CE7", "#E84393", "#FDCB6E"],
    edgecolor="white", linewidth=0.8
)
ax.set_ylim(0, 1.08)
ax.set_title("Performance Comparison of Machine-Learning Models")
ax.set_ylabel("Score")
ax.set_xlabel("Classification Model")
ax.legend(
    title="Evaluation Metric",
    loc="upper center",
    bbox_to_anchor=(0.5, -0.12),
    ncol=3
)
plt.xticks(rotation=0)
save_figure("10_model_performance_comparison.png")

# ================================================================
# 10. CONFUSION MATRICES
# ================================================================
risk_class_order = [
    c for c in ["Low", "Medium", "High", "Critical"]
    if c in set(y_test) | set(y_train)
]
if not risk_class_order:
    risk_class_order = list(class_labels)

for name, pred in predictions.items():
    cm = confusion_matrix(y_test, pred, labels=risk_class_order)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="magma",
        xticklabels=risk_class_order,
        yticklabels=risk_class_order,
        linewidths=1, linecolor="white",
        cbar_kws={"label": "Number of profiles"},
        ax=ax
    )
    ax.set_title(f"Confusion Matrix — {name}")
    ax.set_xlabel("Predicted Risk Level")
    ax.set_ylabel("Actual Risk Level")
    save_figure(
        "11_confusion_matrix_" + name.lower().replace(" ", "_") + ".png"
    )

# ================================================================
# 11. ROC CURVES
# ================================================================
for name in fitted_models:
    pipe = fitted_models[name]
    prob = probabilities[name]
    model_classes = pipe.named_steps["model"].classes_

    plt.figure(figsize=(8, 6))
    if len(model_classes) == 2:
        positive_label = model_classes[1]
        y_binary = (np.asarray(y_test) == positive_label).astype(int)
        fpr, tpr, _ = roc_curve(y_binary, prob[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(
            fpr, tpr, linewidth=3,
            color="#E84393", label=f"{positive_label} (AUC = {roc_auc:.3f})"
        )
    else:
        y_bin = label_binarize(y_test, classes=model_classes)
        colors = sns.color_palette("husl", n_colors=len(model_classes))
        for i, label in enumerate(model_classes):
            if len(np.unique(y_bin[:, i])) == 2:
                fpr, tpr, _ = roc_curve(y_bin[:, i], prob[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(
                    fpr, tpr, linewidth=2.5, color=colors[i],
                    label=f"{label} (AUC = {roc_auc:.3f})"
                )

    plt.plot(
        [0, 1], [0, 1], linestyle="--",
        color="#636E72", linewidth=1.5, label="Random classifier"
    )
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {name}")
    plt.legend(loc="lower right")
    save_figure(
        "12_roc_curve_" + name.lower().replace(" ", "_") + ".png"
    )

# ================================================================
# 12. RANDOM FOREST FEATURE IMPORTANCE
# ================================================================
rf_pipe = fitted_models["Random Forest"]
rf_model = rf_pipe.named_steps["model"]
feature_names = rf_pipe.named_steps[
    "preprocessor"
].get_feature_names_out()

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": rf_model.feature_importances_
}).sort_values("Importance", ascending=False)

importance_df.to_csv(
    os.path.join(OUTPUT_DIR, "random_forest_feature_importance.csv"),
    index=False
)

top_features = importance_df.head(15).sort_values(
    "Importance", ascending=True
)
plt.figure(figsize=(10, 7.5))
sns.barplot(
    data=top_features, x="Importance", y="Feature",
    hue="Feature", palette="turbo", legend=False
)
plt.title("Top Predictors of Overall Risk Classification")
plt.xlabel("Random Forest Feature Importance")
plt.ylabel("Predictor")
save_figure("13_random_forest_feature_importance.png")

print("\nTop Random Forest features:\n",
      importance_df.head(15).to_string(index=False))

# ================================================================
# 13. EXPORT PREDICTIONS AND EXPERIMENT SUMMARY
# ================================================================
prediction_df = pd.DataFrame({
    "Actual": np.asarray(y_test)
}, index=y_test.index)

for name in fitted_models:
    key = name.lower().replace(" ", "_")
    prediction_df[key + "_predicted"] = predictions[name]

prediction_df.to_csv(
    os.path.join(OUTPUT_DIR, "test_set_predictions.csv"),
    index_label="row_index"
)

with open(
    os.path.join(OUTPUT_DIR, "experiment_summary.txt"),
    "w",
    encoding="utf-8"
) as f:
    f.write("DATA SECURITY AND PRIVACY RESEARCH EXPERIMENT\n")
    f.write(f"Dataset rows after cleaning: {len(df)}\n")
    f.write(f"Dataset columns: {len(df.columns)}\n")
    f.write(f"Target variable: {target_col}\n")
    f.write(f"Training rows: {len(X_train)}\n")
    f.write(f"Testing rows: {len(X_test)}\n")
    f.write(f"Missing PQC values: {df[pqc_col].isna().sum()}\n")
    f.write(f"Predictors: {', '.join(X.columns)}\n\n")
    f.write("MODEL METRICS\n")
    f.write(results_df.round(4).to_string())
    f.write("\n\nHYPOTHESIS-RELATED CORRELATIONS\n")
    f.write(pd.Series(correlation_results).to_string())

print("\n" + "=" * 70)
print("ANALYSIS COMPLETED SUCCESSFULLY")
print("All charts and CSV outputs saved in:")
print(OUTPUT_DIR)
print("=" * 70)
