# %%
import sys
import os

# Add the lib directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib"))

from lib.data_simulation import DataSimulator
from InterClassInterpolator.code.lib.ICI import ICI_harmonization
from lib.utils import ModelEvaluator, ResultAnalyzer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt

random_state = 23
# Initialize components
data_simulator = DataSimulator(random_state=random_state)

interpolator = SMOTE(random_state=random_state, k_neighbors=10)
smote_corrector = ICI_harmonization(interpolator=interpolator)

model_evaluator = ModelEvaluator(random_state=random_state)
result_analyzer = ResultAnalyzer()

# Define models
models = LogisticRegression(random_state=random_state, max_iter=1000)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=random_state)

scaler = StandardScaler()

site_effect_strength = 0.7
signal_strength = 0.3
noise_strength = 0.01
n_features = 10
balance = [0.8, 0.2]


print("\n=== SCENARIO 1: Generating Imbalanced Multi-site Data Only site effect ===")
X, Y, sites = data_simulator.simulate_multi_site_data_advanced(
    n_sites=2,
    n_samples=1000,
    balance_per_site=balance,
    n_features=n_features,
    site_effect_strength=site_effect_strength,
    signal_strength=0,
)


# Get statistics
stats = data_simulator.get_data_statistics(X, Y, sites)
print("Data statistics:")
print(f"  Total samples: {stats['n_samples']}")
print(f"  Features: {stats['n_features']}")
print(f"  Sites: {stats['n_sites']}")
print(f"  Overall class balance: {stats['overall_class_balance']:.3f}")
print(f"  Site-target correlation: {stats['site_target_correlation']:.3f}")

for site, site_stats in stats["site_statistics"].items():
    print(
        f"  {site}: balance={site_stats['class_balance']:.3f}, "
        f"samples={site_stats['n_samples']} (0:{site_stats['n_class_0']}, 1:{site_stats['n_class_1']})"
    )

# Evaluate models on imbalanced data
print("\n=== SCENARIO 1: Evaluating on Imbalanced Data (No SMOTE) ===")
scores = []

for train_idx, test_idx in cv.split(X, Y):
    # Split data
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = Y[train_idx], Y[test_idx]
    site_train, site_test = sites[train_idx], sites[test_idx]

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train and evaluate the original model
    models.fit(X_train, y_train)

    y_pred_proba = models.predict_proba(X_test)[:, 1]
    score = roc_auc_score(y_test, y_pred_proba)
    scores.append([score, "Only site effect", "Unharmonized", "Test", "Unbalanced"])

    y_pred_proba_train = models.predict_proba(X_train)[:, 1]
    score_train = roc_auc_score(y_train, y_pred_proba_train)
    scores.append([score, "Only site effect", "Unharmonized", "Train", "Unbalanced"])
    # Apply SMOTE only to training data if requested
    X_train_balanced, y_train_balanced, site_train_balanced = smote_corrector.fit_resample(
        X_train, y_train, site_train
    )

    # Train and evaluate model
    models.fit(X_train_balanced, y_train_balanced)

    y_pred_proba = models.predict_proba(X_test)[:, 1]
    score = roc_auc_score(y_test, y_pred_proba)
    scores.append([score, "Only site effect", "ICI harmonized", "Test", "Unbalanced"])

    y_pred_proba_train = models.predict_proba(X_train)[:, 1]
    score_train = roc_auc_score(y_train, y_pred_proba_train)
    scores.append([score, "Only site effect", "ICI harmonized", "Train", "Unbalanced"])
# # Scenario 2: Balance data


print("\n=== SCENARIO 2: Generating Imbalanced Multi-site Data Only real effect ===")

X, Y, sites = data_simulator.simulate_multi_site_data_advanced(
    n_sites=2,
    n_samples=1000,
    balance_per_site=balance,
    n_features=n_features,
    site_effect_strength=0,
    signal_strength=signal_strength,
)


# Get statistics
stats = data_simulator.get_data_statistics(X, Y, sites)
print("Data statistics:")
print(f"  Total samples: {stats['n_samples']}")
print(f"  Features: {stats['n_features']}")
print(f"  Sites: {stats['n_sites']}")
print(f"  Overall class balance: {stats['overall_class_balance']:.3f}")
print(f"  Site-target correlation: {stats['site_target_correlation']:.3f}")


for train_idx, test_idx in cv.split(X, Y):
    # Split data
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = Y[train_idx], Y[test_idx]
    site_train, site_test = sites[train_idx], sites[test_idx]

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # Train and evaluate the original model
    models.fit(X_train, y_train)

    y_pred_proba = models.predict_proba(X_test)[:, 1]
    score = roc_auc_score(y_test, y_pred_proba)
    scores.append([score, "Only real signal", "Unharmonized", "Test", "Unbalanced"])

    y_pred_proba_train = models.predict_proba(X_train)[:, 1]
    score_train = roc_auc_score(y_train, y_pred_proba_train)
    scores.append([score, "Only real signal", "Unharmonized", "Train", "Unbalanced"])

    # Apply SMOTE only to training data if requested
    X_train_balanced, y_train_balanced, site_train_balanced = smote_corrector.fit_resample(
        X_train, y_train, site_train
    )

    # Train and evaluate model
    models.fit(X_train_balanced, y_train_balanced)

    y_pred_proba = models.predict_proba(X_test)[:, 1]
    score = roc_auc_score(y_test, y_pred_proba)
    scores.append([score, "Only real signal", "ICI harmonized", "Test", "Unbalanced"])

    y_pred_proba_train = models.predict_proba(X_train)[:, 1]
    score_train = roc_auc_score(y_train, y_pred_proba_train)
    scores.append([score, "Only real signal", "ICI harmonized", "Train", "Unbalanced"])


print("\n=== SCENARIO 2: Generating Imbalanced Multi-site Data Only real effect ===")


print("\n=== SCENARIO 1: Generating Imbalanced Multi-site Data Only site effect ===")
X, Y, sites = data_simulator.simulate_multi_site_data_advanced(
    n_sites=2,
    n_samples=1000,
    balance_per_site=balance,
    n_features=n_features,
    site_effect_strength=site_effect_strength,
    signal_strength=signal_strength,
)


# Get statistics
stats = data_simulator.get_data_statistics(X, Y, sites)
print("Data statistics:")
print(f"  Total samples: {stats['n_samples']}")
print(f"  Features: {stats['n_features']}")
print(f"  Sites: {stats['n_sites']}")
print(f"  Overall class balance: {stats['overall_class_balance']:.3f}")
print(f"  Site-target correlation: {stats['site_target_correlation']:.3f}")


for train_idx, test_idx in cv.split(X, Y):
    # Split data
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = Y[train_idx], Y[test_idx]
    site_train, site_test = sites[train_idx], sites[test_idx]

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train and evaluate the original model
    models.fit(X_train, y_train)

    y_pred_proba = models.predict_proba(X_test)[:, 1]
    score = roc_auc_score(y_test, y_pred_proba)
    scores.append([score, "Combined signal", "Unharmonized", "Test", "Unbalanced"])

    y_pred_proba_train = models.predict_proba(X_train)[:, 1]
    score_train = roc_auc_score(y_train, y_pred_proba_train)
    scores.append([score, "Combined signal", "Unharmonized", "Train", "Unbalanced"])
    # Apply SMOTE only to training data if requested
    X_train_balanced, y_train_balanced, site_train_balanced = smote_corrector.fit_resample(
        X_train, y_train, site_train
    )

    # Train and evaluate model
    models.fit(X_train_balanced, y_train_balanced)

    y_pred_proba = models.predict_proba(X_test)[:, 1]
    score = roc_auc_score(y_test, y_pred_proba)
    scores.append([score, "Combined signal", "ICI harmonized", "Test", "Unbalanced"])

    y_pred_proba_train = models.predict_proba(X_train)[:, 1]
    score_train = roc_auc_score(y_train, y_pred_proba_train)
    scores.append([score, "Combined signal", "ICI harmonized", "Train", "Unbalanced"])


scores = pd.DataFrame(
    scores, columns=["AUC", "Signal", "Harmonization", "Dataset", "Balanced"]
)


scores_test_balance = scores[scores["Dataset"] == "Test"]


# %%

balance = [0.5, 0.5]


print("\n=== SCENARIO 1: Generating Imbalanced Multi-site Data Only site effect ===")
X, Y, sites = data_simulator.simulate_multi_site_data_advanced(
    n_sites=2,
    n_samples=1000,
    balance_per_site=balance,
    n_features=n_features,
    site_effect_strength=site_effect_strength,
    signal_strength=0,
)


# Get statistics
stats = data_simulator.get_data_statistics(X, Y, sites)
print("Data statistics:")
print(f"  Total samples: {stats['n_samples']}")
print(f"  Features: {stats['n_features']}")
print(f"  Sites: {stats['n_sites']}")
print(f"  Overall class balance: {stats['overall_class_balance']:.3f}")
print(f"  Site-target correlation: {stats['site_target_correlation']:.3f}")

for site, site_stats in stats["site_statistics"].items():
    print(
        f"  {site}: balance={site_stats['class_balance']:.3f}, "
        f"samples={site_stats['n_samples']} (0:{site_stats['n_class_0']}, 1:{site_stats['n_class_1']})"
    )

# Evaluate models on imbalanced data
print("\n=== SCENARIO 1: Evaluating on Imbalanced Data (No SMOTE) ===")
scores = []

for train_idx, test_idx in cv.split(X, Y):
    # Split data
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = Y[train_idx], Y[test_idx]
    site_train, site_test = sites[train_idx], sites[test_idx]

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train and evaluate the original model
    models.fit(X_train, y_train)

    y_pred_proba = models.predict_proba(X_test)[:, 1]
    score = roc_auc_score(y_test, y_pred_proba)
    scores.append([score, "Only site effect", "Unharmonized", "Test", "Unbalanced"])

    y_pred_proba_train = models.predict_proba(X_train)[:, 1]
    score_train = roc_auc_score(y_train, y_pred_proba_train)
    scores.append([score, "Only site effect", "Unharmonized", "Train", "Unbalanced"])
    # Apply SMOTE only to training data if requested
    X_train_balanced, y_train_balanced, site_train_balanced = smote_corrector.fit_resample(
        X_train, y_train, site_train
    )

    # Train and evaluate model
    models.fit(X_train_balanced, y_train_balanced)

    y_pred_proba = models.predict_proba(X_test)[:, 1]
    score = roc_auc_score(y_test, y_pred_proba)
    scores.append([score, "Only site effect", "ICI harmonized", "Test", "Unbalanced"])

    y_pred_proba_train = models.predict_proba(X_train)[:, 1]
    score_train = roc_auc_score(y_train, y_pred_proba_train)
    scores.append([score, "Only site effect", "ICI harmonized", "Train", "Unbalanced"])
# # Scenario 2: Balance data


print("\n=== SCENARIO 2: Generating Imbalanced Multi-site Data Only real effect ===")

X, Y, sites = data_simulator.simulate_multi_site_data_advanced(
    n_sites=2,
    n_samples=1000,
    balance_per_site=balance,
    n_features=n_features,
    site_effect_strength=0,
    signal_strength=signal_strength,
)


# Get statistics
stats = data_simulator.get_data_statistics(X, Y, sites)
print("Data statistics:")
print(f"  Total samples: {stats['n_samples']}")
print(f"  Features: {stats['n_features']}")
print(f"  Sites: {stats['n_sites']}")
print(f"  Overall class balance: {stats['overall_class_balance']:.3f}")
print(f"  Site-target correlation: {stats['site_target_correlation']:.3f}")


for train_idx, test_idx in cv.split(X, Y):
    # Split data
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = Y[train_idx], Y[test_idx]
    site_train, site_test = sites[train_idx], sites[test_idx]

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # Train and evaluate the original model
    models.fit(X_train, y_train)

    y_pred_proba = models.predict_proba(X_test)[:, 1]
    score = roc_auc_score(y_test, y_pred_proba)
    scores.append([score, "Only real signal", "Unharmonized", "Test", "Unbalanced"])

    y_pred_proba_train = models.predict_proba(X_train)[:, 1]
    score_train = roc_auc_score(y_train, y_pred_proba_train)
    scores.append([score, "Only real signal", "Unharmonized", "Train", "Unbalanced"])

    # Apply SMOTE only to training data if requested
    X_train_balanced, y_train_balanced, site_train_balanced = smote_corrector.fit_resample(
        X_train, y_train, site_train
    )

    # Train and evaluate model
    models.fit(X_train_balanced, y_train_balanced)

    y_pred_proba = models.predict_proba(X_test)[:, 1]
    score = roc_auc_score(y_test, y_pred_proba)
    scores.append([score, "Only real signal", "ICI harmonized", "Test", "Unbalanced"])

    y_pred_proba_train = models.predict_proba(X_train)[:, 1]
    score_train = roc_auc_score(y_train, y_pred_proba_train)
    scores.append([score, "Only real signal", "ICI harmonized", "Train", "Unbalanced"])


print("\n=== SCENARIO 2: Generating Imbalanced Multi-site Data Only real effect ===")


print("\n=== SCENARIO 1: Generating Imbalanced Multi-site Data Only site effect ===")
X, Y, sites = data_simulator.simulate_multi_site_data_advanced(
    n_sites=2,
    n_samples=1000,
    balance_per_site=balance,
    n_features=n_features,
    site_effect_strength=site_effect_strength,
    signal_strength=signal_strength,
)


# Get statistics
stats = data_simulator.get_data_statistics(X, Y, sites)
print("Data statistics:")
print(f"  Total samples: {stats['n_samples']}")
print(f"  Features: {stats['n_features']}")
print(f"  Sites: {stats['n_sites']}")
print(f"  Overall class balance: {stats['overall_class_balance']:.3f}")
print(f"  Site-target correlation: {stats['site_target_correlation']:.3f}")


for train_idx, test_idx in cv.split(X, Y):
    # Split data
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = Y[train_idx], Y[test_idx]
    site_train, site_test = sites[train_idx], sites[test_idx]

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train and evaluate the original model
    models.fit(X_train, y_train)

    y_pred_proba = models.predict_proba(X_test)[:, 1]
    score = roc_auc_score(y_test, y_pred_proba)
    scores.append([score, "Combined signal", "Unharmonized", "Test", "Unbalanced"])

    y_pred_proba_train = models.predict_proba(X_train)[:, 1]
    score_train = roc_auc_score(y_train, y_pred_proba_train)
    scores.append([score, "Combined signal", "Unharmonized", "Train", "Unbalanced"])
    # Apply SMOTE only to training data if requested
    X_train_balanced, y_train_balanced, site_train_balanced = smote_corrector.fit_resample(
        X_train, y_train, site_train
    )

    # Train and evaluate model
    models.fit(X_train_balanced, y_train_balanced)

    y_pred_proba = models.predict_proba(X_test)[:, 1]
    score = roc_auc_score(y_test, y_pred_proba)
    scores.append([score, "Combined signal", "ICI harmonized", "Test", "Unbalanced"])

    y_pred_proba_train = models.predict_proba(X_train)[:, 1]
    score_train = roc_auc_score(y_train, y_pred_proba_train)
    scores.append([score, "Combined signal", "ICI harmonized", "Train", "Unbalanced"])


scores = pd.DataFrame(
    scores, columns=["AUC", "Signal", "Harmonization", "Dataset", "Balanced"]
)
scores_test = scores[scores["Dataset"] == "Test"]

# %%

# -------------------------
# Data preparation
# -------------------------

# Rename for conceptual clarity
scores_test["Preprocessing"] = scores_test["Harmonization"].replace(
    {
        "Unharmonized": "Unbalanced",
        "ICI harmonized": "ICI-balanced",
        "ICI": "ICI-balanced",
    }
)

scores_test["Signal"] = scores_test["Signal"].replace(
    {
        "Only real signal": "Only True signal",
        "Only site effect": "Only EoS signal",
        "Combined signal": "Combined Signal",
    }
)


# Rename for conceptual clarity
scores_test_balance["Preprocessing"] = scores_test_balance["Harmonization"].replace(
    {
        "Unharmonized": "Unbalanced",

        "ICI harmonized": "ICI-balanced",
        "ICI": "ICI-balanced",
    }
)

# Rename for conceptual clarity
scores_test_balance["Signal"] = scores_test_balance["Signal"].replace(
    {
        "Only real signal": "Only True signal",
        "Only site effect": "Only EoS signal",
        "Combined signal": "Combined Signal",
    }
)

metric_to_plot = "AUC"

# -------------------------
# Plot configuration
# -------------------------
palette = {
    "Unbalanced": "#1f77b4",
    "ICI-balanced": "#ff7f0e",
}

ylim = (0.3, 0.9)

fig, axes = plt.subplots(2, 1, figsize=(15, 14), sharex=True, constrained_layout=True)

panels = [
    (
        axes[1],
        scores_test,
        "Site and target are independent",
        "(B)",
    ),
    (
        axes[0],
        scores_test_balance,
        "Model AUC under different signals and preprocessing strategies\nSite and target are dependent",
        "(A)",
    ),
]

# -------------------------
# Plot panels
# -------------------------
for ax, data, title, panel_label in panels:
    sbn.swarmplot(
        x="Signal",
        y=metric_to_plot,
        data=data,
        hue="Preprocessing",
        dodge=True,
        palette=palette,
        alpha=0.8,
        size=4,
        ax=ax,
    )

    handles, labels = ax.get_legend_handles_labels()

    sbn.boxplot(
        x="Signal",
        y=metric_to_plot,
        data=data,
        hue="Preprocessing",
        dodge=True,

        showcaps=True,
        boxprops={"facecolor": "none"},
        showfliers=False,
        whiskerprops={"linewidth": 1},
        ax=ax,
    )

    # Remove duplicate legend and restore single clean legend
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.legend(handles, labels, title="Preprocessing", loc="best")

    # Chance level
    ax.axhline(0.5, linestyle="--", color="gray", linewidth=1)

    # Formatting
    ax.set_ylim(*ylim)
    ax.set_ylabel(metric_to_plot)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.4, linestyle="-")

    # Panel label
    ax.text(
        -0.15,
        1.05,
        panel_label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="left",
    )

axes[1].set_xlabel("Signal")
axes[0].set_xlabel("")

plt.show()

# %%
