# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from scipy import stats
from typing import Tuple, List, Dict, Any
import warnings

warnings.filterwarnings("ignore")


def simulate_multi_site_data(
    n_sites: int = 2,
    n_samples: int = 1000,
    balance_per_site: List[float] = [0.5, 0.5],
    n_features: int = 10,
    random_state: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate multi-site data with site effects and class imbalance

    Parameters:
    -----------
    n_sites : int
        Number of sites
    n_samples : int
        Total number of samples
    balance_per_site : List[float]
        List of class balances for each site (proportion of class 1)
    n_features : int
        Number of features
    random_state : int, optional
        Random seed for reproducibility

    Returns:
    --------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target labels of shape (n_samples,)
    site_labels : np.ndarray
        Site labels of shape (n_samples,)
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Calculate samples per site
    samples_per_site = n_samples // n_sites
    remainder = n_samples % n_sites

    X = []
    y = []
    site_labels = []

    for site in range(n_sites):
        # Determine number of samples for this site
        site_samples = samples_per_site + (1 if site < remainder else 0)

        # Generate site-specific effect (different mean for each site)
        site_effect = np.random.normal(site * 3, 1.5, n_features)

        # Generate class labels based on balance
        n_class1 = int(site_samples * balance_per_site[site])
        n_class0 = site_samples - n_class1

        # Create labels
        site_y = np.concatenate([np.zeros(n_class0), np.ones(n_class1)])

        # Generate features: noise + site effect
        site_X = np.random.normal(0, 1, (site_samples, n_features))

        # Add site effect to all samples
        site_X += site_effect

        # Add small class effect
        class_effect_strength = 0.3
        class_effect_pattern = np.random.normal(0, 1, n_features)
        class_effect_pattern = class_effect_pattern / np.linalg.norm(
            class_effect_pattern
        )

        # Apply class effect only to class 1 samples
        class_1_mask = site_y == 1
        site_X[class_1_mask] += class_effect_strength * class_effect_pattern

        X.append(site_X)
        y.append(site_y)
        site_labels.extend([site] * site_samples)

    X = np.vstack(X)
    y = np.concatenate(y)
    site_labels = np.array(site_labels)

    return X, y, site_labels


def apply_smote_per_site_to_training(
    X_train: np.ndarray,
    y_train: np.ndarray,
    site_train: np.ndarray,
    k_neighbors: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply SMOTE separately to each site in the training data to balance classes to 0.5

    Parameters:
    -----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    site_train : np.ndarray
        Site labels for training data
    k_neighbors : int
        Number of neighbors for SMOTE

    Returns:
    --------
    X_train_resampled : np.ndarray
        Resampled training features
    y_train_resampled : np.ndarray
        Resampled training labels
    site_train_resampled : np.ndarray
        Resampled site labels
    """
    X_resampled = []
    y_resampled = []
    site_resampled = []

    unique_sites = np.unique(site_train)

    for site in unique_sites:
        # Get data for current site
        site_mask = site_train == site
        X_site = X_train[site_mask]
        y_site = y_train[site_mask]

        # Check class distribution
        n_class0 = np.sum(y_site == 0)
        n_class1 = np.sum(y_site == 1)

        # Only apply SMOTE if there's imbalance and we have enough samples
        if n_class0 != n_class1 and min(n_class0, n_class1) > 0:
            # Determine which class is minority
            if n_class0 < n_class1:
                minority_class = 0
            else:
                minority_class = 1

            # Apply SMOTE to balance classes
            smote = SMOTE(
                sampling_strategy="auto",
                k_neighbors=min(k_neighbors, min(n_class0, n_class1) - 1),
                random_state=42,
            )

            X_site_resampled, y_site_resampled = smote.fit_resample(X_site, y_site)
        else:
            # No resampling needed if balanced or only one class
            X_site_resampled, y_site_resampled = X_site, y_site

        X_resampled.append(X_site_resampled)
        y_resampled.append(y_site_resampled)
        site_resampled.extend([site] * len(X_site_resampled))

    X_resampled = np.vstack(X_resampled)
    y_resampled = np.concatenate(y_resampled)
    site_resampled = np.array(site_resampled)

    return X_resampled, y_resampled, site_resampled


def evaluate_model_with_smote(
    X: np.ndarray,
    y: np.ndarray,
    site_labels: np.ndarray,
    model: Any,
    scaler: StandardScaler = None,
    use_smote: bool = False,
    k_neighbors: int = 5,
) -> List[float]:
    """
    Evaluate model using repeated stratified k-fold cross validation with proper SMOTE application

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target labels
    site_labels : np.ndarray
        Site labels
    model : sklearn classifier
        Model to evaluate
    scaler : StandardScaler, optional
        Scaler for features (used only for Logistic Regression)
    use_smote : bool
        Whether to apply SMOTE to training data
    k_neighbors : int
        Number of neighbors for SMOTE

    Returns:
    --------
    scores : List[float]
        List of AUC scores for each fold
    """
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)
    scores = []

    for train_idx, test_idx in cv.split(X, y):
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        site_train, site_test = site_labels[train_idx], site_labels[test_idx]

        # Apply SMOTE only to training data if requested
        if use_smote:
            X_train_processed, y_train_processed, site_train_processed = (
                apply_smote_per_site_to_training(
                    X_train, y_train, site_train, k_neighbors
                )
            )
        else:
            X_train_processed, y_train_processed, site_train_processed = (
                X_train,
                y_train,
                site_train,
            )

        # Scale features if scaler is provided (for Logistic Regression)
        if scaler is not None:
            X_train_processed = scaler.fit_transform(X_train_processed)
            X_test_processed = scaler.transform(X_test)
        else:
            X_train_processed = X_train_processed
            X_test_processed = X_test

        # Train and evaluate model
        model_instance = model.__class__(**model.get_params())
        model_instance.fit(X_train_processed, y_train_processed)

        # Handle models that don't have predict_proba
        if hasattr(model_instance, "predict_proba"):
            y_pred_proba = model_instance.predict_proba(X_test_processed)[:, 1]
        else:
            y_pred_proba = model_instance.decision_function(X_test_processed)
            # Convert to probability-like scores
            y_pred_proba = 1 / (1 + np.exp(-y_pred_proba))

        score = roc_auc_score(y_test, y_pred_proba)
        scores.append(score)

    return scores


def run_experiments() -> Tuple[
    Dict[str, Dict[str, List[float]]], Tuple, Tuple, Tuple, Tuple
]:
    """
    Run all experimental scenarios and compare results

    Returns:
    --------
    results : Dict
        Dictionary containing AUC scores for all scenarios and models
    imbalanced_data : Tuple
        (X, y, site_labels) for imbalanced scenario
    balanced_data : Tuple
        (X, y, site_labels) for balanced scenario
    smote5_data : Tuple
        (X, y, site_labels) for SMOTE k=5 scenario (reference only)
    smote10_data : Tuple
        (X, y, site_labels) for SMOTE k=10 scenario (reference only)
    """
    results = {}

    # Models to compare
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    }

    scaler = StandardScaler()

    print("=== SCENARIO 1: Imbalanced Multi-site Data (No SMOTE) ===")
    X_imbalanced, y_imbalanced, site_imbalanced = simulate_multi_site_data(
        n_sites=2,
        n_samples=1000,
        balance_per_site=[0.8, 0.2],
        n_features=10,
        random_state=42,
    )

    print(
        f"Class distribution - Site 0: {np.mean(y_imbalanced[site_imbalanced == 0]):.3f} "
        f"(n_class1={np.sum(y_imbalanced[site_imbalanced == 0] == 1)}, n_class0={np.sum(y_imbalanced[site_imbalanced == 0] == 0)})"
    )
    print(
        f"Class distribution - Site 1: {np.mean(y_imbalanced[site_imbalanced == 1]):.3f} "
        f"(n_class1={np.sum(y_imbalanced[site_imbalanced == 1] == 1)}, n_class0={np.sum(y_imbalanced[site_imbalanced == 1] == 0)})"
    )
    print(f"Overall class distribution: {np.mean(y_imbalanced):.3f}")
    print(f"Total samples: {len(y_imbalanced)}")
    print(
        f"Site-target correlation: {np.corrcoef(site_imbalanced, y_imbalanced)[0, 1]:.3f}"
    )

    scenario1_results = {}
    for name, model in models.items():
        scores = evaluate_model_with_smote(
            X_imbalanced,
            y_imbalanced,
            site_imbalanced,
            model,
            scaler if "Logistic" in name else None,
            use_smote=False,
        )
        scenario1_results[name] = scores
        print(f"{name}: AUC = {np.mean(scores):.3f} ± {np.std(scores):.3f}")

    results["Imbalanced"] = scenario1_results

    print("\n=== SCENARIO 2: Balanced Multi-site Data (No SMOTE) ===")
    X_balanced, y_balanced, site_balanced = simulate_multi_site_data(
        n_sites=2,
        n_samples=1000,
        balance_per_site=[0.5, 0.5],
        n_features=10,
        random_state=43,
    )

    print(
        f"Class distribution - Site 0: {np.mean(y_balanced[site_balanced == 0]):.3f} "
        f"(n_class1={np.sum(y_balanced[site_balanced == 0] == 1)}, n_class0={np.sum(y_balanced[site_balanced == 0] == 0)})"
    )
    print(
        f"Class distribution - Site 1: {np.mean(y_balanced[site_balanced == 1]):.3f} "
        f"(n_class1={np.sum(y_balanced[site_balanced == 1] == 1)}, n_class0={np.sum(y_balanced[site_balanced == 1] == 0)})"
    )
    print(f"Overall class distribution: {np.mean(y_balanced):.3f}")
    print(f"Total samples: {len(y_balanced)}")
    print(
        f"Site-target correlation: {np.corrcoef(site_balanced, y_balanced)[0, 1]:.3f}"
    )

    scenario2_results = {}
    for name, model in models.items():
        scores = evaluate_model_with_smote(
            X_balanced,
            y_balanced,
            site_balanced,
            model,
            scaler if "Logistic" in name else None,
            use_smote=False,
        )
        scenario2_results[name] = scores
        print(f"{name}: AUC = {np.mean(scores):.3f} ± {np.std(scores):.3f}")

    results["Balanced"] = scenario2_results

    print(
        "\n=== SCENARIO 3: Imbalanced Data with SMOTE k=5 (Applied only to training) ==="
    )
    scenario3_results = {}
    for name, model in models.items():
        scores = evaluate_model_with_smote(
            X_imbalanced,
            y_imbalanced,
            site_imbalanced,
            model,
            scaler if "Logistic" in name else None,
            use_smote=True,
            k_neighbors=5,
        )
        scenario3_results[name] = scores
        print(f"{name}: AUC = {np.mean(scores):.3f} ± {np.std(scores):.3f}")

    results["SMOTE_k5"] = scenario3_results

    print(
        "\n=== SCENARIO 4: Imbalanced Data with SMOTE k=10 (Applied only to training) ==="
    )
    scenario4_results = {}
    for name, model in models.items():
        scores = evaluate_model_with_smote(
            X_imbalanced,
            y_imbalanced,
            site_imbalanced,
            model,
            scaler if "Logistic" in name else None,
            use_smote=True,
            k_neighbors=10,
        )
        scenario4_results[name] = scores
        print(f"{name}: AUC = {np.mean(scores):.3f} ± {np.std(scores):.3f}")

    results["SMOTE_k10"] = scenario4_results

    return (
        results,
        (X_imbalanced, y_imbalanced, site_imbalanced),
        (X_balanced, y_balanced, site_balanced),
        (X_imbalanced, y_imbalanced, site_imbalanced),
        (X_imbalanced, y_imbalanced, site_imbalanced),
    )


def plot_results(results: Dict[str, Dict[str, List[float]]]) -> None:
    """
    Create boxplots and statistical comparisons of results

    Parameters:
    -----------
    results : Dict
        Dictionary containing AUC scores for all scenarios and models
    """
    # Prepare data for plotting
    plot_data = []
    for scenario, models in results.items():
        for model_name, scores in models.items():
            for score in scores:
                plot_data.append(
                    {"Scenario": scenario, "Model": model_name, "AUC": score}
                )

    df = pd.DataFrame(plot_data)

    # Create boxplot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Boxplot by scenario
    sns.boxplot(data=df, x="Scenario", y="AUC", hue="Model", ax=ax1)
    ax1.set_title(
        "Model Performance Across Scenarios (1000 samples, 10x10 CV)\nSMOTE applied only to training data"
    )
    ax1.set_ylabel("AUC Score")
    ax1.legend(title="Model")
    ax1.tick_params(axis="x", rotation=45)

    # Boxplot by model
    sns.boxplot(data=df, x="Model", y="AUC", hue="Scenario", ax=ax2)
    ax2.set_title(
        "Scenario Performance Across Models (1000 samples, 10x10 CV)\nSMOTE applied only to training data"
    )
    ax2.set_ylabel("AUC Score")
    ax2.legend(title="Scenario", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.show()

    # Statistical tests
    print("\n" + "=" * 60)
    print("STATISTICAL COMPARISONS")
    print("=" * 60)

    # Compare scenarios for each model
    for model in df["Model"].unique():
        print(f"\n*** {model} ***")
        model_data = df[df["Model"] == model]

        scenarios = model_data["Scenario"].unique()
        scenario_means = []

        print("Performance summary:")
        for scenario in scenarios:
            scores = model_data[model_data["Scenario"] == scenario]["AUC"]
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            scenario_means.append((scenario, mean_score, std_score))
            print(f"  {scenario:15s}: AUC = {mean_score:.3f} ± {std_score:.3f}")

        # Perform pairwise comparisons
        print(f"\n  Pairwise comparisons (t-tests):")
        for i, (sc1, mean1, std1) in enumerate(scenario_means):
            for j, (sc2, mean2, std2) in enumerate(scenario_means):
                if i < j:
                    scores1 = model_data[model_data["Scenario"] == sc1]["AUC"]
                    scores2 = model_data[model_data["Scenario"] == sc2]["AUC"]

                    t_stat, p_value = stats.ttest_ind(scores1, scores2)

                    print(f"    {sc1:12s} vs {sc2:12s}: p = {p_value:.6f}", end="")

                    if p_value < 0.001:
                        significance = " ***"
                    elif p_value < 0.01:
                        significance = " **"
                    elif p_value < 0.05:
                        significance = " *"
                    else:
                        significance = " ns"

                    print(f"{significance}")

                    if p_value < 0.05:
                        if mean1 > mean2:
                            print(
                                f"               {sc1} > {sc2} (Δ = {mean1 - mean2:.3f})"
                            )
                        else:
                            print(
                                f"               {sc2} > {sc1} (Δ = {mean2 - mean1:.3f})"
                            )


# Run the complete analysis
if __name__ == "__main__":
    print("Running corrected multi-site simulation with proper SMOTE workflow...")
    print("Configuration: 1000 samples, 10x10 repeated CV, 4 scenarios")
    print("SMOTE applied ONLY to training data within CV loop")
    print("=" * 70)

    results, imbalanced_data, balanced_data, smote5_data, smote10_data = (
        run_experiments()
    )
    plot_results(results)

# %%
