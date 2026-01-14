"""
Utility functions for model evaluation and reporting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
from typing import List, Dict, Any, Tuple


class ModelEvaluator:
    """
    Handles model evaluation with cross-validation
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize model evaluator

        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state

    def evaluate_model_with_smote(
        self,
        X: np.ndarray,
        y: np.ndarray,
        site_labels: np.ndarray,
        model: Any,
        scaler: StandardScaler = None,
        use_smote: bool = False,
        k_neighbors: int = 5,
        smote_corrector: Any = None,
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
        smote_corrector : SMOTECorrector
            SMOTE correction instance

        Returns:
        --------
        scores : List[float]
            List of AUC scores for each fold
        """
        cv = RepeatedStratifiedKFold(
            n_splits=10, n_repeats=10, random_state=self.random_state
        )
        scores = []

        for train_idx, test_idx in cv.split(X, y):
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            site_train, site_test = site_labels[train_idx], site_labels[test_idx]

            # Apply SMOTE only to training data if requested
            if use_smote and smote_corrector is not None:
                X_train_processed, y_train_processed, site_train_processed = (
                    smote_corrector.apply_smote_per_site_to_training(
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


class ResultAnalyzer:
    """
    Analyzes and visualizes experiment results
    """

    def __init__(self):
        """Initialize result analyzer"""
        pass

    def create_performance_plot(
        self, results: Dict[str, Dict[str, List[float]]], save_path: str = None
    ) -> None:
        """
        Create boxplots of model performance across scenarios

        Parameters:
        -----------
        results : Dict
            Dictionary containing AUC scores for all scenarios and models
        save_path : str, optional
            Path to save the plot
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

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {save_path}")

        plt.show()

    def perform_statistical_analysis(
        self, results: Dict[str, Dict[str, List[float]]]
    ) -> str:
        """
        Perform statistical analysis of results

        Parameters:
        -----------
        results : Dict
            Dictionary containing AUC scores for all scenarios and models

        Returns:
        --------
        report : str
            Statistical analysis report
        """
        # Prepare data for analysis
        plot_data = []
        for scenario, models in results.items():
            for model_name, scores in models.items():
                for score in scores:
                    plot_data.append(
                        {"Scenario": scenario, "Model": model_name, "AUC": score}
                    )

        df = pd.DataFrame(plot_data)

        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("STATISTICAL COMPARISONS")
        report_lines.append("=" * 60)

        # Compare scenarios for each model
        for model in df["Model"].unique():
            report_lines.append(f"\n*** {model} ***")
            model_data = df[df["Model"] == model]

            scenarios = model_data["Scenario"].unique()
            scenario_means = []

            report_lines.append("Performance summary:")
            for scenario in scenarios:
                scores = model_data[model_data["Scenario"] == scenario]["AUC"]
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                scenario_means.append((scenario, mean_score, std_score))
                report_lines.append(
                    f"  {scenario:15s}: AUC = {mean_score:.3f} ± {std_score:.3f}"
                )

            # Perform pairwise comparisons
            report_lines.append(f"\n  Pairwise comparisons (t-tests):")
            for i, (sc1, mean1, std1) in enumerate(scenario_means):
                for j, (sc2, mean2, std2) in enumerate(scenario_means):
                    if i < j:
                        scores1 = model_data[model_data["Scenario"] == sc1]["AUC"]
                        scores2 = model_data[model_data["Scenario"] == sc2]["AUC"]

                        t_stat, p_value = stats.ttest_ind(scores1, scores2)

                        # NEW CODE:
                        line = f"    {sc1:12s} vs {sc2:12s}: p = {p_value:.6f}"

                        if p_value < 0.001:
                            line += " ***"
                        elif p_value < 0.01:
                            line += " **"
                        elif p_value < 0.05:
                            line += " *"
                        else:
                            line += " ns"

                        report_lines.append(line)

                        if p_value < 0.05:
                            if mean1 > mean2:
                                report_lines.append(
                                    f"               {sc1} > {sc2} (Δ = {mean1 - mean2:.3f})"
                                )
                            else:
                                report_lines.append(
                                    f"               {sc2} > {sc1} (Δ = {mean2 - mean1:.3f})"
                                )

        return "\n".join(report_lines)

    def save_report(self, report: str, save_path: str) -> None:
        """
        Save analysis report to file

        Parameters:
        -----------
        report : str
            Report text
        save_path : str
            Path to save the report
        """
        with open(save_path, "w") as f:
            f.write(report)
        print(f"Report saved to {save_path}")

    def plot_feature_means_by_site_class(
        self,
        X: np.ndarray,
        y: np.ndarray,
        site_labels: np.ndarray,
        save_path: str = None,
    ) -> None:
        """
        Plot mean feature values by site and class to visualize site effects

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target labels
        site_labels : np.ndarray
            Site labels
        save_path : str, optional
            Path to save the plot
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        unique_sites = np.unique(site_labels)
        n_features = min(10, X.shape[1])  # Plot first 10 features

        # Calculate means
        means_data = []
        for site in unique_sites:
            for class_label in [0, 1]:
                mask = (site_labels == site) & (y == class_label)
                if np.sum(mask) > 0:
                    feature_means = np.mean(X[mask], axis=0)
                    for feature_idx in range(n_features):
                        means_data.append(
                            {
                                "Site": f"Site {site}",
                                "Class": f"Class {class_label}",
                                "Feature": f"Feature {feature_idx}",
                                "Mean Value": feature_means[feature_idx],
                            }
                        )

        means_df = pd.DataFrame(means_data)

        plt.figure(figsize=(12, 8))
        sns.heatmap(
            means_df.pivot_table(
                index=["Site", "Class"], columns="Feature", values="Mean Value"
            ),
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
        )
        plt.title("Mean Feature Values by Site and Class\n(Visualizing Site Effects)")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Feature means plot saved to {save_path}")

        plt.show()

    def plot_data_distribution(
        self,
        X: np.ndarray,
        y: np.ndarray,
        site_labels: np.ndarray,
        save_path: str = None,
        n_features_to_plot: int = 4,
    ) -> None:
        """
        Plot histogram of feature values for each site and class

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target labels
        site_labels : np.ndarray
            Site labels
        save_path : str, optional
            Path to save the plot
        n_features_to_plot : int
            Number of features to display (first n features)
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        n_features = min(n_features_to_plot, X.shape[1])
        unique_sites = np.unique(site_labels)

        fig, axes = plt.subplots(
            n_features,
            len(unique_sites),
            figsize=(5 * len(unique_sites), 4 * n_features),
        )

        if n_features == 1:
            axes = axes.reshape(1, -1)

        for feature_idx in range(n_features):
            for site_idx, site in enumerate(unique_sites):
                ax = axes[feature_idx, site_idx]

                site_mask = site_labels == site

                # Plot class 0
                class0_mask = (y == 0) & site_mask
                if np.sum(class0_mask) > 0:
                    ax.hist(
                        X[class0_mask, feature_idx],
                        alpha=0.7,
                        label="Class 0",
                        bins=20,
                        color="blue",
                        density=True,
                    )

                # Plot class 1
                class1_mask = (y == 1) & site_mask
                if np.sum(class1_mask) > 0:
                    ax.hist(
                        X[class1_mask, feature_idx],
                        alpha=0.7,
                        label="Class 1",
                        bins=20,
                        color="red",
                        density=True,
                    )

                ax.set_title(f"Site {site}, Feature {feature_idx}")
                ax.set_xlabel(f"Feature {feature_idx} Value")
                ax.set_ylabel("Density")
                ax.legend()

        plt.suptitle("Feature Distributions by Site and Class", fontsize=16, y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Distribution plot saved to {save_path}")

        plt.show()

    def plot_decision_boundaries_2d(self,
                                X: np.ndarray,
                                y: np.ndarray,
                                site_labels: np.ndarray,
                                models: Dict[str, Any],
                                scenario_name: str,
                                save_path: str = None) -> None:
        """
        Plot 2D data with decision boundaries for multiple models
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix (should have 2 features)
        y : np.ndarray
            Target labels
        site_labels : np.ndarray
            Site labels
        models : Dict
            Dictionary of model names and instances
        scenario_name : str
            Name of the scenario for the title
        save_path : str, optional
            Path to save the plot
        """
        if X.shape[1] != 2:
            raise ValueError("X must have exactly 2 features for 2D plotting")
        
        unique_sites = np.unique(site_labels)
        n_models = len(models)
        
        fig, axes = plt.subplots(2, n_models, figsize=(6 * n_models, 10))
        
        if n_models == 1:
            axes = axes.reshape(2, 1)
        
        # Create mesh grid for decision boundaries
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
        
        for model_idx, (model_name, model) in enumerate(models.items()):
            # Train model on all data for visualization
            model_instance = model.__class__(**model.get_params())
            model_instance.fit(X, y)
            
            # Plot decision boundary
            Z = model_instance.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
            Z = Z.reshape(xx.shape)
            
            # Top row: Contour plot with decision boundary
            ax1 = axes[0, model_idx]
            contour = ax1.contourf(xx, yy, Z, levels=20, alpha=0.6, cmap='RdYlBu')
            ax1.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
            
            # Plot data points
            for site in unique_sites:
                for class_label in [0, 1]:
                    mask = (site_labels == site) & (y == class_label)
                    if np.sum(mask) > 0:
                        color = 'blue' if class_label == 0 else 'red'
                        marker = 'o' if site == 0 else 's'
                        alpha = 0.7 if site == 0 else 0.5
                        size = 50 if site == 0 else 30
                        ax1.scatter(X[mask, 0], X[mask, 1], 
                                c=color, marker=marker, alpha=alpha, s=size,
                                label=f'Site {site}, Class {class_label}')
            
            ax1.set_xlim(x_min, x_max)
            ax1.set_ylim(y_min, y_max)
            ax1.set_title(f'{model_name}\nDecision Boundary')
            ax1.set_xlabel('Feature 1')
            ax1.set_ylabel('Feature 2')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Bottom row: Site-wise distribution
            ax2 = axes[1, model_idx]
            for site in unique_sites:
                for class_label in [0, 1]:
                    mask = (site_labels == site) & (y == class_label)
                    if np.sum(mask) > 0:
                        color = 'blue' if class_label == 0 else 'red'
                        marker = 'o' if site == 0 else 's'
                        alpha = 0.7 if site == 0 else 0.5
                        size = 50 if site == 0 else 30
                        ax2.scatter(X[mask, 0], X[mask, 1], 
                                c=color, marker=marker, alpha=alpha, s=size,
                                label=f'Site {site}, Class {class_label}')
            
            ax2.set_xlim(x_min, x_max)
            ax2.set_ylim(y_min, y_max)
            ax2.set_title(f'{model_name}\nData Distribution')
            ax2.set_xlabel('Feature 1')
            ax2.set_ylabel('Feature 2')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.suptitle(f'2D Visualization: {scenario_name}\nTop: Decision Boundaries, Bottom: Data Distribution', 
                    fontsize=16, y=0.95)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"2D decision boundary plot saved to {save_path}")
        
        plt.show()

    def plot_comparison_2d(self,
                        scenarios: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
                        models: Dict[str, Any],
                        save_path: str = None) -> None:
        """
        Compare decision boundaries across different scenarios
        
        Parameters:
        -----------
        scenarios : Dict
            Dictionary of scenario names and (X, y, site_labels) tuples
        models : Dict
            Dictionary of model names and instances
        save_path : str, optional
            Path to save the plot
        """
        n_scenarios = len(scenarios)
        n_models = len(models)
        
        fig, axes = plt.subplots(n_scenarios, n_models, 
                                figsize=(6 * n_models, 5 * n_scenarios))
        
        if n_scenarios == 1:
            axes = axes.reshape(1, -1)
        if n_models == 1:
            axes = axes.reshape(-1, 1)
        
        for scenario_idx, (scenario_name, (X, y, site_labels)) in enumerate(scenarios.items()):
            if X.shape[1] != 2:
                print(f"Skipping {scenario_name}: Requires 2D data")
                continue
                
            unique_sites = np.unique(site_labels)
            
            # Create mesh grid for decision boundaries
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                                np.linspace(y_min, y_max, 100))
            
            for model_idx, (model_name, model) in enumerate(models.items()):
                ax = axes[scenario_idx, model_idx]
                
                # Train model on all data for visualization
                model_instance = model.__class__(**model.get_params())
                model_instance.fit(X, y)
                
                # Plot decision boundary
                Z = model_instance.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
                Z = Z.reshape(xx.shape)
                
                contour = ax.contourf(xx, yy, Z, levels=20, alpha=0.6, cmap='RdYlBu')
                ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
                
                # Plot data points
                for site in unique_sites:
                    for class_label in [0, 1]:
                        mask = (site_labels == site) & (y == class_label)
                        if np.sum(mask) > 0:
                            color = 'blue' if class_label == 0 else 'red'
                            marker = 'o' if site == 0 else 's'
                            alpha = 0.7 if site == 0 else 0.5
                            size = 30
                            ax.scatter(X[mask, 0], X[mask, 1], 
                                    c=color, marker=marker, alpha=alpha, s=size,
                                    label=f'Site {site}, Class {class_label}')
                
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                ax.set_title(f'{model_name} - {scenario_name}')
                ax.set_xlabel('Feature 1')
                ax.set_ylabel('Feature 2')
                
                # Only add legend to first column
                if model_idx == 0:
                    ax.legend(bbox_to_anchor=(0, 1), loc='lower left')
        
        plt.suptitle('Decision Boundary Comparison Across Scenarios', fontsize=16, y=0.95)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to {save_path}")
        
        plt.show()