"""
Data simulation module for multi-site data generation
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
import os


class DataSimulator:
    """
    Simulates multi-site data with site effects and class imbalance
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize the data simulator

        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)

    def simulate_multi_site_data_advanced(
        self,
        n_sites: int = 2,
        n_samples: int = 1000,
        balance_per_site: List[float] = None,
        n_features: int = 10,
        signal_strength: float = 1.0,
        noise_strength: float = 1.0,
        site_effect_strength: float = 3.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate multi-site data as:
            X = signal + noise + site effect

        All components are Gaussian.

        Returns
        -------
        X : (n_samples, n_features)
        y : (n_samples,)
        site_labels : (n_samples,)
        """

        if self.random_state is not None:
            np.random.seed(self.random_state)

        if balance_per_site is None:
            balance_per_site = [0.5] * n_sites

        if len(balance_per_site) != n_sites:
            raise ValueError("balance_per_site must have length n_sites")

        # Allocate samples per site
        samples_per_site = np.full(n_sites, n_samples // n_sites)
        samples_per_site[: n_samples % n_sites] += 1

        X_list = []
        y_list = []
        site_list = []

        for site in range(n_sites):
            n_site_samples = samples_per_site[site]
            p_class1 = balance_per_site[site]

            # Labels
            n_class1 = int(np.round(n_site_samples * p_class1))
            n_class0 = n_site_samples - n_class1

            y_site = np.concatenate([
                np.zeros(n_class0, dtype=int),
                np.ones(n_class1, dtype=int)
            ])

            # ------------------
            # Signal component
            # ------------------
            # Class 0 mean = -signal_strength / 2
            # Class 1 mean = +signal_strength / 2
            signal = np.zeros((n_site_samples, n_features))

            if n_class0 > 0:
                signal[y_site == 0] = np.random.normal(
                    loc=-signal_strength / 2,
                    scale=1.0,
                    size=(n_class0, n_features),
                )

            if n_class1 > 0:
                signal[y_site == 1] = np.random.normal(
                    loc=signal_strength / 2,
                    scale=1.0,
                    size=(n_class1, n_features),
                )

            # ------------------
            # Noise component
            # ------------------
            noise = np.random.normal(
                loc=0.0,
                scale=noise_strength,
                size=(n_site_samples, n_features),
            )

            # ------------------
            # Site effect
            # ------------------
            # One site-specific shift per feature
            site_effect = np.random.normal(
                loc=0.0,
                scale=site_effect_strength,
                size=(1, n_features),
            )

            # ------------------
            # Combine
            # ------------------
            X_site = signal + noise + site_effect

            X_list.append(X_site)
            y_list.append(y_site)
            site_list.extend([site] * n_site_samples)

        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        site_labels = np.array(site_list)

        return X, y, site_labels


    def simulate_multi_site_data_2d(
        self,
        n_sites: int = 2,
        n_samples: int = 1000,
        balance_per_site: List[float] = [0.5, 0.5],
        site_effect_strength: float = 3.0,
        class_effect_strength: float = 0.3,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate 2D multi-site data for visualization

        Parameters:
        -----------
        n_sites : int
            Number of sites
        n_samples : int
            Total number of samples
        balance_per_site : List[float]
            List of class balances for each site (proportion of class 1)
        site_effect_strength : float
            Strength of site-specific effects
        class_effect_strength : float
            Strength of class-specific effects

        Returns:
        --------
        X : np.ndarray
            Feature matrix of shape (n_samples, 2)
        y : np.ndarray
            Target labels of shape (n_samples,)
        site_labels : np.ndarray
            Site labels of shape (n_samples,)
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

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
            site_effect = np.array(
                [
                    np.random.normal(site * site_effect_strength, 1.5),
                    np.random.normal(site * site_effect_strength, 1.5),
                ]
            )

            # Generate class labels based on balance
            n_class1 = int(site_samples * balance_per_site[site])
            n_class0 = site_samples - n_class1

            # Create labels
            site_y = np.concatenate([np.zeros(n_class0), np.ones(n_class1)])

            # Generate features: noise + site effect
            site_X = np.random.normal(0, 1, (site_samples, 2))

            # Add site effect to all samples
            site_X += site_effect

            # Add class effect
            class_effect = np.array([class_effect_strength, -class_effect_strength])

            # Apply class effect only to class 1 samples
            class_1_mask = site_y == 1
            site_X[class_1_mask] += class_effect

            X.append(site_X)
            y.append(site_y)
            site_labels.extend([site] * site_samples)

        X = np.vstack(X)
        y = np.concatenate(y)
        site_labels = np.array(site_labels)

        return X, y, site_labels

    def simulate_multi_site_data_2d_advanced(
        self,
        n_sites: int = 2,
        n_samples: int = 1000,
        balance_per_site: List[float] = [0.5, 0.5],
        site_effect_strength: float = 3.0,
        class_means: List[List[float]] = None,
        class_stds: List[List[float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate 2D multi-site data with realistic class distributions

        Parameters:
        -----------
        n_sites : int
            Number of sites
        n_samples : int
            Total number of samples
        balance_per_site : List[float]
            List of class balances for each site (proportion of class 1)
        site_effect_strength : float
            Strength of site-specific effects
        class_means : List of lists, optional
            Mean values for each class. Format: [[mean_x0, mean_y0], [mean_x1, mean_y1]]
            If None, uses [[0, 0], [1, 1]]
        class_stds : List of lists, optional
            Standard deviations for each class. Format: [[std_x0, std_y0], [std_x1, std_y1]]
            If None, uses [[1, 1], [1, 1]]

        Returns:
        --------
        X : np.ndarray
            Feature matrix of shape (n_samples, 2)
        y : np.ndarray
            Target labels of shape (n_samples,)
        site_labels : np.ndarray
            Site labels of shape (n_samples,)
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Set default class distributions
        if class_means is None:
            class_means = [[0, 0], [1, 1]]  # Class 0 mean = [0,0], Class 1 mean = [1,1]
        if class_stds is None:
            class_stds = [[1, 1], [1, 1]]  # Same standard deviation for both classes

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
            site_effect = np.array(
                [
                    np.random.normal(site * site_effect_strength, 1.0),
                    np.random.normal(site * site_effect_strength, 1.0),
                ]
            )

            # Generate class labels based on balance
            n_class1 = int(site_samples * balance_per_site[site])
            n_class0 = site_samples - n_class1

            # Create labels
            site_y = np.concatenate([np.zeros(n_class0), np.ones(n_class1)])

            # Generate features with class-specific distributions
            site_X = np.zeros((site_samples, 2))

            # Generate class 0 samples
            if n_class0 > 0:
                site_X[site_y == 0] = (
                    np.random.normal(class_means[0], class_stds[0], (n_class0, 2))
                    + site_effect
                )

            # Generate class 1 samples
            if n_class1 > 0:
                site_X[site_y == 1] = (
                    np.random.normal(class_means[1], class_stds[1], (n_class1, 2))
                    + site_effect
                )

            X.append(site_X)
            y.append(site_y)
            site_labels.extend([site] * site_samples)

        X = np.vstack(X)
        y = np.concatenate(y)
        site_labels = np.array(site_labels)

        return X, y, site_labels

    def save_simulated_data(
        self, X: np.ndarray, y: np.ndarray, site_labels: np.ndarray, filepath: str
    ) -> None:
        """
        Save simulated data to CSV file

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target labels
        site_labels : np.ndarray
            Site labels
        filepath : str
            Path to save the data
        """
        # Create DataFrame
        data_dict = {f"feature_{i}": X[:, i] for i in range(X.shape[1])}
        data_dict["target"] = y
        data_dict["site"] = site_labels

        df = pd.DataFrame(data_dict)
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")

    def load_simulated_data(
        self, filepath: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load simulated data from CSV file

        Parameters:
        -----------
        filepath : str
            Path to load the data from

        Returns:
        --------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target labels
        site_labels : np.ndarray
            Site labels
        """
        df = pd.read_csv(filepath)

        # Extract features
        feature_cols = [col for col in df.columns if col.startswith("feature_")]
        X = df[feature_cols].values

        # Extract target and site labels
        y = df["target"].values
        site_labels = df["site"].values

        return X, y, site_labels

    def get_data_statistics(
        self, X: np.ndarray, y: np.ndarray, site_labels: np.ndarray
    ) -> Dict[str, Any]:
        """
        Get statistics about the simulated data

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target labels
        site_labels : np.ndarray
            Site labels

        Returns:
        --------
        stats : Dict
            Dictionary containing data statistics
        """
        stats = {
            "n_samples": len(y),
            "n_features": X.shape[1],
            "n_sites": len(np.unique(site_labels)),
            "overall_class_balance": np.mean(y),
            "site_target_correlation": np.corrcoef(site_labels, y)[0, 1],
            "site_statistics": {},
            "class_statistics": {},
        }

        # Site statistics
        for site in np.unique(site_labels):
            site_mask = site_labels == site
            stats["site_statistics"][f"site_{site}"] = {
                "n_samples": np.sum(site_mask),
                "class_balance": np.mean(y[site_mask]),
                "n_class_0": np.sum(y[site_mask] == 0),
                "n_class_1": np.sum(y[site_mask] == 1),
                "feature_means": np.mean(X[site_mask], axis=0).tolist(),
                "feature_stds": np.std(X[site_mask], axis=0).tolist(),
            }

        # Class statistics
        for class_label in [0, 1]:
            class_mask = y == class_label
            stats["class_statistics"][f"class_{class_label}"] = {
                "n_samples": np.sum(class_mask),
                "feature_means": np.mean(X[class_mask], axis=0).tolist(),
                "feature_stds": np.std(X[class_mask], axis=0).tolist(),
            }

        return stats
