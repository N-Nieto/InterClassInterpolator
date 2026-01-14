import numpy as np
from typing import Tuple


class ICI_Corrector:
    """
    InterClassInterpolator Corrector for multi-site data. Interpolate the minority class for the sites 
    separately to balance classes, thus removing the correlation between Effects of Site and class.
    """

    def __init__(self, interpolator, random_state: int = 42):
        """
        Initialize SMOTE corrector

        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.interpolator = interpolator

    def balance(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        site_train: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply SMOTE separately to each site in the training data to balance classes
        with optional site effect removal

        Parameters:
        -----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training labels
        site_train : np.ndarray
            Site labels for training data

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

        # check there is at least two sites
        if len(unique_sites) < 2:
            raise ValueError("At least two unique sites are required for site-wise SMOTE.")

        # Calculate site effects if removal is requested
        for site in unique_sites:
            # Get data for current site
            site_mask = site_train == site
            X_site = X_train[
                site_mask
            ].copy()  # Make a copy to avoid modifying original
            y_site = y_train[site_mask]

            # Check class distribution
            n_class0 = np.sum(y_site == 0)
            n_class1 = np.sum(y_site == 1)

            # Check at least two classes are present
            if n_class0 == 0 or n_class1 == 0:
                raise ValueError(
                    f"Site {site} does not have samples from both classes. "
                    "Method requires at least one sample from each class."
                    f"Given class counts: {n_class0}, {n_class1}"
                )
            # Only apply SMOTE if there's imbalance
            if n_class0 != n_class1:
                X_site_resampled, y_site_resampled = self.interpolator.fit_resample(X_site, y_site)
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
