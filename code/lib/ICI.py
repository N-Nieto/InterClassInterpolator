from __future__ import annotations

import numpy as np
from typing import Dict, Tuple, Union, Optional

from collections import Counter
from imblearn.over_sampling.base import BaseOverSampler
from sklearn.utils.validation import check_X_y, check_array


class ICI_harmonization(BaseOverSampler):
    """
    Inter-Class Interpolator (ICI) harmonization.

    This oversampler removes spurious correlations between *site* and *class*
    by performing **site-wise class balancing**. While the data is not directly harmonized,
    as typically done in neuroimaging, the class distributions are equalized
    across sites, which mitigates site-related biases in downstream classification tasks.

    For each site:
        - Detect the majority class
        - Upsample *all minority classes* to match the majority count
        - Works for binary and multi-class classification

    This class wraps any oversampling strategy available in `imblearn`.
    """

    def __init__(
        self,
        interpolator: Union[str, BaseOverSampler] = "smote",
        *,
        random_state: int = 42,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        interpolator : str or BaseOverSampler
            Either:
              - string identifier ("smote", "adasyn", ...)
              - instantiated imblearn oversampler
        random_state : int
            Random seed
        verbose : bool
            Verbose output
        kwargs :
            Passed to the oversampler constructor
        """
        super().__init__()
        self.random_state = random_state
        self.verbose = verbose

        if isinstance(interpolator, str):
            self.interpolator = self._create_interpolator(interpolator, **kwargs)
        elif isinstance(interpolator, BaseOverSampler):
            self.interpolator = interpolator
        else:
            raise TypeError("interpolator must be a string or BaseOverSampler")

        self.set_fit_request(sites=True)

    # ------------------------------------------------------------------ #
    # Core API
    # ------------------------------------------------------------------ #

    def fit_resample(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sites: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform site-wise oversampling.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Class labels
        sites : np.ndarray
            Site labels

        Returns
        -------
        X_resampled, y_resampled, sites_resampled
        """
        X, y = check_X_y(X, y)
        sites = check_array(sites, ensure_2d=False)

        self._sanity_checks(X, y, sites)

        X_out, y_out, sites_out = [], [], []

        for site in np.unique(sites):
            mask = sites == site
            X_site, y_site = X[mask], y[mask]

            if self.verbose:
                print(f"[ICI] Processing site {site}: {Counter(y_site)}")

            # Determine target sampling strategy
            sampling_strategy = self._build_sampling_strategy(y_site)

            if sampling_strategy:
                sampler = self._clone_interpolator(sampling_strategy)
                X_rs, y_rs = sampler.fit_resample(X_site, y_site)
            else:
                X_rs, y_rs = X_site, y_site

            X_out.append(X_rs)
            y_out.append(y_rs)
            sites_out.append(np.full(len(X_rs), site))

        return (
            np.vstack(X_out),
            np.concatenate(y_out),
            np.concatenate(sites_out),
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _build_sampling_strategy(self, y: np.ndarray) -> Optional[Dict[int, int]]:
        """
        Create a per-class sampling strategy that upsamples
        all minority classes to the majority class count.
        """
        class_counts = Counter(y)

        if len(class_counts) < 2:
            return None

        max_count = max(class_counts.values())

        strategy = {
            cls: max_count
            for cls, count in class_counts.items()
            if count < max_count
        }

        return strategy if strategy else None

    def _clone_interpolator(
        self, sampling_strategy: Dict[int, int]
    ) -> BaseOverSampler:
        """
        Clone interpolator with updated sampling strategy.
        """
        params = self.interpolator.get_params()
        params["sampling_strategy"] = sampling_strategy
        return self.interpolator.__class__(**params)

    # ------------------------------------------------------------------ #
    # Validation & sanity checks
    # ------------------------------------------------------------------ #

    def _sanity_checks(
        self, X: np.ndarray, y: np.ndarray, sites: np.ndarray
    ) -> None:
        if X.shape[0] != sites.shape[0]:
            raise ValueError("X and sites must have same number of samples")

        if len(np.unique(sites)) < 2:
            raise ValueError("At least two unique sites are required")

        for site in np.unique(sites):
            y_site = y[sites == site]
            if len(np.unique(y_site)) < 2:
                raise ValueError(
                    f"Site {site} contains only one class. "
                    "Oversampling requires ≥2 classes per site."
                )

    # ------------------------------------------------------------------ #
    # Interpolator factory
    # ------------------------------------------------------------------ #

    def _create_interpolator(
        self, name: str, **kwargs
    ) -> BaseOverSampler:
        """
        Factory for all imblearn oversamplers.
        """
        name = name.lower()

        from imblearn.over_sampling import (
            SMOTE,
            BorderlineSMOTE,
            SVMSMOTE,
            ADASYN,
            KMeansSMOTE,
            RandomOverSampler,
        )

        mapping = {
            "smote": SMOTE,
            "borderline-smote": BorderlineSMOTE,
            "svm-smote": SVMSMOTE,
            "adasyn": ADASYN,
            "kmeans-smote": KMeansSMOTE,
            "random": RandomOverSampler,
        }

        if name not in mapping:
            raise ValueError(f"Unsupported interpolator: {name}")

        return mapping[name](random_state=self.random_state, **kwargs)
