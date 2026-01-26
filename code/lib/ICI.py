from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Union
from collections import Counter

from sklearn.base import BaseEstimator
from imblearn.base import SamplerMixin
from sklearn.utils.validation import check_X_y, check_array


class ICIHarmonization(BaseEstimator, SamplerMixin):
    """
    Inter-Class Interpolation (ICI) Harmonization.

    Performs site-wise oversampling to remove site–class correlation.
    Works for binary and multi-class classification.
    """

    def __init__(
        self,
        interpolator: Union[str, SamplerMixin] = "smote",
        *,
        random_state: int = 42,
        verbose: bool = False,
        **kwargs,
    ):
        self.interpolator = interpolator
        self.random_state = random_state
        self.verbose = verbose
        self.kwargs = kwargs

        if isinstance(interpolator, str):
            self._base_sampler = self._create_interpolator(interpolator)
        else:
            self._base_sampler = interpolator

    # ------------------------------------------------------------------ #
    # Public API (correct extension point)
    # ------------------------------------------------------------------ #

    def fit_resample(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sites: np.ndarray,
    ):
        """
        Fit and resample the dataset using site-wise harmonization.
        """
        X, y = check_X_y(X, y)
        sites = check_array(sites, ensure_2d=False)

        self._sanity_checks(X, y, sites)

        X_out, y_out, sites_out = [], [], []

        for site in np.unique(sites):
            mask = sites == site
            X_site, y_site = X[mask], y[mask]

            if self.verbose:
                print(f"[ICI] Site {site}: {Counter(y_site)}")

            strategy = self._build_sampling_strategy(y_site)

            if strategy:
                sampler = self._clone_sampler(strategy)
                X_rs, y_rs = sampler.fit_resample(X_site, y_site)
            else:
                X_rs, y_rs = X_site, y_site

            X_out.append(X_rs)
            y_out.append(y_rs)
            sites_out.append(np.full(len(X_rs), site))

        self.sites_resampled_ = np.concatenate(sites_out)

        return np.vstack(X_out), np.concatenate(y_out)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _build_sampling_strategy(self, y: np.ndarray) -> Optional[Dict[int, int]]:
        counts = Counter(y)
        if len(counts) < 2:
            return None

        max_count = max(counts.values())
        return {cls: max_count for cls, c in counts.items() if c < max_count} or None

    def _clone_sampler(self, sampling_strategy):
        params = self._base_sampler.get_params()
        params["sampling_strategy"] = sampling_strategy
        return self._base_sampler.__class__(**params)

    def _sanity_checks(self, X, y, sites):
        if X.shape[0] != sites.shape[0]:
            raise ValueError("X and sites must have same length")

        if len(np.unique(sites)) < 2:
            raise ValueError("At least two sites required")

        for site in np.unique(sites):
            if len(np.unique(y[sites == site])) < 2:
                raise ValueError(f"Site {site} has only one class; cannot resample.")

    def _create_interpolator(self, name: str):
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

        name = name.lower()
        if name not in mapping:
            raise ValueError(f"Unsupported interpolator: {name}")

        return mapping[name](random_state=self.random_state, **self.kwargs)

    def _fit_resample(self, X, y, **params):
        raise NotImplementedError(
            "_fit_resample is not used. Use fit_resample(X, y, sites=...) instead."
        )
