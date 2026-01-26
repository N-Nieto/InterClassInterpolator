# %%
import numpy as np
import pytest
from ICI import ICIHarmonization


def generate_data(multiclass=False):
    X = np.random.randn(200, 4)
    sites = np.array([0] * 100 + [1] * 100)

    if multiclass:
        y = np.array([0] * 120 + [1] * 50 + [2] * 30)
    else:
        y = np.array([0] * 150 + [1] * 50)

    return X, y, sites


def test_binary_runs():
    X, y, sites = generate_data()
    y = np.random.permutation(y)
    ici = ICIHarmonization("smote")
    Xr, yr = ici.fit_resample(X, y, sites=sites)
    sr = ici.sites_resampled_
    assert len(Xr) == len(yr) == len(sr)


def test_multiclass_balance():
    X, y, sites = generate_data(multiclass=True)
    y = np.random.permutation(y)
    ici = ICIHarmonization("random")
    _, yr = ici.fit_resample(X, y, sites=sites)
    sr = ici.sites_resampled_

    for site in np.unique(sr):
        counts = np.unique(yr[sr == site], return_counts=True)[1]
        assert len(set(counts)) == 1


def test_invalid_site():
    X = np.random.randn(10, 2)
    y = np.zeros(10)
    y = np.random.permutation(y)
    sites = np.zeros(10)

    ici = ICIHarmonization()
    with pytest.raises(ValueError):
        ici.fit_resample(X, y, sites=sites)
