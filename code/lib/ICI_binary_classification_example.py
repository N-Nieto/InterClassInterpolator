import numpy as np
from uniharmony.interpolation import ICICorrector

X = np.random.randn(200, 5)
y = np.array([0] * 140 + [1] * 60)
sites = np.array([0] * 100 + [1] * 100)

ici = ICICorrector("smote", verbose=True)
X_r, y_r, s_r = ici.fit_resample(X, y, sites=sites)

print(np.unique(y_r, return_counts=True))
