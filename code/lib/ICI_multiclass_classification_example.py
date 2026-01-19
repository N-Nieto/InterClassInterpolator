import numpy as np
from uniharmony.interpolation import ICICorrector

X = np.random.randn(300, 10)
y = np.array([0] * 180 + [1] * 80 + [2] * 40)
sites = np.array([0] * 150 + [1] * 150)

ici = ICICorrector("adasyn")
X_r, y_r, s_r = ici.fit_resample(X, y, sites=sites)

for site in np.unique(s_r):
    print(site, np.unique(y_r[s_r == site], return_counts=True))
