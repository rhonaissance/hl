import math
from statistics import NormalDist

import numpy as np

_N = NormalDist()


def tstat(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    s = x.std(ddof=1) if len(x) > 1 else 0.0
    return x.mean() / s * math.sqrt(len(x)) if s > 0 else float('nan')


def dsr(x, n_trials):
    """Deflated Sharpe (Bailey & Lopez de Prado): P(true SR > expected max SR of n_trials null trials)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3 or x.std(ddof=1) == 0:
        return float('nan')
    sr = x.mean() / x.std(ddof=1)
    z = (x - x.mean()) / x.std()
    g3, g4 = float((z ** 3).mean()), float((z ** 4).mean())
    em = 0.5772156649015329
    sr0 = 0.0 if n_trials <= 1 else math.sqrt(1 / (n - 1)) * (
        (1 - em) * _N.inv_cdf(1 - 1 / n_trials) + em * _N.inv_cdf(1 - 1 / (n_trials * math.e)))
    den = max(1 - g3 * sr + (g4 - 1) / 4 * sr * sr, 1e-12)
    return _N.cdf((sr - sr0) * math.sqrt(n - 1) / math.sqrt(den))
