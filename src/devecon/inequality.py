"""Inequality measures: Gini, Generalized Entropy, Atkinson.

All measures accept survey weights and operate on a distribution of a
positive welfare variable (income or consumption). Like the poverty
module, they share one weight-preparation helper so the rules live in
one place.
"""

from __future__ import annotations

import numpy as np

from .poverty import _prepare_sample_weights


def _check_positive(x):
    """Inequality indices need strictly positive values (logs, ratios)."""
    if np.any(x <= 0):
        raise ValueError("all values must be strictly positive.")


def gini(values, weights=None):
    """Gini coefficient of a distribution.

    0 = perfect equality, approaching 1 = perfect inequality.
    Computed as the weighted relative mean absolute difference:

        G = sum_i sum_j w_i w_j |x_i - x_j|  /  (2 * (sum w)^2 * mean)

    Parameters
    ----------
    values : array-like
        Income or consumption per unit. Must be non-negative.
    weights : array-like, optional
        Survey weights. Defaults to equal weights.

    Returns
    -------
    float
        Gini coefficient in [0, 1).
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 1:
        raise ValueError("values must be 1-D.")
    if np.any(x < 0):
        raise ValueError("values must be non-negative.")
    w = _prepare_sample_weights(weights, x.shape[0])

    wsum = w.sum()
    mean = (w * x).sum() / wsum
    if mean == 0:
        return 0.0
    # Weighted sum of absolute differences via broadcasting.
    diff = np.abs(x[:, None] - x[None, :])
    num = (w[:, None] * w[None, :] * diff).sum()
    return float(num / (2.0 * wsum * wsum * mean))


def generalized_entropy(values, alpha=1.0, weights=None):
    """Generalized Entropy index GE(alpha).

    alpha controls sensitivity to different parts of the distribution:
      GE(0) = mean log deviation (Theil-L), sensitive to the bottom
      GE(1) = Theil-T index
      GE(2) = half the squared coefficient of variation

    Parameters
    ----------
    values : array-like
        Strictly positive income or consumption per unit.
    alpha : float, default 1.0
        Sensitivity parameter (any real number; 0 and 1 are limits).
    weights : array-like, optional
        Survey weights. Defaults to equal weights.

    Returns
    -------
    float
        GE(alpha) >= 0; 0 means perfect equality.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 1:
        raise ValueError("values must be 1-D.")
    _check_positive(x)
    w = _prepare_sample_weights(weights, x.shape[0])

    wsum = w.sum()
    mean = (w * x).sum() / wsum
    r = x / mean  # normalized incomes

    if np.isclose(alpha, 0.0):
        return float((w * (-np.log(r))).sum() / wsum)
    if np.isclose(alpha, 1.0):
        return float((w * (r * np.log(r))).sum() / wsum)
    coef = 1.0 / (alpha * (alpha - 1.0))
    return float(coef * (w * (r ** alpha - 1.0)).sum() / wsum)


def atkinson(values, epsilon=1.0, weights=None):
    """Atkinson inequality index A(epsilon).

    epsilon >= 0 is the inequality-aversion parameter: higher epsilon
    weights the bottom of the distribution more heavily.

    0 = perfect equality; approaches 1 as inequality rises.

    Parameters
    ----------
    values : array-like
        Strictly positive income or consumption per unit.
    epsilon : float, default 1.0
        Inequality-aversion parameter (>= 0).
    weights : array-like, optional
        Survey weights. Defaults to equal weights.

    Returns
    -------
    float
        Atkinson index in [0, 1).
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 1:
        raise ValueError("values must be 1-D.")
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative.")
    _check_positive(x)
    w = _prepare_sample_weights(weights, x.shape[0])

    wsum = w.sum()
    mean = (w * x).sum() / wsum

    if np.isclose(epsilon, 1.0):
        # EDE is the weighted geometric mean.
        log_ede = (w * np.log(x)).sum() / wsum
        ede = np.exp(log_ede)
    else:
        power = 1.0 - epsilon
        ede = ((w * (x ** power)).sum() / wsum) ** (1.0 / power)
    return float(1.0 - ede / mean)


def palma(values, weights=None):
    """Palma ratio: income share of the top 10% over the bottom 40%.

    A simple, policy-legible inequality measure. Values above 1 mean the
    richest tenth receive more than the poorest four-tenths combined.

    Parameters
    ----------
    values : array-like
        Non-negative income or consumption per unit.
    weights : array-like, optional
        Survey weights; equal by default.

    Returns
    -------
    float
        Palma ratio (>= 0).
    """
    x = np.asarray(values, dtype=float)
    if np.any(x < 0):
        raise ValueError("values must be non-negative.")
    w = _prepare_sample_weights(weights, x.shape[0])
    order = np.argsort(x)
    x, w = x[order], w[order]
    cum_pop = np.cumsum(w) / w.sum()

    
    bottom40 = (w * x)[cum_pop <= 0.40].sum()
    top10 = (w * x)[cum_pop > 0.90].sum()
    if bottom40 == 0:
        raise ValueError("bottom 40% has zero income; Palma is undefined.")
    return float(top10 / bottom40)


def hoover(values, weights=None):
    """Hoover index (Robin Hood index).

    The share of total income that would need to be redistributed to reach
    perfect equality. Equals half the sum of absolute deviations of income
    shares from population shares.

        H = 0.5 * sum_i w_i * | x_i/mean - 1 | / sum_i w_i

    Parameters
    ----------
    values : array-like
        Non-negative income or consumption per unit.
    weights : array-like, optional
        Survey weights; equal by default.

    Returns
    -------
    float
        Hoover index in [0, 1).
    """
    x = np.asarray(values, dtype=float)
    if np.any(x < 0):
        raise ValueError("values must be non-negative.")
    w = _prepare_sample_weights(weights, x.shape[0])
    wsum = w.sum()
    mean = (w * x).sum() / wsum
    if mean == 0:
        return 0.0
    return float(0.5 * (w * np.abs(x / mean - 1.0)).sum() / wsum)