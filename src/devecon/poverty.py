from __future__ import annotations

from collections import namedtuple

import numpy as np


def _prepare_sample_weights(weights, n):
    if weights is None:
        return np.ones(n, dtype=float)
    w = np.asarray(weights, dtype=float)
    if w.shape != (n,):
        raise ValueError(f"sample weights must have shape ({n},).")
    if np.any(w < 0):
        raise ValueError("sample weights must be non-negative.")
    return w

def fgt(income, poverty_line, alpha=0.0, weights=None):
    income = np.asarray(income, dtype=float)
    if poverty_line <= 0:
        raise ValueError("poverty_line must be positive.")
    if alpha < 0:
        raise ValueError("alpha must be non-negative.")
    weights = _prepare_sample_weights(weights, income.shape[0])
    if income.ndim != 1:
        raise ValueError("income must be 1-D.")
    poor = income < poverty_line
    gaps = np.clip((poverty_line - income) / poverty_line, 0.0, None)
    contributions = weights * np.where(poor, gaps ** alpha, 0.0)
    return float(contributions.sum() / weights.sum())

AFResult = namedtuple("AFResult", ["H", "A", "M0"])

def alkire_foster(deprivations, cutoff_k, indicator_weights=None, sample_weights=None):
    dep = np.asarray(deprivations, dtype=float)
    if dep.ndim != 2:
        raise ValueError("deprivations must be 2-D (people x indicators).")
    n, d = dep.shape
    if not np.all((dep == 0) | (dep == 1)):
        raise ValueError("deprivations must contain only 0 and 1.")
    if not (0 < cutoff_k <= 1):
        raise ValueError("cutoff_k must be in the interval (0, 1].")
    if indicator_weights is None:
        w = np.ones(d, dtype=float)
    else:
        w = np.asarray(indicator_weights, dtype=float)
        if w.shape != (d,):
            raise ValueError(f"indicator_weights must have shape ({d},).")
        if np.any(w < 0):
            raise ValueError("indicator_weights must be non-negative.")
    w = w / w.sum()
    s = _prepare_sample_weights(sample_weights, n)
    scores = dep @ w
    poor = scores >= cutoff_k
    censored = np.where(poor, scores, 0.0)
    total = s.sum()
    H = s[poor].sum() / total
    M0 = (s * censored).sum() / total
    A = M0 / H if H > 0 else 0.0
    return AFResult(H=float(H), A=float(A), M0=float(M0))


def watts(income, poverty_line, weights=None):
    """Watts poverty index.

    W = (1/N) * sum over poor of [ ln(z / y_i) ], weighted.

    A distribution-sensitive, subgroup-decomposable poverty measure that
    satisfies the strong transfer axiom. Larger shortfalls below the line
    contribute more than proportionally.

    Parameters
    ----------
    income : array-like
        Income or consumption per unit. Poor units must be positive.
    poverty_line : float
        The poverty line z (> 0).
    weights : array-like, optional
        Survey weights; equal by default.

    Returns
    -------
    float
        Watts index (>= 0); 0 when no one is poor.
    """
    y = np.asarray(income, dtype=float)
    if poverty_line <= 0:
        raise ValueError("poverty_line must be positive.")
    w = _prepare_sample_weights(weights, y.shape[0])
    poor = y < poverty_line
    if np.any(poor & (y <= 0)):
        raise ValueError("poor incomes must be positive to take logarithms.")
    contrib = np.where(poor, np.log(poverty_line / np.where(poor, y, poverty_line)), 0.0)
    return float((w * contrib).sum() / w.sum())


def sen(income, poverty_line, weights=None):
    """Sen poverty index.

    S = H * (I + (1 - I) * G_p)

    where H is the headcount ratio, I the average normalized income gap
    among the poor, and G_p the Gini coefficient of incomes among the
    poor. Combines incidence, depth, and inequality among the poor into
    one measure.

    Parameters
    ----------
    income : array-like
        Income or consumption per unit.
    poverty_line : float
        The poverty line z (> 0).
    weights : array-like, optional
        Survey weights; equal by default.

    Returns
    -------
    float
        Sen index in [0, 1].
    """
    y = np.asarray(income, dtype=float)
    if poverty_line <= 0:
        raise ValueError("poverty_line must be positive.")
    w = _prepare_sample_weights(weights, y.shape[0])

    poor = y < poverty_line
    wsum = w.sum()
    H = w[poor].sum() / wsum
    if H == 0:
        return 0.0

    yp = y[poor]
    wp = w[poor]
    # Average normalized gap among the poor.
    gaps = (poverty_line - yp) / poverty_line
    I = (wp * gaps).sum() / wp.sum()
    # Gini among the poor (weighted relative mean absolute difference).
    mean_p = (wp * yp).sum() / wp.sum()
    if mean_p == 0:
        Gp = 0.0
    else:
        diff = np.abs(yp[:, None] - yp[None, :])
        Gp = (wp[:, None] * wp[None, :] * diff).sum() / (
            2.0 * wp.sum() * wp.sum() * mean_p)
    return float(H * (I + (1 - I) * Gp))