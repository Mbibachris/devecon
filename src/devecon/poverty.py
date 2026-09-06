"""Poverty measures: unidimensional (FGT) and multidimensional (Alkire-Foster)."""

from __future__ import annotations
from collections import namedtuple
import numpy as np


def fgt(income, poverty_line, alpha=0.0, weights=None):
    """Foster-Greer-Thorbecke (FGT) poverty index.

    P_alpha = sum_i [ w_i * ((z - y_i)/z)^alpha * 1{y_i < z} ] / sum_i w_i

    Parameters
    ----------
    income : array-like
        Income or consumption per person (or household).
    poverty_line : float
        The poverty line z, in the same units as income. Must be > 0.
    alpha : float, default 0.0
        Poverty-aversion parameter (>= 0):
        0 -> headcount ratio, 1 -> poverty gap, 2 -> squared gap (severity).
    weights : array-like, optional
        Survey/sampling weights. Defaults to equal weights.

    Returns
    -------
    float
        The FGT index, in [0, 1].
    """
    income = np.asarray(income, dtype=float)
    weights = (np.ones_like(income) if weights is None
               else np.asarray(weights, dtype=float))

    if poverty_line <= 0:
        raise ValueError("poverty_line must be positive.")
    if alpha < 0:
        raise ValueError("alpha must be non-negative.")
    if income.shape != weights.shape:
        raise ValueError("income and weights must have the same shape.")

    poor = income < poverty_line
    gaps = np.clip((poverty_line - income) / poverty_line, 0.0, None)
    contributions = weights * np.where(poor, gaps ** alpha, 0.0)
    return float(contributions.sum() / weights.sum())


AFResult = namedtuple("AFResult", ["H", "A", "M0"])


def alkire_foster(deprivations, cutoff_k, indicator_weights=None,
                  sample_weights=None):
    """Alkire-Foster multidimensional poverty measures.

    Parameters
    ----------
    deprivations : array-like, shape (n, d)
        Deprivation matrix. Entry (i, j) is 1 if person i is deprived
        in indicator j, else 0. Rows are people (n), columns indicators (d).
    cutoff_k : float
        Poverty cutoff in (0, 1]. A person is multidimensionally poor if
        their weighted deprivation score is >= cutoff_k.
    indicator_weights : array-like, shape (d,), optional
        Weights across indicators (columns). Default equal.
        Rescaled internally to sum to 1.
    sample_weights : array-like, shape (n,), optional
        Survey/sampling weights across people (rows). Default equal.

    Returns
    -------
    AFResult
        Named tuple with fields H (incidence), A (intensity), and
        M0 (adjusted headcount ratio). Satisfies M0 = H * A.
    """
    dep = np.asarray(deprivations, dtype=float)
    if dep.ndim != 2:
        raise ValueError("deprivations must be 2-D (people x indicators).")
    n, d = dep.shape

    if not np.all((dep == 0) | (dep == 1)):
        raise ValueError("deprivations must contain only 0 and 1.")
    if not (0 < cutoff_k <= 1):
        raise ValueError("cutoff_k must be in the interval (0, 1].")

    # Indicator weights: equal by default, rescaled to sum to 1.
    if indicator_weights is None:
        w = np.ones(d, dtype=float)
    else:
        w = np.asarray(indicator_weights, dtype=float)
        if w.shape != (d,):
            raise ValueError(f"indicator_weights must have shape ({d},).")
        if np.any(w < 0):
            raise ValueError("indicator_weights must be non-negative.")
    w = w / w.sum()

    # Sample weights: equal by default.
    if sample_weights is None:
        s = np.ones(n, dtype=float)
    else:
        s = np.asarray(sample_weights, dtype=float)
        if s.shape != (n,):
            raise ValueError(f"sample_weights must have shape ({n},).")
        if np.any(s < 0):
            raise ValueError("sample_weights must be non-negative.")

    # Step 1: weighted deprivation score for each person (in [0, 1]).
    scores = dep @ w

    # Step 2: dual cutoff - identify the multidimensionally poor.
    poor = scores >= cutoff_k

    # Step 3: censor - the non-poor contribute nothing.
    censored = np.where(poor, scores, 0.0)

    # Step 4: aggregate, using sample weights.
    total = s.sum()
    H = s[poor].sum() / total
    M0 = (s * censored).sum() / total
    A = M0 / H if H > 0 else 0.0

    return AFResult(H=float(H), A=float(A), M0=float(M0))
