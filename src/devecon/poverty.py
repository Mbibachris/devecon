"""Unidimensional poverty measures."""

from __future__ import annotations
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