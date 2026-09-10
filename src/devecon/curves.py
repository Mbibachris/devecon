"""Distributional curves: Lorenz, generalized Lorenz, and growth incidence.

These describe the *shape* of a distribution rather than collapsing it to a
single index. The Lorenz curve underpins the Gini; the generalized Lorenz
curve underpins welfare dominance; the growth incidence curve compares two
distributions to reveal whether growth was pro-poor.
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np

from .poverty import _prepare_sample_weights

LorenzCurve = namedtuple("LorenzCurve", ["population_share", "value_share"])
GIC = namedtuple("GIC", ["percentile", "growth_rate"])


def _sorted_weighted(values, weights):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1:
        raise ValueError("values must be 1-D.")
    if np.any(x < 0):
        raise ValueError("values must be non-negative.")
    w = _prepare_sample_weights(weights, x.shape[0])
    order = np.argsort(x)
    return x[order], w[order]


def lorenz(values, weights=None):
    """Lorenz curve points.

    Returns cumulative population share (x-axis) against cumulative share
    of total income (y-axis), both starting at (0, 0). The 45-degree line
    is perfect equality; the area between it and the curve is the Gini.

    Returns
    -------
    LorenzCurve
        Named tuple (population_share, value_share), each length n+1.
    """
    x, w = _sorted_weighted(values, weights)
    cum_pop = np.concatenate([[0.0], np.cumsum(w)]) / w.sum()
    cum_val = np.concatenate([[0.0], np.cumsum(w * x)])
    total = (w * x).sum()
    cum_val = cum_val / total if total > 0 else np.zeros_like(cum_val)
    return LorenzCurve(population_share=cum_pop, value_share=cum_val)


def generalized_lorenz(values, weights=None):
    """Generalized Lorenz curve: Lorenz ordinates scaled by the mean.

    y-axis is cumulative income per capita rather than a share, so the
    curve encodes both inequality and the level of income. Second-order
    dominance in generalized Lorenz terms implies higher social welfare.

    Returns
    -------
    LorenzCurve
        Named tuple (population_share, value_share); value_share here is
        cumulative mean income (not normalized to 1).
    """
    x, w = _sorted_weighted(values, weights)
    wsum = w.sum()
    cum_pop = np.concatenate([[0.0], np.cumsum(w)]) / wsum
    cum_mean = np.concatenate([[0.0], np.cumsum(w * x)]) / wsum
    return LorenzCurve(population_share=cum_pop, value_share=cum_mean)


def growth_incidence(values_1, values_2, weights_1=None, weights_2=None,
                     n_points=99):
    """Growth incidence curve (GIC) between two distributions.

    At each percentile p, reports the growth rate from period 1 to period 2
    of the income at that percentile:  q2(p) / q1(p) - 1. An upward-sloping
    curve (higher growth at low percentiles) indicates pro-poor growth.

    Parameters
    ----------
    values_1, values_2 : array-like
        Income distributions in period 1 and period 2.
    weights_1, weights_2 : array-like, optional
        Survey weights for each period.
    n_points : int, default 99
        Number of evenly spaced percentiles (default gives 1..99).

    Returns
    -------
    GIC
        Named tuple (percentile, growth_rate), each length n_points.
    """
    percentiles = np.linspace(1, 99, n_points) if n_points == 99 else \
        np.linspace(100 / (n_points + 1), 100 * n_points / (n_points + 1), n_points)
    q1 = _weighted_quantiles(values_1, percentiles / 100.0, weights_1)
    q2 = _weighted_quantiles(values_2, percentiles / 100.0, weights_2)
    growth = q2 / q1 - 1.0
    return GIC(percentile=percentiles, growth_rate=growth)


def _weighted_quantiles(values, probs, weights):
    x = np.asarray(values, dtype=float)
    w = _prepare_sample_weights(weights, x.shape[0])
    order = np.argsort(x)
    x, w = x[order], w[order]
    cum = np.cumsum(w) - 0.5 * w
    cum /= w.sum()
    return np.interp(probs, cum, x)