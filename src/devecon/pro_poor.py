"""Shared-prosperity and pro-poor growth metrics.

These compare two distributions (two periods) to describe *who* benefited
from growth. They build on the same weighted-quantile machinery as the
growth incidence curve.
"""

from __future__ import annotations

import numpy as np

from .poverty import _prepare_sample_weights


def _weighted_mean(values, weights=None):
    x = np.asarray(values, dtype=float)
    w = _prepare_sample_weights(weights, x.shape[0])
    return float((w * x).sum() / w.sum())


def bottom_share_growth(values_1, values_2, weights_1=None, weights_2=None,
                        bottom=0.40):
    """Growth rate of the mean income of the poorest `bottom` fraction.

    The World Bank's "shared prosperity" indicator is the growth of the
    bottom 40% (bottom=0.40). Returns the growth rate from period 1 to
    period 2 of the mean income among the poorest `bottom` share.

    Parameters
    ----------
    values_1, values_2 : array-like
        Income distributions in period 1 and period 2.
    weights_1, weights_2 : array-like, optional
        Survey weights for each period.
    bottom : float, default 0.40
        The bottom fraction of the population, in (0, 1].

    Returns
    -------
    float
        Growth rate (period2 / period1 - 1) of the bottom group's mean.
    """
    if not (0 < bottom <= 1):
        raise ValueError("bottom must be in the interval (0, 1].")
    m1 = _bottom_mean(values_1, weights_1, bottom)
    m2 = _bottom_mean(values_2, weights_2, bottom)
    if m1 == 0:
        raise ValueError("bottom-group mean in period 1 is zero.")
    return float(m2 / m1 - 1.0)


def _bottom_mean(values, weights, bottom):
    """Weighted mean income of the poorest `bottom` share of the population."""
    x = np.asarray(values, dtype=float)
    w = _prepare_sample_weights(weights, x.shape[0])
    order = np.argsort(x)
    x, w = x[order], w[order]
    cum = np.cumsum(w) / w.sum()
    # Fractional inclusion of the unit that straddles the cutoff.
    incl = np.clip((bottom - (cum - w / w.sum())) / (w / w.sum()), 0.0, 1.0)
    eff = w * incl
    if eff.sum() == 0:
        return 0.0
    return float((eff * x).sum() / eff.sum())


def mean_growth_rate(values_1, values_2, weights_1=None, weights_2=None):
    """Growth rate of the overall (weighted) mean income."""
    m1 = _weighted_mean(values_1, weights_1)
    m2 = _weighted_mean(values_2, weights_2)
    if m1 == 0:
        raise ValueError("mean income in period 1 is zero.")
    return float(m2 / m1 - 1.0)


def is_pro_poor(values_1, values_2, weights_1=None, weights_2=None,
                bottom=0.40):
    """Whether growth was pro-poor by the shared-prosperity premium.

    Returns a small result: the bottom-group growth, the overall mean
    growth, the premium (bottom minus mean), and a boolean flag that is
    True when the bottom grew faster than the mean.
    """
    b = bottom_share_growth(values_1, values_2, weights_1, weights_2, bottom)
    m = mean_growth_rate(values_1, values_2, weights_1, weights_2)
    return {
        "bottom_growth": b,
        "mean_growth": m,
        "premium": b - m,
        "pro_poor": b > m,
    }