"""Concentration curves and indices for targeting analysis.

A Lorenz curve ranks people by their own income. A *concentration* curve
ranks people by one variable (living standard) but plots the cumulative
share of a *different* variable (a subsidy, a health outcome, a tax). It
answers: is this benefit pro-poor (concentrated among the worse-off) or
pro-rich? The concentration index summarises that in one number in
[-1, 1]: negative = pro-poor, positive = pro-rich.
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np

from .poverty import _prepare_sample_weights

ConcentrationCurve = namedtuple(
    "ConcentrationCurve", ["rank_share", "value_share"])


def concentration_curve(ranking, target, weights=None):
    """Concentration curve of `target`, with people ranked by `ranking`.

    Parameters
    ----------
    ranking : array-like
        The living-standards variable used to order people (poorest first).
    target : array-like
        The variable whose distribution is being examined (e.g. a subsidy).
        Must be non-negative.
    weights : array-like, optional
        Survey weights; equal by default.

    Returns
    -------
    ConcentrationCurve
        Cumulative population share (ordered by `ranking`) against the
        cumulative share of `target`. Above the 45-degree line means the
        target is concentrated among the poor (pro-poor).
    """
    r = np.asarray(ranking, dtype=float)
    t = np.asarray(target, dtype=float)
    if r.shape != t.shape:
        raise ValueError("ranking and target must have the same shape.")
    if np.any(t < 0):
        raise ValueError("target must be non-negative.")
    w = _prepare_sample_weights(weights, r.shape[0])

    order = np.argsort(r)
    t, w = t[order], w[order]
    cum_pop = np.concatenate([[0.0], np.cumsum(w)]) / w.sum()
    total = (w * t).sum()
    cum_val = np.concatenate([[0.0], np.cumsum(w * t)])
    cum_val = cum_val / total if total > 0 else np.zeros_like(cum_val)
    return ConcentrationCurve(rank_share=cum_pop, value_share=cum_val)


def concentration_index(ranking, target, weights=None):
    """Concentration index: twice the area between the curve and diagonal.

    CI = 1 - 2 * (area under the concentration curve).

    Ranges in [-1, 1]. Negative means `target` is concentrated among the
    poor (pro-poor); positive means concentrated among the rich; 0 means
    proportional. When `ranking` and `target` are the same variable, CI
    equals the Gini coefficient.

    Parameters
    ----------
    ranking : array-like
        Living-standards variable used to order people.
    target : array-like
        Non-negative variable being examined.
    weights : array-like, optional
        Survey weights; equal by default.

    Returns
    -------
    float
        Concentration index in [-1, 1].
    """
    curve = concentration_curve(ranking, target, weights)
    p, v = curve.rank_share, curve.value_share
    # Trapezoidal area under the curve, computed explicitly (version-proof).
    area = np.sum((p[1:] - p[:-1]) * (v[1:] + v[:-1]) / 2.0)
    return float(1.0 - 2.0 * area)