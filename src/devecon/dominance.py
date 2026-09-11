"""Stochastic dominance tests for welfare comparisons.

Dominance lets you rank two distributions *without* committing to a single
poverty line or inequality index. If distribution A first-order dominates
B, then A has less poverty than B for EVERY poverty line and every poverty
measure in a broad class -- a far more robust conclusion than a
single-line comparison. Second-order dominance ranks some cases that
first-order cannot, by accounting for the depth of shortfalls.
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np

from .poverty import _prepare_sample_weights

DominanceResult = namedtuple(
    "DominanceResult",
    ["dominates_1_over_2", "dominates_2_over_1", "crosses",
     "grid", "curve_1", "curve_2"],
)


def _cdf_on_grid(values, weights, grid):
    """Weighted CDF F(z) = share of population with value <= z, on a grid."""
    x = np.asarray(values, dtype=float)
    w = _prepare_sample_weights(weights, x.shape[0])
    order = np.argsort(x)
    x, w = x[order], w[order]
    cw = np.cumsum(w) / w.sum()
    idx = np.searchsorted(x, grid, side="right")
    return np.where(idx > 0, cw[np.clip(idx - 1, 0, len(cw) - 1)], 0.0)


def _build_grid(x1, x2, n_points):
    lo = min(x1.min(), x2.min())
    hi = max(x1.max(), x2.max())
    return np.linspace(lo, hi, n_points)


def _classify(curve_1, curve_2, tol):
    """Given two 'lower is better' curves, decide the dominance relation.

    For welfare, the distribution with the *lower* curve (CDF or its
    integral) dominates: fewer people below any threshold. Returns the
    (1>2, 2>1, crosses) booleans. Weak dominance -- equal curves count
    as mutual dominance and are not treated as crossing.
    """
    one_below = np.all(curve_1 <= curve_2 + tol)   # 1 dominates 2
    two_below = np.all(curve_2 <= curve_1 + tol)   # 2 dominates 1
    crosses = (not one_below) and (not two_below)
    return bool(one_below), bool(two_below), bool(crosses)


def first_order_dominance(values_1, values_2, weights_1=None, weights_2=None,
                          n_points=100):
    """Test first-order stochastic dominance between two distributions.

    Distribution 1 dominates 2 (less poverty at every line) when its CDF
    lies at or below distribution 2's everywhere: F1(z) <= F2(z) for all z.

    Returns
    -------
    DominanceResult
        dominates_1_over_2 : F1 <= F2 everywhere (weak).
        dominates_2_over_1 : F2 <= F1 everywhere (weak).
        crosses            : the CDFs cross -> no first-order ranking.
        grid, curve_1, curve_2 : evaluation grid and the two CDFs.

    Note: two identical distributions return True for both dominance
    flags (each weakly dominates the other) and crosses=False.
    """
    x1 = np.asarray(values_1, dtype=float)
    x2 = np.asarray(values_2, dtype=float)
    grid = _build_grid(x1, x2, n_points)
    f1 = _cdf_on_grid(x1, weights_1, grid)
    f2 = _cdf_on_grid(x2, weights_2, grid)
    one, two, crosses = _classify(f1, f2, tol=1e-12)
    return DominanceResult(one, two, crosses, grid, f1, f2)


def second_order_dominance(values_1, values_2, weights_1=None, weights_2=None,
                           n_points=200):
    """Test second-order stochastic dominance between two distributions.

    Uses the running integral of each CDF (area up to each point), which
    captures both the level and the depth of shortfalls. Distribution 1
    dominates 2 when that integral is everywhere no greater. Second-order
    dominance can rank distributions whose CDFs cross once.

    Returns
    -------
    DominanceResult
        dominates_1_over_2, dominates_2_over_1 : weak second-order flags.
        crosses            : neither integral is everywhere below the other.
        grid, curve_1, curve_2 : grid and the cumulative CDF integrals.
    """
    x1 = np.asarray(values_1, dtype=float)
    x2 = np.asarray(values_2, dtype=float)
    grid = _build_grid(x1, x2, n_points)
    f1 = _cdf_on_grid(x1, weights_1, grid)
    f2 = _cdf_on_grid(x2, weights_2, grid)

    dz = grid[1] - grid[0]
    i1 = np.concatenate([[0.0], np.cumsum((f1[1:] + f1[:-1]) / 2.0) * dz])
    i2 = np.concatenate([[0.0], np.cumsum((f2[1:] + f2[:-1]) / 2.0) * dz])
    i1, i2 = i1[:len(grid)], i2[:len(grid)]

    one, two, crosses = _classify(i1, i2, tol=1e-10)
    return DominanceResult(one, two, crosses, grid, i1, i2)