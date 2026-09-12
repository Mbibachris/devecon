"""Consistent, human-readable summaries of devecon results.

Every measure in devecon returns a plain value or a small named tuple.
These helpers turn any of those into a clean, consistent text summary --
without changing what the measures return. summary() dispatches on the
result type; describe() gives a quick weighted description of a raw
distribution.
"""

from __future__ import annotations

import numpy as np

from .poverty import _prepare_sample_weights


def describe(values, weights=None):
    """Quick weighted description of a distribution.

    Returns a dict of n, weighted mean, median, min, max, and standard
    deviation -- a fast orientation before computing indices.

    Parameters
    ----------
    values : array-like
        The distribution.
    weights : array-like, optional
        Survey weights; equal by default.

    Returns
    -------
    dict
        {"n", "mean", "median", "min", "max", "std"}.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 1:
        raise ValueError("values must be 1-D.")
    w = _prepare_sample_weights(weights, x.shape[0])

    wsum = w.sum()
    mean = (w * x).sum() / wsum
    var = (w * (x - mean) ** 2).sum() / wsum

    # Weighted median via the 50% point of the cumulative weight.
    order = np.argsort(x)
    xs, ws = x[order], w[order]
    cum = np.cumsum(ws) - 0.5 * ws
    median = float(np.interp(0.5, cum / wsum, xs))

    return {
        "n": int(x.shape[0]),
        "mean": float(mean),
        "median": median,
        "min": float(x.min()),
        "max": float(x.max()),
        "std": float(np.sqrt(var)),
    }


def _fmt(value, places=4):
    return f"{value:.{places}f}"


def summary(result, title=None):
    """Return a clean text summary for any devecon result.

    Dispatches on the result type: AF results (with H, A, M0 fields),
    AF decompositions (with contributions), plain floats (a single index),
    and dominance results. Returns a multi-line string; does not print.

    Parameters
    ----------
    result : object
        A value or named tuple returned by a devecon measure.
    title : str, optional
        A heading for the summary block.

    Returns
    -------
    str
        Formatted, human-readable summary.
    """
    lines = []
    if title:
        lines.append(title)
        lines.append("-" * len(title))

    # AF poverty result: has H, A, M0.
    if _has_fields(result, ("H", "A", "M0")) and not _has_fields(
            result, ("contributions",)):
        lines.append(f"Incidence (H):        {_fmt(result.H)}")
        lines.append(f"Intensity (A):        {_fmt(result.A)}")
        lines.append(f"Adjusted headcount M0: {_fmt(result.M0)}")

    # AF decomposition: has contributions + indicator_names.
    elif _has_fields(result, ("M0", "contributions", "indicator_names")):
        lines.append(f"Adjusted headcount M0: {_fmt(result.M0)}")
        lines.append("Contributions to M0:")
        pairs = sorted(zip(result.indicator_names, result.contributions),
                       key=lambda t: -t[1])
        for name, contrib in pairs:
            lines.append(f"  {name:<20} {_fmt(contrib * 100, 1)}%")

    # Dominance result.
    elif _has_fields(result, ("dominates_1_over_2", "crosses")):
        if result.crosses:
            lines.append("No dominance: the curves cross.")
        elif result.dominates_1_over_2 and result.dominates_2_over_1:
            lines.append("Distributions are equivalent (mutual dominance).")
        elif result.dominates_1_over_2:
            lines.append("Distribution 1 dominates distribution 2.")
        elif result.dominates_2_over_1:
            lines.append("Distribution 2 dominates distribution 1.")

    # A plain scalar index.
    elif isinstance(result, (int, float, np.floating)):
        lines.append(f"Value: {_fmt(float(result))}")

    else:
        lines.append(repr(result))

    return "\n".join(lines)


def _has_fields(obj, fields):
    """True if obj is a named-tuple-like with all the given fields."""
    return all(hasattr(obj, f) for f in fields)