"""Identification step: turn achievements into 0/1 deprivation flags.

The Alkire-Foster method separates *identification* (deciding who is
deprived in each indicator) from *aggregation* (combining those into
H, A, M0). These helpers do identification: each turns one column of
real achievements into deprivation flags (1 = deprived, 0 = not,
NaN = unknown) that alkire_foster then consumes.
"""

from __future__ import annotations

import numpy as np

# Each allowed operator string maps to the comparison it performs.
_OPS = {
    "<": np.less,
    "<=": np.less_equal,
    ">": np.greater,
    ">=": np.greater_equal,
}


def deprive(values, op, cutoff):
    """Flag deprivation by a threshold on a scale.

    A unit is deprived where ``values op cutoff`` is True. Stating the
    operator makes the rule explicit: no ambiguity about direction (is
    low or high deprived?) or about whether the boundary itself counts.

    Works for continuous achievements (years of schooling ``< 6``) and
    for coded ordinal / Likert scales (food insecurity ``>= 2``) alike.

    Parameters
    ----------
    values : array-like
        Achievements for one indicator. Missing entries (NaN) yield NaN.
    op : {"<", "<=", ">", ">="}
        Comparison defining deprivation.
    cutoff : float
        The deprivation threshold, in the units of ``values``.

    Returns
    -------
    numpy.ndarray of float
        1.0 where deprived, 0.0 where not, NaN where the input was NaN.
    """
    if op not in _OPS:
        raise ValueError(f"op must be one of {sorted(_OPS)}; got {op!r}.")

    arr = np.asarray(values, dtype=float)
    flags = _OPS[op](arr, cutoff).astype(float)
    flags[np.isnan(arr)] = np.nan   # unknown achievement -> unknown deprivation
    return flags


def deprive_in(values, categories):
    """Flag deprivation by membership in a set of categories.

    For unordered categorical indicators (water source, sanitation type,
    cooking fuel), where deprivation is defined by the value falling in a
    set of "deprived" categories rather than by a threshold.

    Parameters
    ----------
    values : array-like
        Categorical achievements. None / NaN yield NaN.
    categories : iterable
        The values that count as deprived.

    Returns
    -------
    numpy.ndarray of float
        1.0 where the value is a deprived category, 0.0 where not,
        NaN where the input was missing.
    """
    deprived = set(categories)
    out = np.empty(len(values), dtype=float)
    for i, v in enumerate(values):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            out[i] = np.nan
        else:
            out[i] = 1.0 if v in deprived else 0.0
    return out