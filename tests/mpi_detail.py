"""Alkire-Foster decomposition: censored headcounts and indicator contributions.

After alkire_foster gives the headline M0, policy work needs to know *which
indicators drive it* and *how much each contributes*. These functions break
M0 apart without recomputing it, using the same censoring logic.
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np

from .poverty import _prepare_sample_weights

AFDecomposition = namedtuple(
    "AFDecomposition",
    ["M0", "censored_headcounts", "contributions", "indicator_names"],
)


def af_decompose(deprivations, cutoff_k, indicator_weights=None,
                 sample_weights=None, indicator_names=None):
    """Decompose the adjusted headcount M0 by indicator.

    Returns, for each indicator: its *censored headcount ratio* (the
    weighted share of people who are BOTH multidimensionally poor AND
    deprived in that indicator), and its *contribution* to M0 (the share
    of M0 attributable to that indicator). Contributions sum to 1.

    The identity used:  M0 = sum_j ( w_j * censored_headcount_j ),
    where w_j is the indicator weight. Each indicator's contribution is
    that term divided by M0.

    Parameters
    ----------
    deprivations : array-like, shape (n, d)
        0/1 deprivation matrix.
    cutoff_k : float
        Poverty cutoff in (0, 1].
    indicator_weights : array-like, shape (d,), optional
        Indicator weights; equal by default, rescaled to sum to 1.
    sample_weights : array-like, shape (n,), optional
        Survey weights across people.
    indicator_names : list of str, optional
        Names for labelling the output; defaults to index positions.

    Returns
    -------
    AFDecomposition
        M0, per-indicator censored headcounts, per-indicator contributions
        (summing to 1 when M0 > 0), and indicator names.
    """
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

    if indicator_names is None:
        indicator_names = [f"indicator_{j}" for j in range(d)]
    elif len(indicator_names) != d:
        raise ValueError(f"indicator_names must have length {d}.")

    # Identify the poor (dual cutoff), then censor: keep only the poor.
    scores = dep @ w
    poor = scores >= cutoff_k
    total = s.sum()

    # Censored deprivation matrix: non-poor rows zeroed out.
    censored_dep = dep * poor[:, None]

    # Censored headcount ratio per indicator: weighted share poor & deprived.
    censored_headcounts = (s[:, None] * censored_dep).sum(axis=0) / total

    # M0 = sum_j w_j * censored_headcount_j
    M0 = float((w * censored_headcounts).sum())

    # Contribution of each indicator to M0.
    if M0 > 0:
        contributions = (w * censored_headcounts) / M0
    else:
        contributions = np.zeros(d, dtype=float)

    return AFDecomposition(
        M0=M0,
        censored_headcounts=censored_headcounts,
        contributions=contributions,
        indicator_names=list(indicator_names),
    )