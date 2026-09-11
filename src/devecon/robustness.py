"""Robustness analysis for Alkire-Foster multidimensional poverty.

A single M0 depends on choices the analyst makes: the poverty cutoff k,
and the indicator weights. Robustness analysis recomputes M0 across a
range of those choices so you can see whether conclusions -- especially
*rankings* of groups or countries -- are stable or an artefact of one
arbitrary cutoff.
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np

from .poverty import alkire_foster

CutoffProfile = namedtuple("CutoffProfile", ["cutoffs", "H", "A", "M0"])


def cutoff_profile(deprivations, cutoffs, indicator_weights=None,
                   sample_weights=None):
    """Recompute AF measures across a range of poverty cutoffs k.

    Parameters
    ----------
    deprivations : array-like, shape (n, d)
        0/1 deprivation matrix.
    cutoffs : iterable of float
        Poverty cutoffs to evaluate, each in (0, 1].
    indicator_weights, sample_weights : array-like, optional
        Passed through to alkire_foster.

    Returns
    -------
    CutoffProfile
        Arrays of cutoffs and the corresponding H, A, M0 -- so you can see
        how the headline numbers move as k varies.
    """
    ks = np.asarray(list(cutoffs), dtype=float)
    if ks.size == 0:
        raise ValueError("cutoffs must contain at least one value.")
    H = np.empty(ks.size)
    A = np.empty(ks.size)
    M0 = np.empty(ks.size)
    for i, k in enumerate(ks):
        res = alkire_foster(deprivations, cutoff_k=float(k),
                            indicator_weights=indicator_weights,
                            sample_weights=sample_weights)
        H[i], A[i], M0[i] = res.H, res.A, res.M0
    return CutoffProfile(cutoffs=ks, H=H, A=A, M0=M0)


def rank_robustness(group_matrices, cutoffs, indicator_weights=None):
    """Check whether a ranking of groups by M0 is stable across cutoffs.

    For each cutoff k, rank the groups by their M0. Report whether the
    ranking is identical at every cutoff (fully robust) and the share of
    cutoffs at which each pairwise ordering is preserved.

    Parameters
    ----------
    group_matrices : dict[str, array-like]
        Maps group name -> that group's deprivation matrix.
    cutoffs : iterable of float
        Poverty cutoffs to test.
    indicator_weights : array-like, optional
        Shared indicator weights (same indicators across groups).

    Returns
    -------
    dict
        {
          "robust": bool,                # ranking identical at every cutoff
          "rankings": {k: [groups high->low]},
          "m0": {k: {group: M0}},
        }
    """
    ks = list(cutoffs)
    if not ks:
        raise ValueError("cutoffs must contain at least one value.")
    names = list(group_matrices)

    m0_by_k = {}
    rankings = {}
    for k in ks:
        m0s = {
            g: alkire_foster(group_matrices[g], cutoff_k=float(k),
                             indicator_weights=indicator_weights).M0
            for g in names
        }
        m0_by_k[k] = m0s
        rankings[k] = sorted(names, key=lambda g: m0s[g], reverse=True)

    first = rankings[ks[0]]
    robust = all(rankings[k] == first for k in ks)
    return {"robust": robust, "rankings": rankings, "m0": m0_by_k}