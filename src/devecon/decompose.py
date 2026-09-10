"""Subgroup decompositions of poverty and inequality.

Poverty (FGT) is additively decomposable: total poverty is the
population-weighted sum of subgroup poverty. Generalized Entropy
inequality decomposes into a within-group and a between-group component.
These functions report those breakdowns by a grouping variable.
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np

from .inequality import generalized_entropy
from .poverty import _prepare_sample_weights, fgt

FGTDecomposition = namedtuple(
    "FGTDecomposition",
    ["total", "group_indices", "population_shares", "contributions", "groups"],
)
TheilDecomposition = namedtuple(
    "TheilDecomposition",
    ["total", "within", "between", "group_indices", "groups"],
)


def _as_groups(groups):
    g = np.asarray(groups)
    labels = list(dict.fromkeys(g.tolist()))  # unique, order-preserving
    return g, labels


def fgt_by_group(income, poverty_line, groups, alpha=0.0, weights=None):
    """Decompose the FGT index by population subgroup.

    Total FGT = sum_g (population share_g * FGT_g). Reports each group's
    own FGT, its population share, and its contribution to total poverty.

    Returns
    -------
    FGTDecomposition
        total, per-group indices, population shares, contributions
        (summing to total), and group labels.
    """
    y = np.asarray(income, dtype=float)
    w = _prepare_sample_weights(weights, y.shape[0])
    g, labels = _as_groups(groups)
    if g.shape[0] != y.shape[0]:
        raise ValueError("groups must have the same length as income.")

    wsum = w.sum()
    group_indices = {}
    pop_shares = {}
    contributions = {}
    for lab in labels:
        mask = g == lab
        pop_shares[lab] = w[mask].sum() / wsum
        group_indices[lab] = fgt(y[mask], poverty_line, alpha=alpha,
                                 weights=w[mask])
        contributions[lab] = pop_shares[lab] * group_indices[lab]

    total = float(sum(contributions.values()))
    return FGTDecomposition(
        total=total,
        group_indices=group_indices,
        population_shares=pop_shares,
        contributions=contributions,
        groups=labels,
    )


def theil_by_group(values, groups, weights=None):
    """Within/between decomposition of the Theil-T index GE(1).

    GE(1)_total = within + between, where:
      within  = sum_g s_g * (mu_g/mu) * GE(1)_g
      between = sum_g s_g * (mu_g/mu) * ln(mu_g/mu)
    with s_g the population share and mu_g the group mean.

    Returns
    -------
    TheilDecomposition
        total, within component, between component, per-group GE(1), and
        group labels.
    """
    x = np.asarray(values, dtype=float)
    if np.any(x <= 0):
        raise ValueError("all values must be strictly positive.")
    w = _prepare_sample_weights(weights, x.shape[0])
    g, labels = _as_groups(groups)
    if g.shape[0] != x.shape[0]:
        raise ValueError("groups must have the same length as values.")

    wsum = w.sum()
    mu = (w * x).sum() / wsum

    within = 0.0
    between = 0.0
    group_indices = {}
    for lab in labels:
        mask = g == lab
        wg = w[mask]
        xg = x[mask]
        s_g = wg.sum() / wsum
        mu_g = (wg * xg).sum() / wg.sum()
        ge_g = generalized_entropy(xg, alpha=1.0, weights=wg)
        group_indices[lab] = ge_g
        within += s_g * (mu_g / mu) * ge_g
        between += s_g * (mu_g / mu) * np.log(mu_g / mu)

    total = generalized_entropy(x, alpha=1.0, weights=w)
    return TheilDecomposition(
        total=float(total),
        within=float(within),
        between=float(between),
        group_indices=group_indices,
        groups=labels,
    )