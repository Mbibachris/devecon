"""Assembly step: build a deprivation matrix and weight vector from a spec.

 per-column coarsening (deprive, deprive_in). This module
stitches many columns into one aligned matrix, and builds the matching
indicator-weight vector from a dimension structure. A single spec drives
both, so matrix columns and weights are guaranteed to line up -- the
alignment bug (right weights, wrong order) becomes impossible.
"""

from __future__ import annotations

import numpy as np

from .deprivation import deprive, deprive_in


def build_nested_weights(structure):
    """Build a flat indicator-weight vector from a dimension structure.

    Each dimension carries a weight; its indicators split that weight
    (equally by default, or by explicit per-indicator weights). The
    result is flattened and normalized to sum to 1, with each
    dimension's indicators summing to that dimension's share -- the
    invariant that prevents silent reweighting.

    Parameters
    ----------
    structure : dict
        Maps dimension name -> spec dict with keys:
          "weight"     : float, the dimension's weight (relative; the
                         whole structure is renormalized to sum to 1).
          "indicators" : list of indicator names (equal split), OR
                         dict of indicator name -> weight (explicit split).

    Returns
    -------
    names : list of str
        Indicator names, in order.
    weights : numpy.ndarray
        Weights aligned to names, summing to 1.

    Example
    -------
    >>> structure = {
    ...     "health":    {"weight": 1/3, "indicators": ["nutrition", "mortality"]},
    ...     "education": {"weight": 1/3, "indicators": ["years", "attendance"]},
    ...     "living":    {"weight": 1/3, "indicators": ["water", "electricity"]},
    ... }
    >>> names, w = build_nested_weights(structure)
    """
    if not structure:
        raise ValueError("structure must contain at least one dimension.")

    names = []
    raw = []
    dim_total = sum(spec["weight"] for spec in structure.values())
    if dim_total <= 0:
        raise ValueError("dimension weights must sum to a positive number.")

    for dim, spec in structure.items():
        dw = spec["weight"] / dim_total          # normalized dimension share
        inds = spec["indicators"]

        if isinstance(inds, dict):               # explicit within-dimension weights
            sub_total = sum(inds.values())
            if sub_total <= 0:
                raise ValueError(f"indicator weights in '{dim}' must be positive.")
            for name, iw in inds.items():
                names.append(name)
                raw.append(dw * iw / sub_total)
        else:                                    # equal split across indicators
            if len(inds) == 0:
                raise ValueError(f"dimension '{dim}' has no indicators.")
            share = dw / len(inds)
            for name in inds:
                names.append(name)
                raw.append(share)

    if len(names) != len(set(names)):
        raise ValueError("indicator names must be unique across all dimensions.")

    weights = np.asarray(raw, dtype=float)
    return names, weights


def build_deprivation_matrix(data, spec, missing="raise"):
    """Assemble an aligned deprivation matrix from a DataFrame and a spec.

    Parameters
    ----------
    data : pandas.DataFrame
        Survey data; columns are achievement variables.
    spec : list of dict
        One entry per indicator, in the desired column order. Each dict:
          "name"   : indicator name (should match a weight name).
          "column" : column in `data` holding the achievement.
          Then EITHER a threshold rule:
            "op"     : one of "<", "<=", ">", ">="
            "cutoff" : the threshold
          OR a categorical rule:
            "deprived_categories" : iterable of deprived values
    missing : {"raise", "drop", "deprived=0"}, default "raise"
        How to resolve unknown (NaN) deprivations:
          "raise"      -> error if any deprivation is unknown.
          "drop"       -> drop rows with any unknown deprivation.
          "deprived=0" -> treat unknown as not deprived (understates poverty).

    Returns
    -------
    names : list of str
        Indicator names, in column order.
    matrix : numpy.ndarray, shape (n, d)
        0/1 deprivation matrix, ready for alkire_foster.
    kept : numpy.ndarray of bool
        Row mask indicating which input rows are present in `matrix`
        (all True unless missing="drop").
    """
    try:
        import pandas as pd  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "build_deprivation_matrix needs pandas. "
            'Install it with:  pip install "devecon[pandas]"'
        ) from exc

    if missing not in {"raise", "drop", "deprived=0"}:
        raise ValueError(
            'missing must be "raise", "drop", or "deprived=0"; '
            f"got {missing!r}."
        )
    if not spec:
        raise ValueError("spec must list at least one indicator.")

    names = []
    columns = []
    for item in spec:
        name = item["name"]
        col = data[item["column"]].to_numpy()

        if "deprived_categories" in item:
            flags = deprive_in(col, item["deprived_categories"])
        else:
            flags = deprive(col, item["op"], item["cutoff"])

        names.append(name)
        columns.append(flags)

    matrix = np.column_stack(columns)
    unknown = np.isnan(matrix)                    # (n, d) mask of unknowns

    if missing == "raise":
        if unknown.any():
            bad = int(np.unique(np.where(unknown)[0]).size)
            raise ValueError(
                f"{bad} row(s) have unknown deprivations. Resolve them with "
                'missing="drop" or missing="deprived=0", or clean the data.'
            )
        kept = np.ones(matrix.shape[0], dtype=bool)

    elif missing == "drop":
        kept = ~unknown.any(axis=1)
        matrix = matrix[kept]

    else:  # "deprived=0"
        matrix = np.where(unknown, 0.0, matrix)
        kept = np.ones(matrix.shape[0], dtype=bool)

    return names, matrix, kept