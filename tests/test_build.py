import numpy as np
import pytest

pd = pytest.importorskip("pandas")   # skip file cleanly if pandas absent

from devecon import (
    alkire_foster,
    build_deprivation_matrix,
    build_nested_weights,
)

# ---- build_nested_weights ------------------------------------------------

def test_equal_split_within_dimensions():
    structure = {
        "health":    {"weight": 1/3, "indicators": ["a", "b"]},
        "education": {"weight": 1/3, "indicators": ["c", "d"]},
        "living":    {"weight": 1/3, "indicators": ["e", "f"]},
    }
    names, w = build_nested_weights(structure)
    assert names == ["a", "b", "c", "d", "e", "f"]
    assert w == pytest.approx([1/6] * 6)
    assert w.sum() == pytest.approx(1.0)


def test_unequal_indicator_counts_preserve_dimension_share():
    # Health has 4 indicators, others 2 -> the 1/8 trap must NOT happen.
    structure = {
        "health":    {"weight": 1/3, "indicators": ["a", "b", "c", "d"]},
        "education": {"weight": 1/3, "indicators": ["e", "f"]},
        "living":    {"weight": 1/3, "indicators": ["g", "h"]},
    }
    _, w = build_nested_weights(structure)
    assert w[:4] == pytest.approx([1/12] * 4)   # health each
    assert w[4:] == pytest.approx([1/6] * 4)    # others each
    assert w[:4].sum() == pytest.approx(1/3)    # dimension share intact
    assert w.sum() == pytest.approx(1.0)


def test_explicit_within_dimension_weights():
    structure = {
        "d1": {"weight": 1/2, "indicators": {"x": 3, "y": 1}},
        "d2": {"weight": 1/2, "indicators": ["z"]},
    }
    names, w = build_nested_weights(structure)
    # d1 share 1/2 split 3:1 -> 3/8, 1/8 ; d2 -> 1/2
    assert dict(zip(names, w)) == pytest.approx({"x": 3/8, "y": 1/8, "z": 1/2})


def test_relative_dimension_weights_renormalize():
    # Weights given as 2 and 2 (not summing to 1) -> renormalized to half each.
    structure = {
        "d1": {"weight": 2, "indicators": ["x", "y"]},
        "d2": {"weight": 2, "indicators": ["z"]},
    }
    names, w = build_nested_weights(structure)
    assert dict(zip(names, w)) == pytest.approx({"x": 1/4, "y": 1/4, "z": 1/2})


def test_duplicate_indicator_names_raise():
    structure = {
        "d1": {"weight": 1/2, "indicators": ["x", "y"]},
        "d2": {"weight": 1/2, "indicators": ["x"]},
    }
    with pytest.raises(ValueError):
        build_nested_weights(structure)


# ---- build_deprivation_matrix --------------------------------------------

def _frame():
    return pd.DataFrame({
        "school_years": [3, 6, 9, 12],
        "water_src":    ["surface", "surface", "piped", "piped"],
    })


def test_matrix_columns_follow_spec_order():
    spec = [
        {"name": "school", "column": "school_years", "op": "<", "cutoff": 6},
        {"name": "water",  "column": "water_src",
         "deprived_categories": {"surface"}},
    ]
    names, matrix, kept = build_deprivation_matrix(_frame(), spec)
    assert names == ["school", "water"]
    assert matrix.tolist() == [[1, 1], [0, 1], [0, 0], [0, 0]]
    assert kept.all()


def test_missing_raise_is_default():
    df = pd.DataFrame({"x": [1.0, np.nan, 5.0]})
    spec = [{"name": "i", "column": "x", "op": "<", "cutoff": 3}]
    with pytest.raises(ValueError):
        build_deprivation_matrix(df, spec)             # default missing="raise"


def test_missing_drop_removes_rows():
    df = pd.DataFrame({"x": [1.0, np.nan, 5.0]})
    spec = [{"name": "i", "column": "x", "op": "<", "cutoff": 3}]
    _, matrix, kept = build_deprivation_matrix(df, spec, missing="drop")
    assert matrix.tolist() == [[1], [0]]               # middle row dropped
    assert kept.tolist() == [True, False, True]


def test_missing_deprived_zero_fills():
    df = pd.DataFrame({"x": [1.0, np.nan, 5.0]})
    spec = [{"name": "i", "column": "x", "op": "<", "cutoff": 3}]
    _, matrix, kept = build_deprivation_matrix(df, spec, missing="deprived=0")
    assert matrix.tolist() == [[1], [0], [0]]          # NaN -> 0
    assert kept.all()


def test_bad_missing_policy_raises():
    df = pd.DataFrame({"x": [1.0]})
    spec = [{"name": "i", "column": "x", "op": "<", "cutoff": 3}]
    with pytest.raises(ValueError):
        build_deprivation_matrix(df, spec, missing="whatever")


# ---- the whole pipeline, spec -> weights -> matrix -> M0 -----------------

def test_end_to_end_alignment():
    df = pd.DataFrame({
        "school_years": [3, 6, 9, 12],
        "water_src":    ["surface", "surface", "piped", "piped"],
    })
    structure = {
        "education": {"weight": 1/2, "indicators": ["school"]},
        "living":    {"weight": 1/2, "indicators": ["water"]},
    }
    names, weights = build_nested_weights(structure)
    spec = [
        {"name": "school", "column": "school_years", "op": "<", "cutoff": 6},
        {"name": "water",  "column": "water_src",
         "deprived_categories": {"surface"}},
    ]
    mat_names, matrix, _ = build_deprivation_matrix(df, spec)

    # The alignment guarantee: matrix columns and weights share an order.
    assert mat_names == names

    res = alkire_foster(matrix, cutoff_k=0.5, indicator_weights=weights)
    # scores: person0 = 1.0, person1 = 0.5, others 0 -> poor if >= 0.5
    assert res.H == 0.5
    assert res.M0 == pytest.approx(0.5 * res.A)   # identity still holds