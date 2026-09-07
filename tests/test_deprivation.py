import numpy as np
import pytest

from devecon import deprive, deprive_in


def test_below_cutoff_flags_low_values():
    # Deprived if years of schooling < 6.
    out = deprive([3, 6, 9], "<", 6)
    assert list(out) == [1.0, 0.0, 0.0]      # exactly 6 is NOT deprived


def test_boundary_strict_vs_inclusive():
    # The value sitting exactly on the cutoff is the classic trap.
    assert deprive([6], "<", 6)[0] == 0.0    # 6 < 6  is False
    assert deprive([6], "<=", 6)[0] == 1.0   # 6 <= 6 is True


def test_above_cutoff_when_high_is_bad():
    # Deprived if more than 3 people per room (crowding).
    out = deprive([2, 3, 4], ">", 3)
    assert list(out) == [0.0, 0.0, 1.0]


def test_ordinal_likert_threshold():
    # Food insecurity coded 0..4; deprived if >= 2.
    out = deprive([0, 1, 2, 3, 4], ">=", 2)
    assert list(out) == [0.0, 0.0, 1.0, 1.0, 1.0]


def test_missing_propagates_as_nan():
    out = deprive([3, np.nan, 9], "<", 6)
    assert out[0] == 1.0
    assert np.isnan(out[1])                   # unknown -> unknown, not 0
    assert out[2] == 0.0


def test_bad_operator_raises():
    with pytest.raises(ValueError):
        deprive([1, 2], "==", 1)


def test_deprive_in_membership():
    src = ["piped", "well", "surface", "piped"]
    out = deprive_in(src, {"well", "surface"})
    assert list(out) == [0.0, 1.0, 1.0, 0.0]


def test_deprive_in_missing_is_nan():
    out = deprive_in(["piped", None], {"surface"})
    assert out[0] == 0.0
    assert np.isnan(out[1])


def test_output_feeds_alkire_foster():
    # The payoff: coarsened columns assemble into a matrix AF accepts.
    from devecon import alkire_foster
    school = deprive([3, 6, 9, 12], "<", 6)                     # [1,0,0,0]
    water = deprive_in(["surface", "surface", "piped", "piped"],
                       {"surface"})                             # [1,1,0,0]
    matrix = np.column_stack([school, water])
    res = alkire_foster(matrix, cutoff_k=0.5)
    # scores: 1.0, 0.5, 0, 0 ; poor if >= 0.5 -> first two
    assert res.H == 0.5
    assert res.M0 == pytest.approx(0.375)