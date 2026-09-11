import numpy as np
import pytest

from devecon import concentration_curve, concentration_index, gini

# ---- concentration_index -------------------------------------------------

def test_ci_equals_gini_when_target_is_ranking():
    # Ranking by income and examining income itself -> CI collapses to Gini.
    x = [1, 2, 4, 8, 3, 5, 7]
    assert concentration_index(x, x) == pytest.approx(gini(x))


def test_ci_zero_for_proportional_target():
    # Everyone receives the same amount -> no concentration either way.
    assert concentration_index([1, 2, 3, 4, 5],
                               [10, 10, 10, 10, 10]) == pytest.approx(0.0)


def test_ci_negative_when_pro_poor():
    # Poorest (ranked first) receive the most -> pro-poor -> negative CI.
    ci = concentration_index([1, 2, 3, 4, 5], [50, 40, 30, 20, 10])
    assert ci < 0


def test_ci_positive_when_pro_rich():
    ci = concentration_index([1, 2, 3, 4, 5], [10, 20, 30, 40, 50])
    assert ci > 0


def test_ci_sign_symmetry():
    # Reversing the target flips the sign of the index.
    rank = [1, 2, 3, 4, 5]
    a = concentration_index(rank, [50, 40, 30, 20, 10])
    b = concentration_index(rank, [10, 20, 30, 40, 50])
    assert a == pytest.approx(-b)


def test_ci_negative_target_raises():
    with pytest.raises(ValueError):
        concentration_index([1, 2, 3], [1, -1, 2])


# ---- concentration_curve -------------------------------------------------

def test_curve_endpoints():
    c = concentration_curve([1, 2, 3], [5, 10, 15])
    assert c.rank_share[0] == 0.0
    assert c.value_share[0] == 0.0
    assert c.rank_share[-1] == pytest.approx(1.0)
    assert c.value_share[-1] == pytest.approx(1.0)


def test_pro_poor_curve_above_diagonal():
    c = concentration_curve([1, 2, 3, 4, 5], [50, 40, 30, 20, 10])
    assert np.all(c.value_share >= c.rank_share - 1e-12)


def test_mismatched_shapes_raise():
    with pytest.raises(ValueError):
        concentration_curve([1, 2, 3], [1, 2])