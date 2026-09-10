import numpy as np
import pytest

from devecon import atkinson, generalized_entropy, gini

# ---- Gini ----------------------------------------------------------------

def test_gini_perfect_equality():
    assert gini([5, 5, 5, 5]) == 0.0


def test_gini_maximal_inequality():
    # One unit holds everything: theoretical max is (n-1)/n.
    assert gini([0, 0, 0, 100]) == pytest.approx(0.75)


def test_gini_known_value():
    # [1,2,3,4,5] has a documented Gini of 4/15.
    assert gini([1, 2, 3, 4, 5]) == pytest.approx(4 / 15)


def test_gini_scale_invariant():
    # Multiplying every income by a constant leaves Gini unchanged.
    base = gini([1, 2, 3, 4, 5])
    scaled = gini([10, 20, 30, 40, 50])
    assert scaled == pytest.approx(base)


def test_gini_weights():
    # Two people at 10 (weight 1 each) equals one person at 10 weight 2.
    unweighted = gini([10, 10, 40])
    weighted = gini([10, 40], weights=[2, 1])
    assert weighted == pytest.approx(unweighted)


def test_gini_negative_raises():
    with pytest.raises(ValueError):
        gini([1, -2, 3])


# ---- Generalized Entropy -------------------------------------------------

@pytest.mark.parametrize("alpha", [0.0, 1.0, 2.0, -1.0, 0.5])
def test_ge_zero_at_equality(alpha):
    assert generalized_entropy([4, 4, 4], alpha=alpha) == pytest.approx(0.0)


def test_ge2_equals_half_cv_squared():
    # GE(2) = 0.5 * CV^2 ; for [1,2,3] that is 1/12.
    assert generalized_entropy([1, 2, 3], alpha=2) == pytest.approx(1 / 12)


def test_ge_positive_under_inequality():
    for alpha in (0.0, 1.0, 2.0):
        assert generalized_entropy([1, 2, 4, 8], alpha=alpha) > 0


def test_ge_requires_positive():
    with pytest.raises(ValueError):
        generalized_entropy([0, 1, 2], alpha=1)


# ---- Atkinson ------------------------------------------------------------

@pytest.mark.parametrize("eps", [0.5, 1.0, 2.0])
def test_atkinson_zero_at_equality(eps):
    assert atkinson([7, 7, 7], epsilon=eps) == pytest.approx(0.0, abs=1e-12)


def test_atkinson_known_value():
    # A(1) of [1,2,4]: 1 - geomean/mean = 1 - 2/(7/3) = 1 - 6/7.
    assert atkinson([1, 2, 4], epsilon=1) == pytest.approx(1 - 6 / 7)


def test_atkinson_negative_epsilon_raises():
    with pytest.raises(ValueError):
        atkinson([1, 2, 3], epsilon=-1)


def test_atkinson_requires_positive():
    with pytest.raises(ValueError):
        atkinson([0, 1, 2], epsilon=1)


# ---- The identity linking the two families -------------------------------

def test_atkinson1_equals_ge0_identity():
    # A(1) = 1 - exp(-GE(0)). Both derive from the geometric/arithmetic
    # mean ratio, so this must hold exactly for any positive distribution.
    x = [1, 2, 4, 8, 3, 5]
    ge0 = generalized_entropy(x, alpha=0)
    a1 = atkinson(x, epsilon=1)
    assert a1 == pytest.approx(1 - np.exp(-ge0))