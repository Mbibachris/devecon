import numpy as np
import pytest

from devecon import af_decompose, alkire_foster

# Worked example from the Alkire-Foster build:
#   person 0: deprived in all 3   (score 1.00)
#   person 1: deprived in 2 of 3  (score 0.67)
#   person 2: deprived in 1 of 3  (score 0.33)
#   person 3: deprived in none    (score 0.00)
DEP = [
    [1, 1, 1],
    [1, 1, 0],
    [1, 0, 0],
    [0, 0, 0],
]


def test_decomposition_m0_matches_alkire_foster():
    # The decomposed M0 must equal the headline M0, computed independently.
    res = alkire_foster(DEP, cutoff_k=1/3)
    dec = af_decompose(DEP, cutoff_k=1/3)
    assert dec.M0 == pytest.approx(res.M0)


def test_contributions_sum_to_one():
    dec = af_decompose(DEP, cutoff_k=1/3)
    assert dec.contributions.sum() == pytest.approx(1.0)


def test_censored_headcounts_known_values():
    dec = af_decompose(DEP, cutoff_k=1/3)
    # All three below-cutoff people are poor; deprivation counts among them.
    assert dec.censored_headcounts == pytest.approx([0.75, 0.5, 0.25])


def test_contributions_known_values():
    dec = af_decompose(DEP, cutoff_k=1/3)
    # w = 1/3 each; contributions proportional to censored headcounts.
    assert dec.contributions == pytest.approx([0.5, 1/3, 1/6])


def test_censoring_drops_non_poor_deprivations():
    # Raise k so person 2 is no longer poor; their indicator-0 deprivation
    # must drop out of indicator 0's censored headcount (0.75 -> 0.5).
    dec = af_decompose(DEP, cutoff_k=0.5)
    assert dec.censored_headcounts == pytest.approx([0.5, 0.5, 0.25])


def test_identity_m0_equals_weighted_censored_headcounts():
    dec = af_decompose(DEP, cutoff_k=1/3)
    w = np.array([1/3, 1/3, 1/3])
    assert dec.M0 == pytest.approx((w * dec.censored_headcounts).sum())


def test_names_are_carried_through():
    dec = af_decompose(DEP, cutoff_k=1/3,
                       indicator_names=["school", "water", "health"])
    assert dec.indicator_names == ["school", "water", "health"]


def test_wrong_name_count_raises():
    with pytest.raises(ValueError):
        af_decompose(DEP, cutoff_k=1/3, indicator_names=["only_one"])


def test_no_poor_gives_zero_contributions():
    dec = af_decompose([[0, 0], [0, 0]], cutoff_k=0.5)
    assert dec.M0 == 0.0
    assert np.all(dec.contributions == 0.0)


def test_indicator_weights_shift_contributions():
    # Weight indicator 0 heavily -> it should dominate the contributions.
    dec = af_decompose(DEP, cutoff_k=1/3, indicator_weights=[0.8, 0.1, 0.1])
    assert dec.contributions[0] > dec.contributions[1]
    assert dec.contributions[0] > dec.contributions[2]
    assert dec.contributions.sum() == pytest.approx(1.0)