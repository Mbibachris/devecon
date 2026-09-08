import pytest

from devecon import alkire_foster

# Reused deprivation matrix (matches the worked example):
#   person 0: deprived in all 3   -> score 1.00
#   person 1: deprived in 2 of 3  -> score 0.67
#   person 2: deprived in 1 of 3  -> score 0.33
#   person 3: deprived in none    -> score 0.00
DEP = [
    [1, 1, 1],
    [1, 1, 0],
    [1, 0, 0],
    [0, 0, 0],
]


def test_headcount_intensity_and_m0():
    res = alkire_foster(DEP, cutoff_k=1/3)
    assert res.H == 0.75                    # 3 of 4 people poor
    assert res.A == pytest.approx(2/3)      # mean score among the poor
    assert res.M0 == pytest.approx(0.5)     # (1 + 2/3 + 1/3) / 4


def test_identity_M0_equals_H_times_A():
    res = alkire_foster(DEP, cutoff_k=1/3)
    assert res.M0 == pytest.approx(res.H * res.A)


def test_censoring_drops_the_barely_deprived():
    # Raise k to 0.5: person 2 (score 1/3) is no longer poor and is censored.
    res = alkire_foster(DEP, cutoff_k=0.5)
    assert res.H == 0.5                      # only persons 0 and 1 poor
    assert res.M0 == pytest.approx(5/12)     # (1 + 2/3) / 4
    # Without censoring, M0 would wrongly be (1 + 2/3 + 1/3)/4 = 0.5.
    assert res.M0 != pytest.approx(0.5)


def test_indicator_weights_matter():
    dep = [[1, 0], [0, 1]]
    res = alkire_foster(dep, cutoff_k=0.5, indicator_weights=[0.75, 0.25])
    # scores: person 0 = 0.75 (poor), person 1 = 0.25 (not poor)
    assert res.H == 0.5
    assert res.M0 == pytest.approx(0.375)
    assert res.A == pytest.approx(0.75)


def test_sample_weights_change_incidence():
    dep = [[1], [0]]
    equal = alkire_foster(dep, cutoff_k=0.5)
    weighted = alkire_foster(dep, cutoff_k=0.5, sample_weights=[3, 1])
    assert equal.H == 0.5
    assert weighted.H == 0.75                # the poor person now counts 3x
    assert weighted.M0 == pytest.approx(0.75)


def test_no_one_poor():
    res = alkire_foster([[0, 0], [0, 0]], cutoff_k=0.5)
    assert res.H == 0.0
    assert res.M0 == 0.0
    assert res.A == 0.0                      # convention: 0 when there are no poor


def test_bad_cutoff_raises():
    with pytest.raises(ValueError):
        alkire_foster(DEP, cutoff_k=0)       # must be > 0
    with pytest.raises(ValueError):
        alkire_foster(DEP, cutoff_k=1.5)     # must be <= 1


def test_non_binary_matrix_raises():
    with pytest.raises(ValueError):
        alkire_foster([[0, 2], [1, 0]], cutoff_k=0.5)