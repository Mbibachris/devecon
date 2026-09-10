import pytest

from devecon import bottom_share_growth, is_pro_poor, mean_growth_rate

# ---- bottom_share_growth -------------------------------------------------

def test_bottom40_known_value():
    # Bottom 40% of 5 people = poorest 2.
    # P1 [10,20,30,40,50] -> bottom-2 mean 15
    # P2 [20,40,30,40,50] -> sorted (20,30,40,40,50), bottom-2 mean 25
    g = bottom_share_growth([10, 20, 30, 40, 50],
                            [20, 40, 30, 40, 50], bottom=0.40)
    assert g == pytest.approx(25 / 15 - 1)


def test_bottom_full_population_equals_mean_growth():
    # bottom=1.0 covers everyone, so it must equal overall mean growth.
    p1, p2 = [1, 2, 3, 4, 5], [2, 3, 4, 5, 6]
    assert bottom_share_growth(p1, p2, bottom=1.0) == pytest.approx(
        mean_growth_rate(p1, p2))


def test_bottom_uniform_growth_matches_mean():
    # Scaling everyone by the same factor: every subgroup grows equally.
    p1 = [1, 2, 3, 4, 5]
    p2 = [1.2, 2.4, 3.6, 4.8, 6.0]
    assert bottom_share_growth(p1, p2, bottom=0.40) == pytest.approx(0.20)


def test_bottom_invalid_fraction_raises():
    with pytest.raises(ValueError):
        bottom_share_growth([1, 2], [1, 2], bottom=0)
    with pytest.raises(ValueError):
        bottom_share_growth([1, 2], [1, 2], bottom=1.5)


# ---- mean_growth_rate ----------------------------------------------------

def test_mean_growth_known_value():
    # mean 3 -> mean 4 : growth 1/3.
    assert mean_growth_rate([1, 2, 3, 4, 5], [2, 3, 4, 5, 6]) == \
        pytest.approx(1 / 3)


def test_mean_growth_weighted():
    # Weighting shifts the mean and thus the growth rate.
    g = mean_growth_rate([10, 20], [10, 20], weights_1=[1, 3], weights_2=[1, 3])
    assert g == pytest.approx(0.0)   # identical distributions -> no growth


# ---- is_pro_poor ---------------------------------------------------------

def test_is_pro_poor_flags_bottom_outgrowing_mean():
    r = is_pro_poor([1, 2, 3, 4, 5], [2, 3, 4, 5, 6])
    assert r["pro_poor"] is True
    assert r["premium"] > 0
    assert r["bottom_growth"] > r["mean_growth"]


def test_is_pro_poor_false_under_uniform_growth():
    r = is_pro_poor([1, 2, 3, 4, 5], [1.2, 2.4, 3.6, 4.8, 6.0])
    assert r["pro_poor"] is False
    assert r["premium"] == pytest.approx(0.0)