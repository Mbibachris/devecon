import numpy as np
import pytest

from devecon import first_order_dominance, second_order_dominance

# ---- First-order dominance -----------------------------------------------

def test_fod_richer_distribution_dominates():
    # Every income higher -> less poverty at every line -> 1 dominates 2.
    r = first_order_dominance([10, 20, 30, 40], [5, 15, 25, 35])
    assert r.dominates_1_over_2 is True
    assert r.dominates_2_over_1 is False
    assert r.crosses is False


def test_fod_is_antisymmetric():
    # Swapping the arguments flips which side dominates.
    r = first_order_dominance([5, 15, 25, 35], [10, 20, 30, 40])
    assert r.dominates_1_over_2 is False
    assert r.dominates_2_over_1 is True
    assert r.crosses is False


def test_fod_identical_distributions_mutually_dominate():
    # Equal CDFs: each weakly dominates the other; not a crossing.
    r = first_order_dominance([1, 2, 3], [1, 2, 3])
    assert r.dominates_1_over_2 is True
    assert r.dominates_2_over_1 is True
    assert r.crosses is False


def test_fod_crossing_cdfs_have_no_ranking():
    r = first_order_dominance([1, 50, 51, 52], [10, 11, 12, 100])
    assert r.crosses is True
    assert r.dominates_1_over_2 is False
    assert r.dominates_2_over_1 is False


# ---- Second-order dominance ----------------------------------------------

def test_sod_ranks_a_mean_preserving_spread():
    # Same mean (20), but distribution 1 is tighter. First-order cannot
    # rank them (CDFs cross); second-order says the tighter one dominates.
    tight = [18, 19, 21, 22]
    spread = [5, 10, 30, 35]

    fod = first_order_dominance(tight, spread)
    assert fod.crosses is True                      # FOD can't decide

    sod = second_order_dominance(tight, spread)
    assert sod.dominates_1_over_2 is True           # SOD can
    assert sod.dominates_2_over_1 is False
    assert sod.crosses is False


def test_sod_implied_by_fod():
    # First-order dominance implies second-order dominance.
    r_fod = first_order_dominance([10, 20, 30, 40], [5, 15, 25, 35])
    r_sod = second_order_dominance([10, 20, 30, 40], [5, 15, 25, 35])
    assert r_fod.dominates_1_over_2 is True
    assert r_sod.dominates_1_over_2 is True


def test_dominance_result_exposes_curves():
    r = first_order_dominance([1, 2, 3], [2, 3, 4])
    # CDFs are monotone non-decreasing and end at 1.
    assert r.curve_1[-1] == pytest.approx(1.0)
    assert r.curve_2[-1] == pytest.approx(1.0)
    assert np.all(np.diff(r.curve_1) >= -1e-12)