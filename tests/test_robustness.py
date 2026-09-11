import numpy as np
import pytest

from devecon import alkire_foster, cutoff_profile, rank_robustness

DEP = [
    [1, 1, 1],
    [1, 1, 0],
    [1, 0, 0],
    [0, 0, 0],
]


# ---- cutoff_profile ------------------------------------------------------

def test_profile_matches_alkire_foster_at_each_cutoff():
    prof = cutoff_profile(DEP, cutoffs=[1/3, 2/3, 1.0])
    for i, k in enumerate(prof.cutoffs):
        direct = alkire_foster(DEP, cutoff_k=float(k))
        assert prof.M0[i] == pytest.approx(direct.M0)
        assert prof.H[i] == pytest.approx(direct.H)
        assert prof.A[i] == pytest.approx(direct.A)


def test_profile_m0_non_increasing_in_k():
    # A stricter cutoff can only reduce (or hold) M0.
    prof = cutoff_profile(DEP, cutoffs=[0.2, 0.4, 0.6, 0.8, 1.0])
    assert np.all(np.diff(prof.M0) <= 1e-12)


def test_profile_lengths_match_cutoffs():
    prof = cutoff_profile(DEP, cutoffs=[0.25, 0.5, 0.75])
    assert len(prof.cutoffs) == 3
    assert len(prof.M0) == 3
    assert len(prof.H) == 3
    assert len(prof.A) == 3


def test_profile_empty_cutoffs_raises():
    with pytest.raises(ValueError):
        cutoff_profile(DEP, cutoffs=[])


# ---- rank_robustness -----------------------------------------------------

def test_rank_robust_when_ordering_is_clear():
    groups = {
        "poor": [[1, 1, 1], [1, 1, 0], [1, 1, 1]],
        "rich": [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
    }
    r = rank_robustness(groups, cutoffs=[1/3, 2/3, 1.0])
    assert r["robust"] is True
    assert all(rk == ["poor", "rich"] for rk in r["rankings"].values())


def test_rank_flip_is_detected():
    # 'shallow' leads at a lenient cutoff; 'deep' leads once k rises.
    groups = {
        "shallow": [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]],
        "deep":    [[1, 1, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
    }
    r = rank_robustness(groups, cutoffs=[1/3, 0.5, 1.0])
    assert r["robust"] is False


def test_rank_robustness_reports_m0_per_cutoff():
    groups = {
        "a": [[1, 1], [1, 0]],
        "b": [[0, 0], [1, 0]],
    }
    r = rank_robustness(groups, cutoffs=[0.5, 1.0])
    assert set(r["m0"].keys()) == {0.5, 1.0}
    for k in (0.5, 1.0):
        assert set(r["m0"][k].keys()) == {"a", "b"}


def test_rank_robustness_empty_cutoffs_raises():
    with pytest.raises(ValueError):
        rank_robustness({"a": [[1, 0]]}, cutoffs=[])