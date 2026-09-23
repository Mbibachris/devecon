import numpy as np
import pytest

from devecon import af_identify, alkire_foster

# Same worked example as test_alkire_foster.py:
#   scores 1, 2/3, 1/3, 0 with three equally weighted indicators.
DEP = [
    [1, 1, 1],
    [1, 1, 0],
    [1, 0, 0],
    [0, 0, 0],
]


def test_scores_and_poor_flags():
    ident = af_identify(DEP, cutoff_k=1/3)
    assert ident.scores == pytest.approx([1, 2/3, 1/3, 0])
    assert list(ident.poor) == [True, True, True, False]
    assert ident.censored_scores == pytest.approx([1, 2/3, 1/3, 0])


def test_censoring_at_higher_k():
    ident = af_identify(DEP, cutoff_k=0.5)
    assert list(ident.poor) == [True, True, False, False]
    assert ident.censored_scores == pytest.approx([1, 2/3, 0, 0])   # person 2 censored


def test_unit_level_results_reproduce_aggregates():
    # H, A, M0 recomputed from af_identify must equal alkire_foster exactly.
    sample_weights = np.array([1.0, 2.0, 1.0, 4.0])
    ident = af_identify(DEP, cutoff_k=1/3)
    res = alkire_foster(DEP, cutoff_k=1/3, sample_weights=sample_weights)
    H = sample_weights[ident.poor].sum() / sample_weights.sum()
    M0 = (sample_weights * ident.censored_scores).sum() / sample_weights.sum()
    assert res.H == pytest.approx(H)
    assert res.M0 == pytest.approx(M0)


def test_indicator_weights_change_who_is_poor():
    dep = [[1, 0], [0, 1]]
    ident = af_identify(dep, cutoff_k=0.5, indicator_weights=[0.8, 0.2])
    assert list(ident.poor) == [True, False]


def test_invalid_input_is_rejected():
    with pytest.raises(ValueError, match="only 0 and 1"):
        af_identify([[0, 2]], cutoff_k=0.5)
    with pytest.raises(ValueError, match="cutoff_k"):
        af_identify(DEP, cutoff_k=0)