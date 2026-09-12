import pytest

from devecon import (
    af_decompose,
    alkire_foster,
    describe,
    first_order_dominance,
    gini,
    summary,
)

DEP = [
    [1, 1, 1],
    [1, 1, 0],
    [1, 0, 0],
    [0, 0, 0],
]


# ---- describe ------------------------------------------------------------

def test_describe_basic_stats():
    d = describe([1, 2, 3, 4, 5])
    assert d["n"] == 5
    assert d["mean"] == pytest.approx(3.0)
    assert d["median"] == pytest.approx(3.0)
    assert d["min"] == 1.0
    assert d["max"] == 5.0


def test_describe_weighted_mean_shifts():
    # Heavier weight on the high value pulls the mean up.
    d = describe([10, 20], weights=[1, 3])
    assert d["mean"] == pytest.approx((10 + 60) / 4)


def test_describe_requires_1d():
    with pytest.raises(ValueError):
        describe([[1, 2], [3, 4]])


# ---- summary dispatch ----------------------------------------------------

def test_summary_of_af_result_has_all_three_measures():
    text = summary(alkire_foster(DEP, cutoff_k=1/3))
    assert "Incidence" in text
    assert "Intensity" in text
    assert "0.5000" in text          # M0


def test_summary_of_decomposition_lists_contributions():
    dec = af_decompose(DEP, cutoff_k=1/3,
                       indicator_names=["school", "water", "health"])
    text = summary(dec)
    assert "school" in text
    assert "%" in text
    # Largest contributor should be listed first.
    assert text.index("school") < text.index("health")


def test_summary_of_plain_float():
    text = summary(gini([1, 2, 3, 4, 5]))
    assert "0.2667" in text


def test_summary_of_dominance_verdicts():
    dom = summary(first_order_dominance([10, 20, 30], [5, 15, 25]))
    assert "dominates" in dom.lower()
    cross = summary(first_order_dominance([1, 50, 51], [10, 11, 100]))
    assert "cross" in cross.lower()


def test_summary_title_is_included():
    text = summary(gini([1, 2, 3]), title="My Gini")
    assert text.startswith("My Gini")


# ---- the critical guarantee: existing returns are untouched --------------

def test_existing_returns_unchanged():
    # The summary layer must not have altered any measure's return type.
    assert isinstance(gini([1, 2, 3, 4, 5]), float)
    res = alkire_foster(DEP, cutoff_k=1/3)
    assert res.H == 0.75 and res.M0 == pytest.approx(0.5)