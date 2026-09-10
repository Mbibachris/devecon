import pytest

from devecon import fgt, fgt_by_group, generalized_entropy, theil_by_group

# ---- FGT decomposition ---------------------------------------------------

INC = [5, 15, 8, 25, 3, 40]
GRP = ["rural", "urban", "rural", "urban", "rural", "urban"]


def test_fgt_decomposition_total_matches_pooled():
    dec = fgt_by_group(INC, poverty_line=10, groups=GRP, alpha=0)
    pooled = fgt(INC, poverty_line=10, alpha=0)
    assert dec.total == pytest.approx(pooled)


def test_fgt_contributions_sum_to_total():
    dec = fgt_by_group(INC, poverty_line=10, groups=GRP, alpha=0)
    assert sum(dec.contributions.values()) == pytest.approx(dec.total)


def test_fgt_population_shares_sum_to_one():
    dec = fgt_by_group(INC, poverty_line=10, groups=GRP, alpha=0)
    assert sum(dec.population_shares.values()) == pytest.approx(1.0)


def test_fgt_group_indices_correct():
    dec = fgt_by_group(INC, poverty_line=10, groups=GRP, alpha=0)
    # rural incomes [5,8,3] all below 10 -> FGT0 = 1.0
    # urban incomes [15,25,40] all above 10 -> FGT0 = 0.0
    assert dec.group_indices["rural"] == pytest.approx(1.0)
    assert dec.group_indices["urban"] == pytest.approx(0.0)


def test_fgt_mismatched_group_length_raises():
    with pytest.raises(ValueError):
        fgt_by_group([1, 2, 3], poverty_line=2, groups=["a", "b"])


# ---- Theil decomposition -------------------------------------------------

VALS = [2, 4, 3, 20, 25, 30]
GRP2 = ["A", "A", "A", "B", "B", "B"]


def test_theil_within_plus_between_equals_total():
    td = theil_by_group(VALS, groups=GRP2)
    assert td.within + td.between == pytest.approx(td.total)


def test_theil_total_matches_pooled_ge1():
    td = theil_by_group(VALS, groups=GRP2)
    assert td.total == pytest.approx(generalized_entropy(VALS, alpha=1.0))


def test_theil_between_dominates_when_groups_differ():
    # Group A is poor, group B is rich: inequality is mostly between-group.
    td = theil_by_group(VALS, groups=GRP2)
    assert td.between > td.within


def test_theil_between_zero_when_group_means_equal():
    # Two groups with identical means: between-group component is zero.
    vals = [1, 3, 1, 3]        # both groups mean 2
    groups = ["A", "A", "B", "B"]
    td = theil_by_group(vals, groups=groups)
    assert td.between == pytest.approx(0.0)


def test_theil_requires_positive():
    with pytest.raises(ValueError):
        theil_by_group([0, 1, 2, 3], groups=["a", "a", "b", "b"])