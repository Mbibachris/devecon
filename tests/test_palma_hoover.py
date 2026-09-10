import pytest

from devecon import hoover, palma

# ---- Palma ---------------------------------------------------------------

def test_palma_known_value():
    # Incomes 1..10: bottom 40% = 1+2+3+4 = 10; top 10% = 10 -> ratio 1.0.
    assert palma(list(range(1, 11))) == pytest.approx(1.0)


def test_palma_higher_under_more_inequality():
    equalish = palma([4, 5, 5, 5, 5, 5, 5, 5, 5, 6])
    skewed = palma([1, 1, 1, 1, 1, 1, 1, 1, 1, 50])
    assert skewed > equalish


def test_palma_negative_raises():
    with pytest.raises(ValueError):
        palma([1, -2, 3, 4, 5, 6, 7, 8, 9, 10])


# ---- Hoover --------------------------------------------------------------

def test_hoover_perfect_equality():
    assert hoover([5, 5, 5, 5]) == 0.0


def test_hoover_known_value():
    # [0,0,0,100]: mean 25; |x/mean-1| = 1,1,1,3; 0.5*6/4 = 0.75.
    assert hoover([0, 0, 0, 100]) == pytest.approx(0.75)


def test_hoover_in_unit_interval():
    h = hoover([1, 2, 3, 4, 5, 100])
    assert 0.0 <= h < 1.0


def test_hoover_negative_raises():
    with pytest.raises(ValueError):
        hoover([1, -1, 2])