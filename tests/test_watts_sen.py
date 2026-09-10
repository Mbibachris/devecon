import math

import pytest

from devecon import sen, watts

# ---- Watts ---------------------------------------------------------------

def test_watts_no_poverty():
    assert watts([10, 20, 30], poverty_line=5) == 0.0


def test_watts_known_value():
    # One poor person at 5, line 10: ln(10/5)/3 = ln(2)/3.
    assert watts([5, 15, 25], poverty_line=10) == pytest.approx(math.log(2) / 3)


def test_watts_poor_zero_income_raises():
    with pytest.raises(ValueError):
        watts([0, 15, 25], poverty_line=10)


def test_watts_bad_line_raises():
    with pytest.raises(ValueError):
        watts([5, 15], poverty_line=0)


# ---- Sen -----------------------------------------------------------------

def test_sen_no_poverty():
    assert sen([10, 20, 30], poverty_line=5) == 0.0


def test_sen_equal_poor_reduces_to_H_times_I():
    # Two equally-poor people: Gini among poor = 0, so S = H * I.
    # [5,5], z=10: H=1, I=0.5 -> 0.5.
    assert sen([5, 5], poverty_line=10) == pytest.approx(0.5)


def test_sen_single_poor_person():
    # One poor person: Gini among poor = 0. H=1/3, I=(10-4)/10=0.6 -> 0.2.
    assert sen([4, 20, 30], poverty_line=10) == pytest.approx(0.2)


def test_sen_in_unit_interval():
    s = sen([2, 4, 6, 8, 12], poverty_line=10)
    assert 0.0 <= s <= 1.0