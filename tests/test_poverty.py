import pytest
from devecon.poverty import fgt


def test_no_poverty():
    assert fgt([10, 20, 30], poverty_line=5) == 0.0


def test_headcount():
    # 10 and 20 fall below z=25; 30 and 40 don't -> 2 of 4 poor
    assert fgt([10, 20, 30, 40], poverty_line=25, alpha=0) == 0.5


def test_poverty_gap():
    # gaps: 0.6 and 0.2 -> sum 0.8 over 4 -> 0.2
    assert fgt([10, 20, 30, 40], poverty_line=25, alpha=1) == pytest.approx(0.2)


def test_squared_gap():
    # 0.6^2 + 0.2^2 = 0.4 over 4 -> 0.1
    assert fgt([10, 20, 30, 40], poverty_line=25, alpha=2) == pytest.approx(0.1)


def test_weights_shift_result():
    base = fgt([10, 40], poverty_line=25, alpha=0)                     # 0.5
    weighted = fgt([10, 40], poverty_line=25, alpha=0, weights=[2, 1]) # 2/3
    assert weighted > base


def test_bad_line_raises():
    with pytest.raises(ValueError):
        fgt([10, 20], poverty_line=0)