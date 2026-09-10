import numpy as np
import pytest

from devecon import (
    generalized_lorenz,
    gini,
    growth_incidence,
    lorenz,
)

# ---- Lorenz --------------------------------------------------------------

def test_lorenz_equality_on_diagonal():
    lc = lorenz([5, 5, 5, 5])
    # Perfect equality: cumulative income share equals population share.
    assert np.allclose(lc.population_share, lc.value_share)


def test_lorenz_endpoints():
    lc = lorenz([1, 2, 3, 4, 5])
    assert lc.population_share[0] == 0.0
    assert lc.value_share[0] == 0.0
    assert lc.population_share[-1] == pytest.approx(1.0)
    assert lc.value_share[-1] == pytest.approx(1.0)


def test_lorenz_is_below_diagonal_under_inequality():
    lc = lorenz([1, 2, 4, 8])
    # Every interior Lorenz ordinate lies at or below the 45-degree line.
    assert np.all(lc.value_share <= lc.population_share + 1e-12)


def test_lorenz_gini_consistency():
    # Gini = 1 - 2 * (area under the Lorenz curve). Compute the area with
    # the trapezoidal rule by hand to stay independent of numpy naming.
    x = [1, 2, 4, 8, 3, 5, 7]
    lc = lorenz(x)
    p, v = lc.population_share, lc.value_share
    area = np.sum((p[1:] - p[:-1]) * (v[1:] + v[:-1]) / 2.0)
    assert (1 - 2 * area) == pytest.approx(gini(x))


def test_lorenz_negative_raises():
    with pytest.raises(ValueError):
        lorenz([1, -1, 2])


# ---- Generalized Lorenz --------------------------------------------------

def test_generalized_lorenz_endpoint_is_mean():
    glc = generalized_lorenz([2, 4, 6, 8])
    assert glc.value_share[-1] == pytest.approx(5.0)   # mean of the data


def test_generalized_lorenz_starts_at_zero():
    glc = generalized_lorenz([3, 1, 4, 1, 5])
    assert glc.population_share[0] == 0.0
    assert glc.value_share[0] == 0.0


# ---- Growth incidence curve ----------------------------------------------

def test_gic_uniform_growth_is_flat():
    # Every income scaled by 1.1 -> 10% growth at every percentile.
    g = growth_incidence([1, 2, 3, 4, 5], [1.1, 2.2, 3.3, 4.4, 5.5])
    assert np.allclose(g.growth_rate, 0.10)


def test_gic_pro_poor_slopes_down():
    # Adding 1 to everyone: the poor gain more in percentage terms.
    g = growth_incidence([1, 2, 3, 4, 5], [2, 3, 4, 5, 6])
    assert g.growth_rate[9] > g.growth_rate[-10]   # low percentile > high


def test_gic_length_matches_points():
    g = growth_incidence([1, 2, 3, 4, 5], [2, 3, 4, 5, 6], n_points=99)
    assert len(g.percentile) == 99
    assert len(g.growth_rate) == 99