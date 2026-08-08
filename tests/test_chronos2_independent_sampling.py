"""Focused tests for Chronos-2 post-processing method 2 (independent sampling).

The point of this method is that the percentile draws carry NO dependence, so
these tests exist mainly to prove that: independence through time for one asset,
independence across assets, independence across scenarios, and no accidental
reuse of a single draw. Determinism at seed 42 and agreement with the copula
method's marginal handling are also covered.

Trick used throughout: passing the quantile LEVELS themselves as the marginal
curve makes `invert_marginal` the identity (with clipping), so the returned
scenarios are exactly the sampled percentiles and can be inspected directly.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import evaluate_chronos2_spatiotemporal_copula as copula  # noqa: E402
import evaluate_chronos2_independent_sampling as independent  # noqa: E402
from evaluate_chronos2_independent_sampling import (  # noqa: E402
    generate_independent_scenarios,
)

LEVELS = np.array([0.01, 0.05] + [round(0.05 * i, 2) for i in range(2, 19)] + [0.95, 0.99])


def percentile_draws(n_assets=8, n_scenarios=4000, horizon=20, seed=42):
    """Scenarios generated with an identity curve == the raw percentile draws."""
    curve = np.tile(LEVELS, (n_assets, horizon, 1))
    return generate_independent_scenarios(curve, LEVELS, n_scenarios, seed)


def test_shape_and_range():
    draws = percentile_draws()
    assert draws.shape == (8, 4000, 20)
    assert draws.min() >= LEVELS[0] and draws.max() <= LEVELS[-1]


def test_draws_are_independent_through_time_for_one_asset():
    draws = percentile_draws()
    asset = draws[3]                                   # (scenarios, days)
    correlations = np.corrcoef(asset.T)
    off_diagonal = correlations[~np.eye(20, dtype=bool)]
    assert np.abs(off_diagonal).max() < 0.06           # no temporal structure


def test_draws_are_independent_across_assets():
    draws = percentile_draws()
    day = draws[:, :, 7]                               # (assets, scenarios)
    correlations = np.corrcoef(day)
    off_diagonal = correlations[~np.eye(day.shape[0], dtype=bool)]
    assert np.abs(off_diagonal).max() < 0.06           # no cross-asset structure


def test_draws_are_independent_across_scenarios():
    draws = percentile_draws()
    series = draws[2, :, 4]                            # one asset/day over scenarios
    lag_one = np.corrcoef(series[:-1], series[1:])[0, 1]
    assert abs(lag_one) < 0.06

    # no scenario is a copy of another
    scenarios = draws[:, :50, :].transpose(1, 0, 2).reshape(50, -1)
    unique = np.unique(scenarios, axis=0)
    assert unique.shape[0] == 50


def test_no_single_percentile_is_reused_across_the_grid():
    draws = percentile_draws(n_assets=5, n_scenarios=200, horizon=20)
    flat = draws.ravel()
    # Every draw is continuous and distinct, except that clipping deliberately
    # collapses the ~1% below 0.01 and the ~1% above 0.99 onto the boundaries.
    at_boundary = (flat == LEVELS[0]) | (flat == LEVELS[-1])
    assert at_boundary.mean() == pytest.approx(0.02, abs=0.005)
    interior = flat[~at_boundary]
    assert np.unique(interior).size == interior.size
    assert np.unique(flat).size > 0.97 * flat.size
    # and not constant along any axis
    assert draws.std(axis=1).min() > 0
    assert draws.std(axis=2).min() > 0
    assert draws.std(axis=0).min() > 0


def test_marginals_are_uniform_over_the_supported_range():
    draws = percentile_draws(n_assets=2, n_scenarios=20000, horizon=3)
    sample = draws[1, :, 2]
    # U(0,1) clipped to [0.01, 0.99]: mean stays 0.5, and the interior deciles
    # should sit close to their nominal positions
    assert sample.mean() == pytest.approx(0.5, abs=0.02)
    for q in (0.25, 0.5, 0.75):
        assert np.quantile(sample, q) == pytest.approx(q, abs=0.02)


def test_clipping_matches_the_copula_methods_rule():
    curve = np.linspace(-0.05, 0.05, LEVELS.size)
    # both methods route percentiles through the same imported function
    assert independent.invert_marginal is copula.invert_marginal
    assert copula.invert_marginal(curve, LEVELS, np.array([0.0, 1.0]))[0] == pytest.approx(curve[0])
    assert copula.invert_marginal(curve, LEVELS, np.array([0.0, 1.0]))[1] == pytest.approx(curve[-1])


def test_seed_42_is_deterministic_and_other_seeds_differ():
    first = percentile_draws(seed=42)
    again = percentile_draws(seed=42)
    assert np.array_equal(first, again)
    assert not np.array_equal(first, percentile_draws(seed=7))


def test_scenarios_map_through_the_model_curve():
    """Non-identity curves: output must be the curve evaluated at the draws."""
    n_assets, horizon = 3, 4
    curve = np.empty((n_assets, horizon, LEVELS.size))
    for a in range(n_assets):
        for d in range(horizon):
            curve[a, d] = np.linspace(-0.02 * (a + 1), 0.02 * (d + 1), LEVELS.size)

    scenarios = generate_independent_scenarios(curve, LEVELS, 500, 42)
    draws = generate_independent_scenarios(np.tile(LEVELS, (n_assets, horizon, 1)),
                                           LEVELS, 500, 42)
    for a in range(n_assets):
        for d in range(horizon):
            expected = np.interp(draws[a, :, d], LEVELS, curve[a, d])
            assert np.allclose(scenarios[a, :, d], expected)


def test_settings_match_the_copula_experiment():
    """The comparison is only meaningful if nothing else differs."""
    assert independent.N_SCENARIOS == copula.N_SCENARIOS == 1000
    assert independent.RANDOM_SEED == copula.RANDOM_SEED == 42
    assert independent.marginal_curves is copula.marginal_curves
    assert independent.load_quantile_forecasts is copula.load_quantile_forecasts
