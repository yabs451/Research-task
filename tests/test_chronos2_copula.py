"""Focused tests for the Chronos-2 spatiotemporal copula post-processing.

Covers only the new statistical machinery, where a silent error would corrupt
the research result: the marginal inverse-CDF, the AR(1) time matrix, the
lag-one estimator, the separable copula's dependence structure, and the fact
that leading missing history is excluded rather than imputed. The M6 quintile
and RPS logic is already covered by tests/test_evaluate_m6_rps.py.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_chronos2_spatiotemporal_copula import (  # noqa: E402
    ar1_time_correlation, asset_correlation, gaussian_scores, generate_scenarios,
    invert_marginal, marginal_curves, temporal_rho,
)

LEVELS = np.array([0.01, 0.05] + [round(0.05 * i, 2) for i in range(2, 19)] + [0.95, 0.99])


def test_invert_marginal_reproduces_the_model_quantiles():
    curve = np.linspace(-0.05, 0.05, LEVELS.size)
    assert np.allclose(invert_marginal(curve, LEVELS, LEVELS), curve)


def test_invert_marginal_is_monotonic_and_clipped_at_the_tails():
    curve = np.linspace(-0.05, 0.05, LEVELS.size)
    u = np.linspace(0.0, 1.0, 501)
    values = invert_marginal(curve, LEVELS, u)
    assert np.all(np.diff(values) >= 0)
    # nothing beyond the model's own 1st/99th percentile is invented
    assert values.min() == pytest.approx(curve[0])
    assert values.max() == pytest.approx(curve[-1])
    assert invert_marginal(curve, LEVELS, np.array([0.0])) == pytest.approx(curve[0])
    assert invert_marginal(curve, LEVELS, np.array([1.0])) == pytest.approx(curve[-1])


def test_marginal_curves_repair_crossing_without_touching_valid_curves():
    valid = np.tile(np.linspace(-0.05, 0.05, LEVELS.size), (2, 3, 1))
    assert np.array_equal(marginal_curves(valid), valid)

    crossed = valid.copy()
    crossed[0, 0, 5] = crossed[0, 0, 4] - 0.01        # a crossing
    repaired = marginal_curves(crossed)
    assert np.all(np.diff(repaired, axis=2) >= 0)


def test_ar1_time_correlation_structure():
    matrix = ar1_time_correlation(0.5, horizon=4)
    assert matrix.shape == (4, 4)
    assert np.allclose(np.diag(matrix), 1.0)
    assert matrix[0, 1] == pytest.approx(0.5)
    assert matrix[0, 3] == pytest.approx(0.125)
    assert np.allclose(matrix, matrix.T)
    # rho = 0 must give the identity, i.e. no temporal dependence imposed
    assert np.allclose(ar1_time_correlation(0.0, horizon=4), np.eye(4))


def test_temporal_rho_recovers_a_known_ar1_and_stays_near_zero_for_noise():
    rng = np.random.default_rng(0)
    n, rho = 4000, 0.6
    series = np.zeros((n, 3))
    for t in range(1, n):
        series[t] = rho * series[t - 1] + rng.normal(scale=np.sqrt(1 - rho ** 2), size=3)
    assert temporal_rho(series) == pytest.approx(rho, abs=0.05)

    white = rng.normal(size=(4000, 3))
    assert abs(temporal_rho(white)) < 0.05


def test_gaussian_scores_drop_leading_gaps_instead_of_imputing():
    rng = np.random.default_rng(1)
    context = rng.normal(size=(300, 4))
    context[:50, 2] = np.nan                      # OGN-style leading history gap
    scores, mask = gaussian_scores(context)

    assert scores.shape == (250, 4)               # only complete rows are usable
    assert int(mask.sum()) == 250
    assert not mask[:50].any() and mask[50:].all()
    assert np.isfinite(scores).all()              # no imputed values leaked in
    assert abs(scores.mean()) < 0.1               # standard-normal scores


def test_asset_correlation_is_shrunk_valid_and_positive_definite():
    rng = np.random.default_rng(2)
    # fewer observations than a full-rank sample estimate would need
    scores = rng.normal(size=(60, 40))
    correlation, shrinkage, _, min_after = asset_correlation(scores)

    assert correlation.shape == (40, 40)
    assert np.allclose(np.diag(correlation), 1.0)
    assert np.allclose(correlation, correlation.T)
    assert 0.0 < shrinkage <= 1.0
    assert min_after > 0
    np.linalg.cholesky(correlation)               # must not raise


def test_scenarios_reproduce_the_target_dependence_and_are_deterministic():
    rng = np.random.default_rng(3)
    n_assets, horizon, n_quantiles = 100, 20, LEVELS.size

    # identity marginals: the curve maps a percentile back to its normal score,
    # so the generated scenarios are the latent Gaussians themselves.
    curve = np.tile(norm.ppf(LEVELS), (n_assets, horizon, 1))

    # positive-definite correlation from a factor model: F F' + diagonal noise
    loadings = rng.normal(size=(n_assets, 5))
    covariance = loadings @ loadings.T + np.diag(rng.uniform(0.5, 1.5, n_assets))
    sd = np.sqrt(np.diag(covariance))
    asset_corr = covariance / np.outer(sd, sd)
    asset_corr = (asset_corr + asset_corr.T) / 2
    np.fill_diagonal(asset_corr, 1.0)
    rho = 0.4
    time_corr = ar1_time_correlation(rho, horizon)

    scenarios = generate_scenarios(curve, LEVELS, asset_corr, time_corr, 4000, 42)
    assert scenarios.shape == (n_assets, 4000, horizon)

    # cross-asset dependence on a single day
    day0 = scenarios[:, :, 0]
    empirical_assets = np.corrcoef(day0)
    off = ~np.eye(n_assets, dtype=bool)
    assert np.abs(empirical_assets[off] - asset_corr[off]).mean() < 0.05

    # temporal dependence within one asset
    asset0 = scenarios[0]
    empirical_time = np.corrcoef(asset0.T)
    assert empirical_time[0, 1] == pytest.approx(rho, abs=0.05)
    assert empirical_time[0, 3] == pytest.approx(rho ** 3, abs=0.05)

    # same seed -> identical scenarios
    again = generate_scenarios(curve, LEVELS, asset_corr, time_corr, 4000, 42)
    assert np.array_equal(scenarios, again)
    different = generate_scenarios(curve, LEVELS, asset_corr, time_corr, 4000, 7)
    assert not np.array_equal(scenarios, different)


def test_scenario_margins_stay_uniform_per_asset_and_day():
    n_assets, horizon = 5, 20
    curve = np.tile(norm.ppf(LEVELS), (n_assets, horizon, 1))
    asset_corr = np.eye(n_assets)
    scenarios = generate_scenarios(curve, LEVELS, asset_corr,
                                   ar1_time_correlation(0.0, horizon), 5000, 42)
    # each (asset, day) marginal should look standard normal (clipped at +-2.33)
    sample = scenarios[2, :, 7]
    assert abs(sample.mean()) < 0.05
    assert sample.std() == pytest.approx(1.0, abs=0.05)
