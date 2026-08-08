"""Focused tests for the M6 RPS evaluator (Stage 5).

These cover only the places where a methodological mistake would silently
corrupt the research results: the tie-aware rank-to-quintile rule, the RPS
formula, the four-week return conversion, and the naive benchmark.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_m6_rps import (  # noqa: E402
    N_QUINTILES,
    official_reference_quintiles,
    quintile_probabilities,
    rank_to_quintiles,
    rps_scores,
    sampled_four_week_returns,
)


def test_untied_ranking_is_one_hot_and_ordered():
    values = np.arange(100, dtype=float)          # strictly increasing, no ties
    membership = rank_to_quintiles(values)
    assert membership.shape == (100, N_QUINTILES)
    assert np.allclose(membership.sum(axis=1), 1.0)
    assert set(np.unique(membership)) <= {0.0, 1.0}
    # lowest 20 assets in quintile 1, highest 20 in quintile 5
    assert membership[:20, 0].all()
    assert membership[80:, 4].all()


def test_untied_ranking_matches_the_official_script_port():
    rng = np.random.default_rng(0)
    values = rng.normal(size=100)                 # continuous => no ties
    assert np.allclose(rank_to_quintiles(values), official_reference_quintiles(values))


def test_tie_inside_one_quintile_matches_official_and_stays_one_hot():
    values = np.arange(100, dtype=float)
    values[5] = values[6]                         # tie well inside quintile 1
    ours = rank_to_quintiles(values)
    assert np.allclose(ours, official_reference_quintiles(values))
    assert np.allclose(ours.sum(axis=1), 1.0)


def test_tie_across_a_quintile_boundary_splits_membership():
    values = np.arange(100, dtype=float)
    values[19] = values[20]                       # slots 20 and 21 => Q1 and Q2
    membership = rank_to_quintiles(values)
    for index in (19, 20):
        assert membership[index, 0] == pytest.approx(0.5)
        assert membership[index, 1] == pytest.approx(0.5)
    assert np.allclose(membership.sum(axis=1), 1.0)
    # the official port drops the second quintile, which is why we depart from it
    official = official_reference_quintiles(values)
    assert official[19].sum() == pytest.approx(0.5)


def test_all_tied_gives_uniform_membership():
    membership = rank_to_quintiles(np.zeros(100))
    assert np.allclose(membership, 0.2)


def test_rps_matches_the_official_formula():
    actual = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
    forecast = np.array([0.08, 0.17, 0.29, 0.31, 0.15])
    expected = np.mean((np.cumsum(actual) - np.cumsum(forecast)) ** 2)
    assert rps_scores(actual, forecast) == pytest.approx(expected)


def test_rps_is_zero_for_a_perfect_forecast_and_positive_otherwise():
    actual = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    assert rps_scores(actual, actual) == pytest.approx(0.0)
    assert rps_scores(actual, np.full(5, 0.2)) > 0.0


def test_naive_benchmark_averages_to_016_over_a_balanced_universe():
    scores = [
        rps_scores(np.eye(N_QUINTILES)[q], np.full(N_QUINTILES, 0.2))
        for q in range(N_QUINTILES)
    ]
    assert float(np.mean(scores)) == pytest.approx(0.16)


def test_four_week_return_conversion():
    samples = np.zeros((2, 3, 20))
    samples[0, 0, :] = np.log(1.01) / 20          # +1% over the window
    returns = sampled_four_week_returns(samples)
    assert returns.shape == (2, 3)
    assert returns[0, 0] == pytest.approx(0.01)
    assert returns[1, 2] == pytest.approx(0.0)


def test_quintile_probabilities_sum_to_one_and_rank_sensibly():
    rng = np.random.default_rng(42)
    samples = rng.normal(scale=0.01, size=(100, 100, 20))
    samples[0] += 0.02                            # asset 0 always the strongest
    probabilities = quintile_probabilities(samples)
    assert probabilities.shape == (100, N_QUINTILES)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert probabilities.min() >= 0.0 and probabilities.max() <= 1.0
    assert probabilities[0, 4] == pytest.approx(1.0)


def test_rank_to_quintiles_rejects_nan():
    with pytest.raises(ValueError):
        rank_to_quintiles(np.array([1.0, np.nan, 3.0]))
