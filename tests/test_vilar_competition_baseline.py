"""Focused tests for the Vilar competition-benchmark reproduction.

The one thing worth protecting here is FIDELITY to the released notebook: if the
clean script ever stops matching `m6paper.ipynb`, the benchmark stops being a
reproduction. One round is checked against the notebook's own code inline; the
rest of the tests cover the specific behaviours that are easy to "improve" by
accident (the uniform fallback, per-asset dropna, the DRE and VXX special cases,
and causality).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

_SPEC = importlib.util.spec_from_file_location(
    "vilar_baseline",
    PROJECT_ROOT / "Competition Benchmark" / "evaluate_vilar_competition_baseline.py")
vilar = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(vilar)

ORIGIN = pd.Timestamp("2022-03-04")


@pytest.fixture(scope="module")
def loaded():
    prices, groups = vilar.load_reference_inputs()
    return prices, groups, vilar.build_quintile_panel(prices)


# --------------------------------------------------------------------------- #
# q_dists: the source's exact semantics
# --------------------------------------------------------------------------- #

def test_q_dists_returns_a_cumulative_distribution() -> None:
    column = pd.Series([1, 1, 2, 3, 5])
    result = vilar.q_dists(column)
    assert np.allclose(result, [0.4, 0.6, 0.8, 0.8, 1.0])
    assert result[-1] == pytest.approx(1.0)
    assert np.all(np.diff(result) >= -1e-15)          # non-decreasing


def test_q_dists_uniform_fallback_when_nothing_survives_dropna() -> None:
    assert np.allclose(vilar.q_dists(pd.Series([np.nan, np.nan])),
                       [0.2, 0.4, 0.6, 0.8, 1.0])
    assert np.allclose(vilar.q_dists(pd.Series([], dtype=float)),
                       [0.2, 0.4, 0.6, 0.8, 1.0])


def test_q_dists_drops_nan_per_asset_not_per_date() -> None:
    """A missing value removes that observation only, never the whole date."""
    with_gaps = pd.Series([1, np.nan, 5, np.nan, 5])
    without = pd.Series([1, 5, 5])
    assert np.allclose(vilar.q_dists(with_gaps), vilar.q_dists(without))


# --------------------------------------------------------------------------- #
# Panel construction and quintile direction
# --------------------------------------------------------------------------- #

def test_panel_is_quintiles_one_to_five_worst_to_best(loaded) -> None:
    prices, _, panel = loaded
    values = panel.to_numpy()
    finite = values[np.isfinite(values)]
    assert set(np.unique(finite)) == {1.0, 2.0, 3.0, 4.0, 5.0}
    # The asset with the largest 20-day log return on a date must be in Q5.
    position = int(panel.index.get_loc(ORIGIN))
    price_position = int(prices.index.get_loc(ORIGIN))
    returns = (np.log(prices.iloc[price_position])
               - np.log(prices.iloc[price_position - vilar.LAG]))
    assert panel.iloc[position][returns.idxmax()] == 5
    assert panel.iloc[position][returns.idxmin()] == 1


def test_groups_isolate_vxx_and_dre(loaded) -> None:
    _, groups, _ = loaded
    assert groups["vxx"] == ["VXX"] and groups["dre"] == ["DRE"]
    assert "VXX" not in groups["etfs"] and "DRE" not in groups["stocks"]
    assert len(groups["etfs"]) == 49 and len(groups["stocks"]) == 49
    assert sum(len(v) for v in groups.values()) == 100


def test_dre_receives_the_uniform_type_component(loaded) -> None:
    """The notebook computes a DRE group vector then multiplies it by zero."""
    _, groups, panel = loaded
    end = int(panel.index.get_loc(ORIGIN)) + 1
    type_vectors = vilar.type_distributions(panel, groups, end)
    assert not np.allclose(type_vectors["dre"], vilar.UNIFORM_CUMULATIVE)  # it exists
    forecast = vilar.mixed_forecast(panel, groups, end)
    temporal = vilar.temporal_distribution(panel, end)[
        :, list(panel.columns).index("DRE")]
    expected = np.diff(0.5 * vilar.UNIFORM_CUMULATIVE + 0.5 * temporal, prepend=0.0)
    assert np.allclose(forecast["DRE"].to_numpy(), expected)   # uniform, not res[3]


# --------------------------------------------------------------------------- #
# Fidelity to the released notebook
# --------------------------------------------------------------------------- #

def test_matches_the_released_notebook_code_exactly(loaded) -> None:
    """Run the notebook's own cell-9 expression and compare, for one round."""
    _, groups, panel = loaded
    universe = pd.read_csv(vilar.UNIVERSE_CSV)
    universe.columns = [c.strip().lstrip("﻿") for c in universe.columns]
    etfs = list(universe[universe["class"] == "ETF"]["symbol"])
    stocks = list(universe[universe["class"] == "Stock"]["symbol"])
    etfs.remove("VXX"); vari = ["VXX"]; stocks.remove("DRE"); vari2 = ["DRE"]
    qf, lag, av = panel, vilar.LAG, 0.5
    end = int(panel.index.get_loc(ORIGIN)) + 1
    j = len(qf) - lag - end

    res = [qf[-lag - j - 50 * lag:len(qf) - lag - j][i]
           .apply(vilar.q_dists, axis=0).mean(axis=1)
           for i in (etfs, stocks, vari, vari2)]
    qfg = ((1 - av) * qf[-lag - j - 1:len(qf) - lag - j].apply(
                lambda x: res[0] if x.name in etfs
                else res[1] if x.name in stocks
                else res[2] if x.name in vari
                else 1 * np.array([0.2, 0.4, 0.6, 0.8, 1]) + 0 * res[3], axis=0).values
           + av * 0.2 * qf[-lag - j - 5 * lag:len(qf) - lag - j].apply(vilar.q_dists, axis=0).values
           + av * 0.2 * qf[-lag - j - 10 * lag:len(qf) - lag - j].apply(vilar.q_dists, axis=0).values
           + av * 0.6 * qf[-lag - j - 400 * lag:len(qf) - lag - j].apply(vilar.q_dists, axis=0).values)

    mine = vilar.mixed_forecast(panel, groups, end).to_numpy()
    assert np.abs(qfg - np.cumsum(mine, axis=0)).max() < 1e-12
    assert np.abs(np.diff(qfg, axis=0, prepend=0.0) - mine).max() < 1e-12


def test_fixed_parameters_are_the_released_ones() -> None:
    assert vilar.LAG == 20
    assert vilar.TYPE_PERIODS == 50
    assert vilar.TEMPORAL_WEIGHTS == ((5, 0.2), (10, 0.2), (400, 0.6))
    assert sum(w for _, w in vilar.TEMPORAL_WEIGHTS) == pytest.approx(1.0)
    assert vilar.MIXED_TEMPORAL_WEIGHT == 0.5


# --------------------------------------------------------------------------- #
# Causality
# --------------------------------------------------------------------------- #

def test_forecast_cannot_see_anything_after_the_origin(loaded) -> None:
    prices, groups, panel = loaded
    end = int(panel.index.get_loc(ORIGIN)) + 1
    price_position = int(prices.index.get_loc(ORIGIN))
    base = vilar.mixed_forecast(panel, groups, end).to_numpy()

    poisoned = prices.copy()
    poisoned.iloc[price_position + 1:] = 1.0
    after = vilar.mixed_forecast(vilar.build_quintile_panel(poisoned), groups, end)
    assert np.array_equal(base, after.to_numpy())

    # Negative control: one asset perturbed BEFORE the origin must move the
    # cross-section (a uniform rescaling of all assets would not).
    control = prices.copy()
    control.iloc[price_position - 40:price_position + 1,
                 control.columns.get_loc("ABBV")] *= 1.5
    changed = vilar.mixed_forecast(vilar.build_quintile_panel(control), groups, end)
    assert not np.array_equal(base, changed.to_numpy())


def test_widest_window_never_reaches_past_the_origin(loaded) -> None:
    _, _, panel = loaded
    end = int(panel.index.get_loc(ORIGIN)) + 1
    for periods, _ in vilar.TEMPORAL_WEIGHTS:
        window = vilar._window(panel, end, periods)
        assert window.index[-1] == ORIGIN
        assert (window.index <= ORIGIN).all()


# --------------------------------------------------------------------------- #
# Generated artifacts
# --------------------------------------------------------------------------- #

def test_saved_predictions_are_valid_and_complete() -> None:
    if not vilar.OUT_PREDICTIONS.is_file():
        pytest.skip("Vilar benchmark has not been run yet")
    predictions = pd.read_csv(vilar.OUT_PREDICTIONS)
    truth = pd.read_csv(PROJECT_ROOT / "Results/Evaluation/m6_ground_truth_quintiles.csv")
    assert len(predictions) == 1200
    assert sorted(predictions["round"].unique()) == list(range(1, 13))
    assert predictions.groupby("round").size().unique().tolist() == [100]
    probabilities = predictions[vilar.RANK_COLUMNS].to_numpy(float)
    assert np.isfinite(probabilities).all()
    assert probabilities.min() >= 0
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    for round_number in range(1, 13):
        assert (set(predictions[predictions["round"] == round_number]["symbol"])
                == set(truth[truth["round"] == round_number]["symbol"]))
    # The project's canonical ticker, not Vilar's EG.
    assert "RE" in set(predictions["symbol"]) and "EG" not in set(predictions["symbol"])


def test_saved_rps_is_consistent_and_close_to_the_published_value() -> None:
    if not vilar.OUT_RPS.is_file():
        pytest.skip("Vilar benchmark has not been run yet")
    record = pd.read_csv(vilar.OUT_RPS).iloc[0]
    rounds = [float(record[f"round_{i:02d}_rps"]) for i in range(1, 13)]
    assert record["mean_m6_rps"] == pytest.approx(float(np.mean(rounds)), abs=1e-8)
    assert record["rounds_beating_naive"] == sum(1 for r in rounds if r < 0.16)
    assert record["all_target_dates_agree_with_official"]
    # A reproduction of the fixed method should land near the published 0.15729,
    # without ever having been tuned toward it.
    assert abs(record["mean_m6_rps"] - vilar.PUBLISHED_VILAR_RPS) < 0.002
