"""Focused tests for the Stage 10 exploratory metrics.

The point of these is to protect the things that would quietly invalidate the
comparison: that RPS still reproduces the saved values, that every model is
scored on the same aligned 1,200 observations, that the AUC is full rather than
partial, and that the ECE implementation is the one that was documented.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_m6_rps import RANK_COLUMNS  # noqa: E402
from exploratory_evaluation import (  # noqa: E402
    ECE_BINS_PRIMARY,
    MODEL_ORDER,
    OUT_METRICS,
    OUT_RMSE,
    N_QUINTILES,
    expected_calibration_error,
    load_ground_truth,
    load_probability_forecasts,
    one_vs_rest_auc,
    verify_rps_reproduces_saved,
)


@pytest.fixture(scope="module")
def data():
    truth = load_ground_truth()
    return truth, load_probability_forecasts(truth)


def test_ground_truth_is_the_official_1200_observation_set(data) -> None:
    truth, _ = data
    assert len(truth) == 1200
    assert sorted(truth["round"].unique()) == list(range(1, 13))
    assert truth.groupby("round").size().unique().tolist() == [100]
    assert np.allclose(truth[RANK_COLUMNS].to_numpy(float).sum(axis=1), 1.0)


def test_every_model_shares_the_same_aligned_observations(data) -> None:
    truth, forecasts = data
    key = list(zip(truth["round"], truth["symbol"]))
    for label, frame in forecasts.items():
        assert len(frame) == 1200, label
        assert list(zip(frame["round"], frame["symbol"])) == key, label
        probabilities = frame[RANK_COLUMNS].to_numpy(float)
        assert np.isfinite(probabilities).all(), label
        assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6), label
    assert list(forecasts) and set(forecasts) == set(MODEL_ORDER)


def test_rank_columns_are_ordered_worst_to_best(data) -> None:
    truth, _ = data
    assert RANK_COLUMNS == [f"Rank{i}" for i in range(1, 6)]
    block = truth[truth["round"] == 1]
    worst = block.loc[block["actual_return"].idxmin()]
    best = block.loc[block["actual_return"].idxmax()]
    assert worst["Rank1"] == 1.0 and worst["true_class"] == 0
    assert best["Rank5"] == 1.0 and best["true_class"] == 4


def test_the_incomplete_lgbm_knn_experiment_is_excluded(data) -> None:
    _, forecasts = data
    # NB: "LightGBM / no-KNN" legitimately contains the substring "KNN"; the
    # excluded specification is the exact label "LightGBM / KNN".
    assert "LightGBM / KNN" not in forecasts
    assert "LightGBM / no-KNN" in forecasts
    status = pd.read_csv(PROJECT_ROOT / "Results/Evaluation/feature_baseline_m6_rps.csv")
    stopped = status[status["status"] != "complete"]
    assert len(stopped) == 1 and stopped.iloc[0]["variant"] == "knn"


def test_rps_reproduces_every_saved_value() -> None:
    if not OUT_METRICS.is_file():
        pytest.skip("exploratory metrics not generated yet")
    checks = verify_rps_reproduces_saved(pd.read_csv(OUT_METRICS))
    assert len(checks) == 9
    assert all("MATCH" in line for line in checks)


def test_auc_is_full_not_partial() -> None:
    """A partial AUC (max_fpr) would give a different, smaller number."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    true_class = rng.integers(0, N_QUINTILES, size=500)
    scores = rng.random((500, N_QUINTILES))
    macro, per_class = one_vs_rest_auc(scores, true_class)
    assert len(per_class) == N_QUINTILES
    assert macro == pytest.approx(float(np.mean(per_class)))
    expected = roc_auc_score((true_class == 0).astype(int), scores[:, 0])
    assert per_class[0] == pytest.approx(expected)          # full curve
    partial = roc_auc_score((true_class == 0).astype(int), scores[:, 0], max_fpr=0.25)
    assert not np.isclose(per_class[0], partial)


def test_ece_matches_the_documented_top_label_definition() -> None:
    # Perfectly calibrated by construction: confidence 1.0 and always correct.
    probabilities = np.eye(N_QUINTILES)[[0, 1, 2, 3, 4]]
    assert expected_calibration_error(probabilities, np.array([0, 1, 2, 3, 4]),
                                      ECE_BINS_PRIMARY) == pytest.approx(0.0)
    # Maximally overconfident: confidence 1.0 and always wrong.
    assert expected_calibration_error(probabilities, np.array([1, 2, 3, 4, 0]),
                                      ECE_BINS_PRIMARY) == pytest.approx(1.0)
    # A uniform forecast over 5 classes has confidence 0.2 throughout.
    uniform = np.full((100, N_QUINTILES), 0.2)
    labels = np.tile([0, 1, 2, 3, 4], 20)
    assert expected_calibration_error(uniform, labels, ECE_BINS_PRIMARY) == pytest.approx(0.0)


def test_tsfm_rmse_table_shape_and_horizons() -> None:
    if not OUT_RMSE.is_file():
        pytest.skip("exploratory RMSE not generated yet")
    rmse = pd.read_csv(OUT_RMSE)
    assert sorted(rmse["horizon"].unique()) == list(range(1, 21))
    assert set(rmse["rmse_type"]) == {"daily", "cumulative"}
    assert (rmse["n_observations"] == 1200).all()
    assert (rmse["rmse"] > 0).all()
    # No feature-based classifier may appear: they do not forecast returns.
    assert not any(m in set(rmse["model"])
                   for m in ("RF / no-KNN", "RF / KNN", "LightGBM / no-KNN",
                             "Softmax / KNN"))
    # Chronos-2's native forecast has daily rows only (no joint-dependence assumption).
    native = rmse[rmse["model"].str.contains("native")]
    assert set(native["rmse_type"]) == {"daily"}
    # Cumulative error must grow with horizon for every series.
    for _, group in rmse[rmse["rmse_type"] == "cumulative"].groupby(
            ["model", "point_forecast"]):
        series = group.sort_values("horizon")["rmse"].to_numpy()
        assert series[-1] > series[0]
