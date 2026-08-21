"""Focused tests for the Stage 9 Softmax baseline.

Only the logic that is NEW in this stage is tested here: the causal
training-median imputation and training-fitted standardization that Softmax
needs and the tree models did not. Everything shared with RF/LightGBM (origins,
causal eligibility, the probability adjustment, the ten predictors) is already
covered by tests/test_tune_feature_baselines.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from softmax_feature_baseline import (  # noqa: E402
    BRANCH,
    C_SPACE,
    EVALS_PER_DIMENSION,
    OUT_RPS,
    OUT_TUNING,
    causal_design,
    fit_predict_softmax,
)
from tune_feature_baselines import (  # noqa: E402
    FEATURE_COLUMNS,
    N_QUINTILES,
    ValidationError,
    load_branch,
    load_tuning_origins,
    split_at_origin,
)

ORIGIN = pd.Timestamp("2021-03-05")


@pytest.fixture(scope="module")
def split():
    frame = load_branch(BRANCH)
    return frame, *split_at_origin(frame, ORIGIN)


def test_only_the_two_volume_features_need_filling(split) -> None:
    _, train, predict = split
    _, _, _, diagnostics = causal_design(train, predict, ORIGIN)
    assert diagnostics["features_imputed"] == ["feat_dollar_volume_2m",
                                               "feat_abs_ret_to_volume_3m"]
    assert diagnostics["train_cells_imputed"] > 0


def test_medians_come_from_the_training_rows_not_the_whole_dataset(split) -> None:
    frame, train, predict = split
    x_train, _, _, _ = causal_design(train, predict, ORIGIN)
    assert not np.isnan(x_train).any()
    # A median fitted on the eligible history must differ from a global one.
    training_median = train[FEATURE_COLUMNS].median()
    global_median = frame[FEATURE_COLUMNS].median()
    assert any(not np.isclose(training_median[c], global_median[c])
               for c in FEATURE_COLUMNS)


def test_prediction_rows_are_filled_with_training_medians(split) -> None:
    _, train, predict = split
    medians = train[FEATURE_COLUMNS].median()
    target = "feat_dollar_volume_2m"
    missing = predict[target].isna()
    assert missing.any(), "expected at least one missing prediction value"
    _, _, x_predict, _ = causal_design(train, predict, ORIGIN)
    # Re-derive the same column by hand and compare after scaling.
    manual = predict[FEATURE_COLUMNS].fillna(medians)
    filled_train = train[FEATURE_COLUMNS].fillna(medians)
    expected = ((manual - filled_train.mean()) / filled_train.std(ddof=0)).to_numpy()
    assert np.allclose(x_predict, expected)


def test_scaler_is_fitted_on_training_rows_only(split) -> None:
    _, train, predict = split
    x_train, _, x_predict, _ = causal_design(train, predict, ORIGIN)
    assert np.abs(x_train.mean(axis=0)).max() < 1e-10      # training is centred
    assert np.abs(x_train.std(axis=0) - 1).max() < 1e-10   # and unit-scaled
    # The prediction block is transformed, never re-fitted, so it is not centred.
    assert np.abs(x_predict.mean(axis=0)).max() > 1e-3


def test_preprocessing_cannot_see_anything_after_the_origin(split) -> None:
    frame, train, predict = split
    x_train, _, x_predict, _ = causal_design(train, predict, ORIGIN)
    poisoned = frame.copy()
    poisoned.loc[poisoned["origin_date"] > ORIGIN, FEATURE_COLUMNS] = 999.0
    p_train, p_predict = split_at_origin(poisoned, ORIGIN)
    x_train2, _, x_predict2, _ = causal_design(p_train, p_predict, ORIGIN)
    assert np.array_equal(x_train, x_train2)
    assert np.array_equal(x_predict, x_predict2)


def test_missing_training_median_stops_rather_than_inventing_a_rule(split) -> None:
    _, train, predict = split
    broken = train.copy()
    broken["feat_vol_3m"] = np.nan          # no median can be formed
    with pytest.raises(ValidationError, match="no training median available"):
        causal_design(broken, predict, ORIGIN)


def test_softmax_produces_valid_five_class_probabilities(split) -> None:
    _, train, predict = split
    proba, diagnostics = fit_predict_softmax(0.8, train, predict, ORIGIN)
    assert proba.shape == (len(predict), N_QUINTILES)
    assert np.isfinite(proba).all()
    assert (proba >= 0).all() and (proba <= 1).all()
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert diagnostics["converged"]


def test_released_search_space_and_budget() -> None:
    assert list(C_SPACE) == ["C"]
    assert EVALS_PER_DIMENSION * len(C_SPACE) == 20


def test_generated_artifacts_are_consistent() -> None:
    if not (OUT_TUNING.is_file() and OUT_RPS.is_file()):
        pytest.skip("softmax stage has not been run yet")
    tuning = pd.read_csv(OUT_TUNING).iloc[0]
    evaluation = pd.read_csv(OUT_RPS).iloc[0]
    # C was frozen at the tuning winner before M6 evaluation.
    assert np.isclose(evaluation["frozen_C"], tuning["best_C"])
    assert 0.5 <= tuning["best_C"] <= 1.0
    assert tuning["evaluations_completed"] == tuning["evaluation_budget"] == 20
    assert (tuning["feasible_trials"] + tuning["infeasible_trials"]
            == tuning["evaluations_completed"])
    assert json.loads(tuning["feature_columns"]) == FEATURE_COLUMNS
    assert len(json.loads(tuning["per_origin_rps"])) == 12
    assert tuning["n_tuning_origins"] == 12
    assert list(load_tuning_origins())[0] == pd.Timestamp(tuning["first_tuning_origin"])
    if evaluation["status"] == "complete":
        assert 0 <= evaluation["min_probability"] <= evaluation["max_probability"] <= 1
        assert evaluation["max_row_sum_error"] < 1e-9
