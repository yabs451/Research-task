"""Focused tests for the Stage 7 RF / LightGBM tuning rules.

These check the things that would silently corrupt the experiment: the tuning
origins, causal training eligibility, the feature set, the quintile-to-Rank
ordering, probability validity and the infeasible-candidate safeguard.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from tune_feature_baselines import (  # noqa: E402
    FEATURE_COLUMNS,
    INFEASIBLE_LOSS,
    N_QUINTILES,
    N_TUNING_ORIGINS,
    OUT_RESULTS,
    TARGET_COLUMN,
    TRUTH_COLUMNS,
    ValidationError,
    adjust_probabilities,
    evaluate_candidate,
    fit_predict,
    load_branch,
    load_tuning_origins,
    probability_report,
    split_at_origin,
)

RF_SMALL = {"criterion": "entropy", "max_features": 0.5, "n_estimators": 40,
            "max_depth": 4, "min_samples_split": 6, "min_samples_leaf": 9}


# --------------------------------------------------------------------------- #
# Tuning origins
# --------------------------------------------------------------------------- #

def test_exactly_the_twelve_pre_m6_tuning_origins_are_used() -> None:
    origins = load_tuning_origins()
    assert len(origins) == N_TUNING_ORIGINS == 12
    assert origins[0] == pd.Timestamp("2021-03-05")
    assert origins[-1] == pd.Timestamp("2022-01-07")
    assert all(o.dayofweek == 4 for o in origins)
    assert {int(d.days) for d in np.diff(origins)} == {28}

    schedule = pd.read_csv(
        PROJECT_ROOT / "Data" / "metadata" / "feature_baseline_origin_schedule.csv",
        parse_dates=["origin_date"])
    m6 = set(schedule.loc[schedule["origin_role"] == "m6", "origin_date"])
    # No M6 evaluation origin may be touched by tuning.
    assert not (set(origins) & m6)
    assert max(origins) < min(m6)


# --------------------------------------------------------------------------- #
# Causal expanding training history
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("variant", ["no_knn", "knn"])
def test_training_history_is_causal_and_expands(variant: str) -> None:
    frame = load_branch(variant)
    origins = load_tuning_origins()
    sizes = []
    for origin in origins:
        train, predict = split_at_origin(frame, origin)
        # Nothing whose four-week outcome was still unknown at the origin.
        assert (train["target_end_date"] <= origin).all()
        assert (train["origin_date"] < origin).all()
        # The prediction cross-section is exactly that origin.
        assert (predict["origin_date"] == origin).all()
        sizes.append(len(train))
    assert sizes == sorted(sizes)          # expanding, never shrinking
    assert sizes[0] < sizes[-1]            # and it really does grow


@pytest.mark.parametrize("variant", ["no_knn", "knn"])
def test_branch_uses_its_own_targets_and_the_ten_features(variant: str) -> None:
    frame = load_branch(variant)
    assert FEATURE_COLUMNS == [
        "feat_ret_4w_recent", "feat_ret_4w_seasonal_11m", "feat_vol_3m",
        "feat_max_ret_3m", "feat_dollar_volume_2m", "feat_abs_ret_to_volume_3m",
        "feat_rank1_freq_4w", "feat_rank2_freq_4w", "feat_rank4_freq_4w",
        "feat_rank5_freq_4w"]
    assert len(FEATURE_COLUMNS) == 10
    # AssetClass and every other metadata column stay out of the predictors.
    assert not any(c in FEATURE_COLUMNS
                   for c in ("asset_class", "asset_type", "sector_or_etf_type"))
    assert frame[TARGET_COLUMN].between(0, 4).all()
    memberships = frame[TRUTH_COLUMNS].to_numpy()
    assert np.allclose(memberships.sum(axis=1), 1.0)
    # The label really is this branch's own file.
    assert (frame["variant"] == variant).all()


# --------------------------------------------------------------------------- #
# Quintile / Rank ordering
# --------------------------------------------------------------------------- #

def test_quintile_zero_is_the_worst_return_and_maps_to_rank1() -> None:
    frame = load_branch("no_knn")
    block = frame[frame["origin_date"] == load_tuning_origins()[0]]
    worst = block.loc[block["hist_target_log_return"].idxmin()]
    best = block.loc[block["hist_target_log_return"].idxmax()]
    assert worst[TARGET_COLUMN] == 0 and worst["hist_target_Rank1"] > 0
    assert best[TARGET_COLUMN] == 4 and best["hist_target_Rank5"] > 0
    assert TRUTH_COLUMNS == [f"hist_target_Rank{i}" for i in range(1, 6)]


def test_probability_columns_follow_the_classifier_classes() -> None:
    """Column j of the returned matrix must be quintile j, i.e. Rank(j+1)."""
    frame = load_branch("no_knn")
    origin = load_tuning_origins()[0]
    train, predict = split_at_origin(frame, origin)
    # Drop one class from training; the mapping must still put each class in its
    # own column and leave the absent class at zero.
    reduced = train[train[TARGET_COLUMN] != 3]
    proba = fit_predict("rf", RF_SMALL, reduced, predict)
    assert proba.shape == (len(predict), N_QUINTILES)
    assert np.allclose(proba[:, 3], 0.0)
    assert np.allclose(proba.sum(axis=1), 1.0)


# --------------------------------------------------------------------------- #
# The released cross-sectional adjustment
# --------------------------------------------------------------------------- #

def test_adjustment_sets_column_means_to_020_and_preserves_row_sums() -> None:
    rng = np.random.default_rng(0)
    raw = rng.dirichlet(np.ones(5), size=100)
    adjusted = adjust_probabilities(raw)
    assert np.allclose(adjusted.mean(axis=0), 0.2)
    assert np.allclose(adjusted.sum(axis=1), 1.0)
    # It is a pure per-column shift: differences between assets are untouched.
    assert np.allclose(adjusted - raw, (adjusted - raw)[0])


def test_probability_report_flags_both_bounds_and_row_sums() -> None:
    good = np.full((10, 5), 0.2)
    assert probability_report(good)["valid"]

    below = np.full((10, 5), 0.2)
    below[0] = [-0.01, 0.21, 0.2, 0.3, 0.3]
    report = probability_report(below)
    assert not report["valid"] and "below 0" in report["reason"]

    above = np.full((10, 5), 0.2)
    above[0] = [1.05, -0.05, 0.0, 0.0, 0.0]
    assert not probability_report(above)["valid"]

    unnormalised = np.full((10, 5), 0.3)
    report = probability_report(unnormalised)
    assert not report["valid"] and "row sum" in report["reason"]

    nonfinite = np.full((10, 5), 0.2)
    nonfinite[0, 0] = np.nan
    assert not probability_report(nonfinite)["valid"]


def test_infeasible_loss_can_never_beat_a_real_rps() -> None:
    """Mean RPS over five quintiles is bounded well below the rejection loss."""
    assert INFEASIBLE_LOSS == 1.0
    frame = load_branch("no_knn")
    origins = load_tuning_origins()[:2]
    result = evaluate_candidate("rf", RF_SMALL, frame, origins)
    assert result["feasible"]
    assert result["mean_rps"] < INFEASIBLE_LOSS


# --------------------------------------------------------------------------- #
# Missing values and reproducibility
# --------------------------------------------------------------------------- #

LGBM_SMALL = {"num_leaves": 32, "learning_rate": 0.05, "min_data_in_leaf": 25,
              "subsample": 0.7, "feature_fraction": 0.8, "max_depth": 6}


def test_random_forest_consumes_the_intentional_missing_values_natively() -> None:
    frame = load_branch("no_knn")
    train, predict = split_at_origin(frame, load_tuning_origins()[0])
    assert train[FEATURE_COLUMNS].isna().to_numpy().any()   # the NaNs are real
    assert predict[FEATURE_COLUMNS].isna().to_numpy().any()
    proba = fit_predict("rf", RF_SMALL, train, predict)
    assert np.isfinite(proba).all()
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_lightgbm_consumes_the_intentional_missing_values_natively() -> None:
    """Run in a fresh interpreter.

    LightGBM's native Dataset builder intermittently faults when it is invoked
    late in a long pytest session that has already exercised several other
    numeric extension modules (observed as an access violation inside
    LGBM_DatasetSetField; it does not reproduce outside pytest, is unrelated to
    memory or thread pressure, and never occurred during the tuning run, which
    executes LightGBM in a dedicated process). Isolating this one call keeps the
    assertion meaningful without depending on accumulated in-process state.
    """
    import json
    import subprocess

    code = (
        "import sys, json, warnings; warnings.filterwarnings('ignore');"
        f"sys.path.insert(0, {str(PROJECT_ROOT / 'scripts')!r});"
        "import numpy as np, tune_feature_baselines as T;"
        "f = T.load_branch('no_knn');"
        "tr, pr = T.split_at_origin(f, T.load_tuning_origins()[0]);"
        f"p = T.fit_predict('lgbm', {LGBM_SMALL!r}, tr, pr);"
        "print(json.dumps({'nan_in_train': bool(tr[T.FEATURE_COLUMNS].isna()"
        ".to_numpy().any()), 'finite': bool(np.isfinite(p).all()),"
        " 'rows_sum_to_one': bool(np.allclose(p.sum(axis=1), 1.0)),"
        " 'shape': list(p.shape)}))"
    )
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True,
                               text=True, cwd=str(PROJECT_ROOT))
    assert completed.returncode == 0, completed.stderr[-2000:]
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["nan_in_train"]          # the NaNs really are present
    assert result["finite"]
    assert result["rows_sum_to_one"]
    assert result["shape"][1] == N_QUINTILES


def test_fixed_seed_reproduces_the_same_objective() -> None:
    """Seeded runs repeat to floating-point precision.

    The fitted forest itself is fully deterministic - with a fixed
    ``random_state`` every tree is structurally identical run to run. Only the
    accumulation of per-tree probabilities across worker threads (``n_jobs=-1``)
    varies in order, which perturbs the aggregate by about 1e-16. Setting
    ``n_jobs=1`` would be bit-exact at a large cost in tuning time.
    """
    frame = load_branch("no_knn")
    origins = load_tuning_origins()[:2]
    first = evaluate_candidate("rf", RF_SMALL, frame, origins)
    second = evaluate_candidate("rf", RF_SMALL, frame, origins)
    assert first["mean_rps"] == pytest.approx(second["mean_rps"], rel=1e-12)
    assert first["per_origin_rps"] == pytest.approx(second["per_origin_rps"],
                                                    rel=1e-12)


def test_fitted_forest_is_deterministic_under_a_fixed_seed() -> None:
    from sklearn.ensemble import RandomForestClassifier
    frame = load_branch("no_knn")
    train, _ = split_at_origin(frame, load_tuning_origins()[0])
    x, y = train[FEATURE_COLUMNS], train[TARGET_COLUMN]
    forests = [RandomForestClassifier(random_state=42, n_jobs=-1, **RF_SMALL).fit(x, y)
               for _ in range(2)]
    for left, right in zip(*[f.estimators_ for f in forests]):
        assert np.array_equal(left.tree_.feature, right.tree_.feature)
        assert np.array_equal(left.tree_.threshold, right.tree_.threshold)


# --------------------------------------------------------------------------- #
# Generated artifact
# --------------------------------------------------------------------------- #

def test_tuning_results_file_is_complete_and_valid() -> None:
    if not OUT_RESULTS.is_file():
        pytest.skip("tuning has not been run yet")
    results = pd.read_csv(OUT_RESULTS)
    if len(results) < 4:
        pytest.skip("tuning still in progress")
    assert set(zip(results["model"], results["variant"])) == {
        ("rf", "no_knn"), ("rf", "knn"), ("lgbm", "no_knn"), ("lgbm", "knn")}
    assert (results["n_tuning_origins"] == 12).all()
    assert (results["first_tuning_origin"] == "2021-03-05").all()
    assert (results["last_tuning_origin"] == "2022-01-07").all()
    # The reported best must be a feasible, valid-probability configuration.
    assert (results["mean_tuning_rps"] < INFEASIBLE_LOSS).all()
    assert (results["min_probability"] >= 0).all()
    assert (results["max_probability"] <= 1).all()
    assert (results["max_row_sum_error"] < 1e-9).all()
    # Infeasible trials consumed budget rather than being replaced.
    assert (results["feasible_trials"] + results["infeasible_trials"]
            == results["evaluations_completed"]).all()
    assert (results["evaluations_completed"] <= results["evaluation_budget"]).all()
