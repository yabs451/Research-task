"""Focused tests for the Stage 6A/6B feature-baseline preprocessing rules.

Unit tests exercise the window arithmetic on small synthetic frames; the
integration tests check the actual generated artifacts (run
scripts/preprocess_dataset_d.py and scripts/build_feature_baseline_dataset.py
first) and are skipped when those files are absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_feature_baseline_dataset import (  # noqa: E402
    DEEPEST_LOOKBACK,
    FEATURE_COLUMNS,
    HORIZON,
    MODEL_HISTORY_FLOOR,
    N_TUNING_ORIGINS,
    OUT_DIR,
    OUT_SCHEDULE,
    VARIANTS,
    build_origin_grid,
    dollar_volume_aggregate,
    features_at_origin,
    knn_impute_returns,
    log_dollar_volume,
    target_at_origin,
)
from preprocess_dataset_d import (  # noqa: E402
    EXPECTED_WEEKDAYS,
    weekday_calendar,
)


# --------------------------------------------------------------------------- #
# Synthetic fixtures
# --------------------------------------------------------------------------- #

def _synthetic_returns(n_dates: int = 400, n_assets: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    index = pd.date_range("2019-01-01", periods=n_dates, freq="B")
    return pd.DataFrame(
        rng.normal(0.0, 0.01, size=(n_dates, n_assets)),
        index=index,
        columns=[f"A{i:02d}" for i in range(n_assets)],
    )


def _synthetic_volume_aggregate(returns: pd.DataFrame) -> pd.DataFrame:
    close = pd.DataFrame(50.0, index=returns.index, columns=returns.columns)
    volume = pd.DataFrame(1_000_000.0, index=returns.index, columns=returns.columns)
    return dollar_volume_aggregate(log_dollar_volume(close, volume))


# --------------------------------------------------------------------------- #
# Calendar
# --------------------------------------------------------------------------- #

def test_dataset_d_weekday_calendar_is_weekdays_only_and_complete() -> None:
    cal = weekday_calendar()
    assert len(cal) == EXPECTED_WEEKDAYS == 3676
    assert (cal.dayofweek <= 4).all()
    assert cal.min() == pd.Timestamp("2009-01-02")
    assert cal.max() == pd.Timestamp("2023-02-03")
    assert cal.is_monotonic_increasing and not cal.duplicated().any()


# --------------------------------------------------------------------------- #
# Origin grid
# --------------------------------------------------------------------------- #

def test_origin_grid_is_four_week_spaced_and_reproduces_the_m6_schedule() -> None:
    schedule = pd.read_csv(PROJECT_ROOT / "Data" / "metadata" / "m6_round_schedule.csv",
                           parse_dates=["origin_date", "forecast_start_date",
                                        "forecast_end_date"])
    grid = build_origin_grid(weekday_calendar(), schedule)

    steps = np.diff(grid["origin_date"].to_numpy()).astype("timedelta64[D]").astype(int)
    assert set(steps) == {28}
    assert (grid["origin_date"].dt.dayofweek == 4).all()
    assert (grid["origin_date"] >= MODEL_HISTORY_FLOOR).all()

    m6 = grid.loc[grid["origin_role"] == "m6"]
    assert list(m6["origin_date"]) == list(schedule["origin_date"])
    assert list(m6["target_start_date"]) == list(schedule["forecast_start_date"])
    assert list(m6["target_end_date"]) == list(schedule["forecast_end_date"])

    tuning = grid.loc[grid["origin_role"] == "tuning", "origin_date"]
    assert len(tuning) == N_TUNING_ORIGINS
    assert tuning.iloc[0] == pd.Timestamp("2021-03-05")
    assert tuning.iloc[-1] == pd.Timestamp("2022-01-07")

    # A row may never be trained on before its own target window has closed.
    assert (grid["eligible_for_training_from"] == grid["target_end_date"]).all()
    assert (grid["eligible_for_training_from"] > grid["origin_date"]).all()


# --------------------------------------------------------------------------- #
# Feature window arithmetic (the translation of the released shift(20) logic)
# --------------------------------------------------------------------------- #

def test_recent_four_week_return_is_the_twenty_days_ending_at_the_origin() -> None:
    returns = _synthetic_returns()
    ldv40 = _synthetic_volume_aggregate(returns)
    i = 300
    feats = features_at_origin(returns.iloc[: i + 1], ldv40, i)
    expected = returns["A00"].iloc[i - 19: i + 1].sum()
    assert feats.loc["A00", "feat_ret_4w_recent"] == pytest.approx(expected)


def test_seasonal_feature_is_the_eleventh_preceding_four_week_block() -> None:
    returns = _synthetic_returns()
    ldv40 = _synthetic_volume_aggregate(returns)
    i = 300
    feats = features_at_origin(returns.iloc[: i + 1], ldv40, i)
    expected = returns["A03"].iloc[i - 219: i - 199].sum()
    assert feats.loc["A03", "feat_ret_4w_seasonal_11m"] == pytest.approx(expected)
    assert DEEPEST_LOOKBACK == 220


def test_no_feature_can_reach_beyond_the_forecast_origin() -> None:
    """Corrupting every observation after the origin must leave features fixed."""
    returns = _synthetic_returns()
    ldv40 = _synthetic_volume_aggregate(returns)
    i = 300
    clean = features_at_origin(returns.iloc[: i + 1], ldv40, i)
    poisoned = returns.copy()
    poisoned.iloc[i + 1:] = 5.0
    dirty = features_at_origin(poisoned.iloc[: i + 1], ldv40, i)
    pd.testing.assert_frame_equal(clean, dirty)
    assert list(clean.columns) == FEATURE_COLUMNS


def test_rank_frequencies_are_nonnegative_and_bounded() -> None:
    returns = _synthetic_returns()
    ldv40 = _synthetic_volume_aggregate(returns)
    feats = features_at_origin(returns.iloc[:301], ldv40, 300)
    rank_cols = [c for c in FEATURE_COLUMNS if "_freq_" in c]
    assert (feats[rank_cols] >= 0).all().all()
    assert (feats[rank_cols].sum(axis=1) <= 1.0 + 1e-12).all()


def test_dollar_volume_aggregate_is_the_released_forty_day_rolling_sum() -> None:
    index = pd.date_range("2020-01-01", periods=60, freq="B")
    close = pd.DataFrame({"X": np.linspace(10, 20, 60)}, index=index)
    volume = pd.DataFrame({"X": np.linspace(1e6, 2e6, 60)}, index=index)
    ldv = log_dollar_volume(close, volume)
    assert ldv.notna().all().all()
    aggregate = dollar_volume_aggregate(ldv)
    pd.testing.assert_frame_equal(aggregate, ldv.rolling(40).sum())
    # Undefined until a full 40-weekday window exists.
    assert aggregate["X"].iloc[:39].isna().all()
    assert aggregate["X"].iloc[39:].notna().all()


def test_zero_volume_days_are_missing_not_infinite() -> None:
    index = pd.date_range("2020-01-01", periods=5, freq="B")
    close = pd.DataFrame({"X": [10.0] * 5}, index=index)
    volume = pd.DataFrame({"X": [1e6, 0.0, 1e6, 0.0, 1e6]}, index=index)
    ldv = log_dollar_volume(close, volume)
    assert ldv["X"].isna().tolist() == [False, True, False, True, False]
    assert np.isfinite(ldv["X"].dropna()).all()


def test_marking_invalid_days_nan_equals_the_released_minus_inf_propagation() -> None:
    """The released code lets log(0) become -inf and cleans up after the sum."""
    index = pd.date_range("2020-01-01", periods=120, freq="B")
    rng = np.random.default_rng(3)
    close = pd.DataFrame({"X": rng.uniform(10, 20, 120)}, index=index)
    vol = rng.uniform(1e5, 1e6, 120)
    vol[[50, 90]] = 0.0
    volume = pd.DataFrame({"X": vol}, index=index)

    with np.errstate(divide="ignore"):
        literal = np.log(volume * close / 1000.0)          # -inf on zero volume
    literal = literal.rolling(40).sum().replace([np.inf, -np.inf], np.nan)
    ours = dollar_volume_aggregate(log_dollar_volume(close, volume))

    pd.testing.assert_frame_equal(ours, literal)


def test_one_zero_volume_day_destroys_forty_consecutive_aggregates() -> None:
    index = pd.date_range("2020-01-01", periods=140, freq="B")
    close = pd.DataFrame({"X": [15.0] * 140}, index=index)
    vol = np.full(140, 5e5)
    vol[80] = 0.0
    aggregate = dollar_volume_aggregate(
        log_dollar_volume(close, pd.DataFrame({"X": vol}, index=index)))
    # Windows ending at 80 .. 119 all contain the invalid day.
    assert aggregate["X"].iloc[80:120].isna().all()
    assert aggregate["X"].iloc[39:80].notna().all()
    assert aggregate["X"].iloc[120:].notna().all()


def test_ratio_feature_is_strict_over_its_sixty_day_window() -> None:
    """One undefined denominator removes feat_abs_ret_to_volume_3m entirely."""
    returns = _synthetic_returns(n_dates=400, n_assets=4)
    index = returns.index
    close = pd.DataFrame(50.0, index=index, columns=returns.columns)
    vol = pd.DataFrame(1_000_000.0, index=index, columns=returns.columns)
    vol.iloc[260, 0] = 0.0                      # a single zero-volume day
    ldv40 = dollar_volume_aggregate(log_dollar_volume(close, vol))
    feats = features_at_origin(returns.iloc[:301], ldv40, 300)
    # Origin 300's own 40-day window is [261, 300] and is clean, so the
    # dollar-volume feature survives...
    assert np.isfinite(feats.loc["A00", "feat_dollar_volume_2m"])
    # ...but the 60-day ratio window [241, 300] contains days whose OWN 40-day
    # aggregate spans day 260, so the strict sum removes the ratio feature.
    assert np.isnan(feats.loc["A00", "feat_abs_ret_to_volume_3m"])
    assert feats.loc["A01", "feat_abs_ret_to_volume_3m"] == pytest.approx(
        float((returns["A01"].iloc[241:301].abs() / ldv40["A01"].iloc[241:301]).sum())
    )
    # Only these two features are touched; the other eight survive.
    others = [c for c in FEATURE_COLUMNS
              if c not in ("feat_dollar_volume_2m", "feat_abs_ret_to_volume_3m")]
    assert feats.loc["A00", others].notna().all()


# --------------------------------------------------------------------------- #
# Target construction
# --------------------------------------------------------------------------- #

def test_target_is_the_cumulative_next_twenty_day_return_not_a_lagged_daily_one() -> None:
    returns = _synthetic_returns()
    i = 300
    target = target_at_origin(returns.iloc[: i + HORIZON + 1], i)
    expected = returns["A05"].iloc[i + 1: i + 21].sum()
    assert target.loc["A05", "hist_target_log_return"] == pytest.approx(expected)
    # A 20-day lag is emphatically not the same object.
    assert target.loc["A05", "hist_target_log_return"] != pytest.approx(
        returns["A05"].iloc[i + 20])
    assert target.loc["A05", "hist_target_simple_return"] == pytest.approx(
        np.expm1(expected))


def test_target_quintiles_run_from_worst_to_best() -> None:
    returns = _synthetic_returns()
    i = 300
    target = target_at_origin(returns.iloc[: i + HORIZON + 1], i)
    ordered = target.sort_values("hist_target_log_return")["hist_target_quintile"]
    assert ordered.is_monotonic_increasing
    assert ordered.iloc[0] == 0 and ordered.iloc[-1] == 4
    memberships = target[[f"hist_target_Rank{q}" for q in range(1, 6)]]
    assert memberships.sum(axis=1).round(12).eq(1.0).all()


# --------------------------------------------------------------------------- #
# Causal KNN
# --------------------------------------------------------------------------- #

def test_knn_imputation_uses_no_data_after_the_cutoff() -> None:
    returns = _synthetic_returns()
    returns.iloc[:50, 0] = np.nan          # a late-listing asset
    cutoff = 200
    clean = knn_impute_returns(returns, cutoff)
    poisoned = returns.copy()
    poisoned.iloc[cutoff + 1:] = 7.0
    dirty = knn_impute_returns(poisoned, cutoff)
    pd.testing.assert_frame_equal(clean, dirty)
    assert len(clean) == cutoff + 1
    assert clean.notna().all().all()


def test_knn_leaves_an_asset_missing_while_it_has_no_history_at_the_cutoff() -> None:
    returns = _synthetic_returns()
    returns.iloc[:150, 0] = np.nan
    imputed = knn_impute_returns(returns, 100)
    assert imputed["A00"].isna().all()      # cannot be fabricated before it exists
    assert imputed.drop(columns="A00").notna().all().all()


# --------------------------------------------------------------------------- #
# Integration checks on the generated artifacts
# --------------------------------------------------------------------------- #

def _load(variant: str) -> pd.DataFrame:
    path = OUT_DIR / f"supervised_rows_{variant}.csv"
    if not path.is_file():
        pytest.skip(f"{path} not generated yet")
    return pd.read_csv(path, parse_dates=["origin_date", "target_start_date",
                                          "target_end_date"])


@pytest.mark.parametrize("variant", VARIANTS)
def test_generated_rows_are_four_week_spaced_and_carry_every_feature(variant: str) -> None:
    rows = _load(variant)
    origins = np.sort(rows["origin_date"].unique())
    steps = set(np.diff(origins).astype("timedelta64[D]").astype(int))
    assert steps == {28}
    assert set(FEATURE_COLUMNS).issubset(rows.columns)
    assert (rows["target_end_date"] - rows["origin_date"]
            == pd.Timedelta(days=28)).all()
    assert not rows.duplicated(["origin_date", "symbol"]).any()
    # No supervised row may predate the asset itself.
    assert (pd.to_datetime(rows["asset_inception_date"])
            <= rows["origin_date"]).all()


def test_the_two_variants_are_not_forced_to_agree() -> None:
    no_knn, knn = _load("no_knn"), _load("knn")
    merged = no_knn.merge(knn, on=["origin_date", "symbol"],
                          suffixes=("_no_knn", "_knn"))
    assert len(merged) > 0
    # The KNN branch must supply at least as many usable rows...
    assert int(knn["features_complete"].sum()) >= int(no_knn["features_complete"].sum())
    # ...and the labels are allowed to differ where preprocessing reaches them.
    differing = (merged["hist_target_quintile_no_knn"]
                 != merged["hist_target_quintile_knn"]).sum()
    assert differing >= 0            # recorded, never forced to zero


def test_official_m6_origins_survive_into_both_variants() -> None:
    schedule = pd.read_csv(OUT_SCHEDULE, parse_dates=["origin_date"])
    m6 = schedule.loc[schedule["origin_role"] == "m6", "origin_date"]
    official = pd.read_csv(PROJECT_ROOT / "Data" / "metadata" / "m6_round_schedule.csv",
                           parse_dates=["origin_date"])["origin_date"]
    assert list(m6) == list(official)
    for variant in VARIANTS:
        rows = _load(variant)
        present = rows.loc[rows["origin_role"] == "m6", "origin_date"].unique()
        assert sorted(pd.to_datetime(present)) == list(official)
