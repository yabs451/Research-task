"""Stage 6B: supervised feature/target construction for the competition-grade
feature-based M6 baselines (Random Forest and LightGBM).

Methodology adapted from Samartzis (2025). The PAPER is the methodological
guide; the released implementation under ``Relevant Context/`` is a critically
checked reference only - nothing is imported from it and nothing in it is
modified. Where the released code's row indexing differs from this project's
explicit forecast-origin formulation, the feature is TRANSLATED (see
``FEATURE_NOTES`` below), never copied mechanically.

What one supervised row is
--------------------------
    (forecast origin t, asset a)
        predictors : functions of information available on or before t
        target     : the cross-sectional M6 quintile of asset a's realised
                     cumulative return over the NEXT four weeks, (t, t+20]
                     on the shared weekday calendar

Forecast origins are FOUR-WEEK spaced Fridays anchored on the project's
established M6 origin convention (2022-03-04 +/- 28 calendar days), which is
exactly 20 shared weekdays per step. Daily series are used only INSIDE feature
and target windows, never as separate classifier rows.

Two complete pipelines
----------------------
    variant="no_knn"  genuine missing returns are preserved
    variant="knn"     the daily log-return matrix is KNN-imputed
                      (KNNImputer(n_neighbors=10, weights="distance"), the
                      released configuration) before any downstream quantity

The KNN branch is fitted CAUSALLY, once per origin:
    predictors at t          use a KNN fit on returns with dates <= t
    historical target for t  uses a KNN fit on returns with dates
                             <= target_end(t)
Because target_end(t) is exactly the next grid origin, one KNN fit per grid
date serves both roles and no future information can reach either.

The two branches are allowed to differ in feature values, valid rows and
historical labels. They share the common calendar preparation, the volume
panels and - crucially - they do NOT define the M6 evaluation ground truth,
which remains the untouched official one in scripts/evaluate_m6_rps.py.

Inputs (read-only):
    Data/processed/dataset_d/*.csv          (Stage 6A)
    Data/metadata/m6_asset_metadata.csv
    Data/metadata/m6_round_schedule.csv     (authoritative M6 origins)

Outputs:
    Data/metadata/feature_baseline_origin_schedule.csv
    Data/processed/feature_baseline/supervised_rows_no_knn.csv
    Data/processed/feature_baseline/supervised_rows_knn.csv
    Data/metadata/feature_baseline_build_summary.csv
    reports/feature_baseline_dataset_report.md

Usage:
    python scripts/build_feature_baseline_dataset.py
    python scripts/build_feature_baseline_dataset.py --variant no_knn
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# The project's single tie-aware cross-sectional quintile implementation, reused
# verbatim so historical labels use exactly the official M6 quintile principle.
from evaluate_m6_rps import RANK_COLUMNS, rank_to_quintiles  # noqa: E402

PROCESSED_D = PROJECT_ROOT / "Data" / "processed" / "dataset_d"
ASSET_METADATA = PROJECT_ROOT / "Data" / "metadata" / "m6_asset_metadata.csv"
M6_SCHEDULE = PROJECT_ROOT / "Data" / "metadata" / "m6_round_schedule.csv"

OUT_DIR = PROJECT_ROOT / "Data" / "processed" / "feature_baseline"
OUT_SCHEDULE = PROJECT_ROOT / "Data" / "metadata" / "feature_baseline_origin_schedule.csv"
OUT_SUMMARY = PROJECT_ROOT / "Data" / "metadata" / "feature_baseline_build_summary.csv"
OUT_REPORT = PROJECT_ROOT / "reports" / "feature_baseline_dataset_report.md"

# --------------------------------------------------------------------------- #
# Windows (all expressed in shared weekdays, relative to forecast origin t)
# --------------------------------------------------------------------------- #
HORIZON = 20                 # one M6 period = 4 weeks = 20 shared weekdays
MONTH = 20                   # the paper's "month" = one four-week block
VOL_WINDOW = 60              # ~3 months
MAX_WINDOW = 60              # ~3 months
DOLLAR_VOLUME_WINDOW = 40    # ~2 months
RATIO_WINDOW = 60            # ~3 months
RANK_WINDOW = 20             # ~4 weeks
SEASONAL_LAG = 200           # 10 four-week blocks beyond the most recent one
DEEPEST_LOOKBACK = SEASONAL_LAG + MONTH   # 220 weekdays before the origin

MODEL_HISTORY_FLOOR = pd.Timestamp("2010-01-01")
N_TUNING_ORIGINS = 12
EXPECTED_ASSETS = 100
VARIANTS = ("no_knn", "knn")

KNN_N_NEIGHBORS = 10
KNN_WEIGHTS = "distance"

FEATURE_COLUMNS = [
    "feat_ret_4w_recent",
    "feat_ret_4w_seasonal_11m",
    "feat_vol_3m",
    "feat_max_ret_3m",
    "feat_dollar_volume_2m",
    "feat_abs_ret_to_volume_3m",
    "feat_rank1_freq_4w",
    "feat_rank2_freq_4w",
    "feat_rank4_freq_4w",
    "feat_rank5_freq_4w",
]

# name -> (released code name, window in origin coordinates, released indexing)
FEATURE_NOTES: dict[str, tuple[str, str, str]] = {
    "feat_ret_4w_recent": (
        "feat_0",
        "sum of daily log returns over weekdays [t-19, t]  ==  log(P_t / P_t-20)",
        "monthly_returns.shift(20) on a target-aligned index; translates exactly "
        "to [t-19, t] with no residual lag.",
    ),
    "feat_ret_4w_seasonal_11m": (
        "feat_6",
        "sum of daily log returns over weekdays [t-219, t-200]",
        "monthly_returns.shift(20*11); i.e. feat_0 lagged by a further 200 "
        "weekdays = the 11th four-week block preceding the origin.",
    ),
    "feat_vol_3m": (
        "feat_1",
        "sample std (ddof=1) of daily log returns over [t-59, t]",
        "returns.rolling(60).std().shift(20) - the released window is "
        "[t-58, t+1], one day into the future of its own origin; corrected here.",
    ),
    "feat_max_ret_3m": (
        "feat_2",
        "max daily log return over [t-59, t]",
        "returns.rolling(60).max().shift(20) - same one-day correction.",
    ),
    "feat_dollar_volume_2m": (
        "feat_3",
        "sum over [t-39, t] of log(raw volume * RAW close / 1000)",
        "log(volume*close/1000).rolling(40).sum().shift(20) - same one-day "
        "correction. Not a return: raw close x raw volume is actual traded value.",
    ),
    "feat_abs_ret_to_volume_3m": (
        "feat_4",
        "sum over j in [t-59, t] of |r_j| / D40_j, where D40_j is the 40-day "
        "log-dollar-volume aggregate ending at j",
        "(|returns| / feat_3_pre_shift).rolling(60).sum().shift(20) - the "
        "released code deliberately uses feat_3 BEFORE its own shift(20), so "
        "the denominator is contemporaneous with the numerator.",
    ),
    "feat_rank1_freq_4w": (
        "feat_Rank1",
        "fraction of weekdays in [t-19, t] on which the asset sat in the LOWEST "
        "cross-sectional daily-return quintile",
        "scores_to_quintiles(returns.shift(20)).rolling(20).mean() - same "
        "one-day correction; daily-rank FEATURE, unrelated to the four-week "
        "target.",
    ),
    "feat_rank2_freq_4w": ("feat_Rank2", "as above, quintile 2", "as above"),
    "feat_rank4_freq_4w": ("feat_Rank4", "as above, quintile 4", "as above"),
    "feat_rank5_freq_4w": ("feat_Rank5", "as above, quintile 5 (highest)", "as above"),
}

TARGET_COLUMNS = (
    ["hist_target_log_return", "hist_target_simple_return", "hist_target_quintile"]
    + [f"hist_target_{c}" for c in RANK_COLUMNS]
)

logger = logging.getLogger("build_feature_baseline_dataset")


class ValidationError(RuntimeError):
    """A structural or causality expectation was violated."""


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #

def _read_panel(name: str) -> pd.DataFrame:
    frame = pd.read_csv(PROCESSED_D / name, parse_dates=["date"]).set_index("date")
    frame.index.name = "date"
    return frame


def load_inputs() -> dict[str, object]:
    """Load the Stage 6A panels, asset metadata and the M6 round schedule."""
    returns = _read_panel("dataset_d_daily_log_returns.csv")
    close = _read_panel("dataset_d_close_weekday.csv")
    adjusted = _read_panel("dataset_d_adjusted_close_weekday.csv")
    volume = _read_panel("dataset_d_volume_weekday.csv")

    metadata = pd.read_csv(ASSET_METADATA)
    symbols = list(metadata["symbol"])
    for name, frame in (("returns", returns), ("close", close),
                        ("adjusted_close", adjusted), ("volume", volume)):
        if list(frame.columns) != symbols:
            raise ValidationError(f"{name} columns differ from the asset metadata "
                                  "order.")
    if len(symbols) != EXPECTED_ASSETS:
        raise ValidationError(f"Expected {EXPECTED_ASSETS} assets, got {len(symbols)}.")

    schedule = pd.read_csv(M6_SCHEDULE, parse_dates=["origin_date",
                                                     "forecast_start_date",
                                                     "forecast_end_date"])
    return {
        "returns": returns,
        "close": close,
        "adjusted_close": adjusted,
        "volume": volume,
        "metadata": metadata,
        "symbols": symbols,
        "calendar": returns.index,
        "m6_schedule": schedule,
    }


# --------------------------------------------------------------------------- #
# Forecast-origin grid
# --------------------------------------------------------------------------- #

def build_origin_grid(calendar: pd.DatetimeIndex,
                      m6_schedule: pd.DataFrame) -> pd.DataFrame:
    """The four-week Friday classifier grid anchored on the M6 convention.

    The grid is the set of dates ``first M6 origin + 28k`` (k integer) that lie
    on the shared weekday calendar. Backward extension supplies the historical
    classifier origins; the 12 tuning origins are the 12 grid dates immediately
    preceding the tuning/M6 gap; the forward extension reproduces exactly the
    12 official M6 origins already used by the TSFM experiments.
    """
    anchor = pd.Timestamp(m6_schedule["origin_date"].iloc[0])
    if anchor not in calendar:
        raise ValidationError(f"M6 anchor {anchor.date()} is not on the calendar.")

    dates: list[pd.Timestamp] = [anchor]
    step = pd.Timedelta(days=28)
    back = anchor - step
    while back >= calendar[0]:
        dates.insert(0, back)
        back -= step
    forward = anchor + step
    while forward <= calendar[-1]:
        dates.append(forward)
        forward += step

    position = {d: i for i, d in enumerate(calendar)}
    missing = [d.date().isoformat() for d in dates if d not in position]
    if missing:
        raise ValidationError(f"Grid dates absent from the weekday calendar: {missing}")

    rows = []
    for date in dates:
        i = position[date]
        if i + HORIZON >= len(calendar):
            continue                      # no complete four-week target window
        if i < DEEPEST_LOOKBACK:
            continue                      # deepest predictor window undefined
        if date < MODEL_HISTORY_FLOOR:
            continue                      # source methodology's ~2010 floor
        rows.append({
            "origin_index": i,
            "origin_date": date,
            "target_start_date": calendar[i + 1],
            "target_end_date": calendar[i + HORIZON],
        })
    grid = pd.DataFrame(rows)

    m6_origins = list(pd.to_datetime(m6_schedule["origin_date"]))
    grid["m6_round"] = grid["origin_date"].map(
        {d: r for d, r in zip(m6_origins, m6_schedule["round"])}
    )
    grid["origin_role"] = np.where(grid["m6_round"].notna(), "m6", "historical")

    # The 12 tuning origins: the last 12 grid dates strictly before the M6
    # sample, EXCLUDING the single grid date immediately preceding the first M6
    # origin. That reproduces Samartzis's tuning window (evaluation Mondays
    # 2021-03-08 .. 2022-01-10, whose feature dates are the preceding Fridays)
    # on this project's Friday forecast-origin convention.
    pre_m6 = grid.loc[grid["origin_date"] < m6_origins[0], "origin_date"]
    tuning = list(pre_m6.iloc[-(N_TUNING_ORIGINS + 1):-1])
    grid.loc[grid["origin_date"].isin(tuning), "origin_role"] = "tuning"

    # A row may enter a training set only once its target window has ended.
    grid["eligible_for_training_from"] = grid["target_end_date"]

    if (grid["target_end_date"] != grid["origin_date"] + pd.Timedelta(days=28)).any():
        raise ValidationError("target_end is not exactly 28 calendar days after "
                              "the origin.")
    if not (grid["origin_date"].dt.dayofweek == 4).all():
        raise ValidationError("Not every classifier origin is a Friday.")
    if int((grid["origin_role"] == "tuning").sum()) != N_TUNING_ORIGINS:
        raise ValidationError("Expected exactly 12 tuning origins.")
    if int((grid["origin_role"] == "m6").sum()) != len(m6_schedule):
        raise ValidationError("The forward grid does not reproduce the 12 M6 "
                              "origins.")
    return grid.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Volume-derived intermediate series (identical in both variants)
# --------------------------------------------------------------------------- #

def log_dollar_volume(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """log(raw volume * RAW close / 1000), per asset-day.

    RAW close is used deliberately: this is traded value in currency units, not
    a return, so the split/dividend adjustment that is required for returns
    would be wrong here (adjusted price x unadjusted share count).

    Nonpositive dollar volume - genuine zero-volume days, overwhelmingly the
    London-listed ETFs - makes the logarithm undefined. The released code lets
    it become -inf and only cleans up after the rolling sum:

        feat_3 = np.log(volume * close / 1000)
        feat_3 = feat_3.rolling(40).sum().replace([np.inf, -np.inf], np.nan)

    That behaviour is adopted exactly. Marking the day NaN here and taking a
    STRICT rolling sum in :func:`dollar_volume_aggregate` is arithmetically
    identical to letting -inf propagate and replacing it afterwards - an
    invalid day still destroys every window that contains it - but it avoids
    relying on pandas' incremental float arithmetic with infinities. Verified
    on the full panel: identical NaN pattern, identical finite values.
    """
    dollar = volume * close / 1000.0
    return np.log(dollar.where(dollar > 0))


def dollar_volume_aggregate(ldv: pd.DataFrame) -> pd.DataFrame:
    """The released 40-day rolling SUM of log dollar volume.

    ``min_periods`` equals the window length, so the aggregate is defined only
    when ALL 40 days are usable. A single zero-volume day - or fewer than 40
    weekdays since the asset's inception - leaves it undefined, exactly as the
    released `-inf` propagation followed by `.replace([inf, -inf], nan)` does.
    """
    return ldv.rolling(DOLLAR_VOLUME_WINDOW,
                       min_periods=DOLLAR_VOLUME_WINDOW).sum()


# --------------------------------------------------------------------------- #
# Cross-sectional daily quintile memberships
# --------------------------------------------------------------------------- #

def daily_quintile_memberships(row: pd.Series) -> pd.DataFrame:
    """Tie-aware quintile membership of one day's cross-section of returns.

    Uses the project's shared ``rank_to_quintiles`` (official M6 tie principle,
    index 0 = lowest return). Assets without a return that day are excluded from
    the cross-section and receive NaN.
    """
    available = row.dropna()
    out = pd.DataFrame(np.nan, index=row.index, columns=RANK_COLUMNS)
    if available.empty:
        return out
    out.loc[available.index, :] = rank_to_quintiles(available.to_numpy())
    return out


# --------------------------------------------------------------------------- #
# Features and target at one forecast origin
# --------------------------------------------------------------------------- #

def features_at_origin(returns: pd.DataFrame, dollar_volume_40: pd.DataFrame,
                       origin_index: int) -> pd.DataFrame:
    """The ten paper-selected predictors for every asset at one origin.

    ``returns`` must already be truncated so that no row later than the origin
    exists (the caller enforces this); every window below is additionally
    expressed in closed form relative to ``origin_index`` so that no value dated
    after the origin can be reached even in principle.
    """
    i = origin_index
    if i < DEEPEST_LOOKBACK:
        raise ValidationError("Origin is earlier than the deepest lookback.")
    if len(returns) <= i:
        raise ValidationError("Return matrix does not reach the origin.")

    recent = returns.iloc[i - MONTH + 1: i + 1]                    # [t-19, t]
    seasonal = returns.iloc[i - SEASONAL_LAG - MONTH + 1:
                            i - SEASONAL_LAG + 1]                  # [t-219, t-200]
    three_month = returns.iloc[i - VOL_WINDOW + 1: i + 1]          # [t-59, t]

    out = pd.DataFrame(index=returns.columns)
    out["feat_ret_4w_recent"] = recent.sum(min_count=MONTH)
    out["feat_ret_4w_seasonal_11m"] = seasonal.sum(min_count=MONTH)

    full_3m = three_month.count() == VOL_WINDOW
    out["feat_vol_3m"] = three_month.std(ddof=1).where(full_3m)
    out["feat_max_ret_3m"] = three_month.max().where(full_3m)

    out["feat_dollar_volume_2m"] = dollar_volume_40.iloc[i]

    ratio_window = (returns.iloc[i - RATIO_WINDOW + 1: i + 1].abs()
                    / dollar_volume_40.iloc[i - RATIO_WINDOW + 1: i + 1])
    # Released form: (|returns| / feat_3).rolling(60).sum() - strict, so a
    # single undefined denominator removes the whole aggregate.
    out["feat_abs_ret_to_volume_3m"] = ratio_window.sum(min_count=RATIO_WINDOW)

    rank_totals = pd.DataFrame(0.0, index=returns.columns, columns=RANK_COLUMNS)
    rank_days = pd.Series(0, index=returns.columns, dtype=int)
    for offset in range(RANK_WINDOW):
        day = returns.iloc[i - RANK_WINDOW + 1 + offset]
        membership = daily_quintile_memberships(day)
        present = membership.notna().all(axis=1)
        rank_totals.loc[present] += membership.loc[present]
        rank_days += present.astype(int)
    complete = rank_days == RANK_WINDOW
    frequencies = (rank_totals.div(RANK_WINDOW)).where(complete, other=np.nan)
    out["feat_rank1_freq_4w"] = frequencies["Rank1"]
    out["feat_rank2_freq_4w"] = frequencies["Rank2"]
    out["feat_rank4_freq_4w"] = frequencies["Rank4"]
    out["feat_rank5_freq_4w"] = frequencies["Rank5"]

    return out[FEATURE_COLUMNS]


def target_at_origin(returns: pd.DataFrame, origin_index: int) -> pd.DataFrame:
    """Realised cumulative four-week log return over (t, t+20] and its
    cross-sectional M6 quintile.

    This is the target the PAPER describes and the M6 task requires. It is NOT
    the released code's fitted label, which is the quintile of a single DAILY
    return (``roll_ranks``); that discrepancy is documented and deliberately not
    reproduced. A 20-day lag is not a 20-day cumulative return.
    """
    i = origin_index
    window = returns.iloc[i + 1: i + HORIZON + 1]
    if len(window) != HORIZON:
        raise ValidationError("Incomplete four-week target window.")

    log_return = window.sum(min_count=HORIZON)
    out = pd.DataFrame(index=returns.columns)
    out["hist_target_log_return"] = log_return
    out["hist_target_simple_return"] = np.expm1(log_return)

    available = log_return.dropna()
    for column in RANK_COLUMNS:
        out[f"hist_target_{column}"] = np.nan
    out["hist_target_quintile"] = np.nan
    if not available.empty:
        membership = rank_to_quintiles(available.to_numpy())
        for q, column in enumerate(RANK_COLUMNS):
            out.loc[available.index, f"hist_target_{column}"] = membership[:, q]
        out.loc[available.index, "hist_target_quintile"] = membership.argmax(axis=1)
    out["hist_target_cross_section_size"] = int(available.shape[0])
    return out


# --------------------------------------------------------------------------- #
# Causal KNN imputation of the return matrix
# --------------------------------------------------------------------------- #

def knn_impute_returns(returns: pd.DataFrame,
                       cutoff_index: int) -> pd.DataFrame:
    """KNN-impute the daily log-return matrix using ONLY dates <= cutoff.

    Configuration is the released one: ``KNNImputer(n_neighbors=10,
    weights="distance")`` fitted on the return matrix. Two necessary local
    rules:
      * assets with no observation at all by the cutoff cannot be imputed (they
        would have an entirely empty column) and are left missing, which is what
        keeps the branch from fabricating an asset's pre-inception existence at
        its own forecast origin;
      * calendar rows that are entirely missing contribute nothing and are
        excluded from the fit, then restored as missing.
    """
    window = returns.iloc[: cutoff_index + 1]
    usable_columns = window.columns[window.notna().any()]
    usable_rows = window.index[window.notna().any(axis=1)]

    out = pd.DataFrame(np.nan, index=window.index, columns=window.columns)
    if len(usable_columns) == 0 or len(usable_rows) == 0:
        return out

    block = window.loc[usable_rows, usable_columns]
    imputer = KNNImputer(n_neighbors=KNN_N_NEIGHBORS, weights=KNN_WEIGHTS)
    imputed = imputer.fit_transform(block.to_numpy())
    out.loc[usable_rows, usable_columns] = imputed
    return out


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #

def build_variant(variant: str, data: dict[str, object],
                  grid: pd.DataFrame) -> pd.DataFrame:
    """Build every supervised row of one complete preprocessing pipeline."""
    if variant not in VARIANTS:
        raise ValidationError(f"Unknown variant {variant!r}.")

    returns: pd.DataFrame = data["returns"]
    calendar: pd.DatetimeIndex = data["calendar"]
    metadata: pd.DataFrame = data["metadata"]
    dollar_volume_40 = dollar_volume_aggregate(
        log_dollar_volume(data["close"], data["volume"]))

    # An asset exists at an origin once it has a genuine observation on or
    # before it; this rule is identical in both branches.
    inception = data["adjusted_close"].apply(lambda s: s.first_valid_index())

    grid = grid.sort_values("origin_index").reset_index(drop=True)
    index_of = {int(r.origin_index): k for k, r in enumerate(grid.itertuples())}

    feature_blocks: dict[int, pd.DataFrame] = {}
    target_blocks: dict[int, pd.DataFrame] = {}

    if variant == "no_knn":
        for record in grid.itertuples():
            i = int(record.origin_index)
            feature_blocks[i] = features_at_origin(
                returns.iloc[: i + 1], dollar_volume_40, i)
            target_blocks[i] = target_at_origin(
                returns.iloc[: i + HORIZON + 1], i)
    else:
        # One causal KNN fit per grid date. The fit at grid date g supplies both
        # the PREDICTORS at g (information <= g) and the historical TARGET of
        # the origin whose four-week target window ends exactly at g. Because
        # consecutive origins are exactly HORIZON weekdays apart, that is one
        # fit per origin plus one final fit at the last target_end; no imputed
        # matrix is ever held longer than the single cutoff that consumes it.
        origins = [int(r.origin_index) for r in grid.itertuples()]
        origin_set = set(origins)
        cutoffs = sorted(origin_set | {i + HORIZON for i in origins})
        for n, cutoff in enumerate(cutoffs, start=1):
            imputed = knn_impute_returns(returns, cutoff)
            if cutoff in index_of:
                feature_blocks[cutoff] = features_at_origin(
                    imputed, dollar_volume_40, cutoff)
            source_origin = cutoff - HORIZON
            if source_origin in origin_set:
                # target_at_origin needs rows up to source_origin + HORIZON,
                # which is exactly this cutoff.
                target_blocks[source_origin] = target_at_origin(
                    imputed, source_origin)
            del imputed
            if n % 25 == 0 or n == len(cutoffs):
                logger.info("KNN cutoffs processed: %d/%d (%s)",
                            n, len(cutoffs), calendar[cutoff].date())

    frames = []
    for record in grid.itertuples():
        i = int(record.origin_index)
        block = feature_blocks[i].join(target_blocks[i])
        block.index.name = "symbol"
        block = block.reset_index()
        block.insert(0, "origin_date", record.origin_date)
        block.insert(1, "target_start_date", record.target_start_date)
        block.insert(2, "target_end_date", record.target_end_date)
        block.insert(3, "origin_role", record.origin_role)
        block.insert(4, "m6_round", record.m6_round)
        frames.append(block)

    rows = pd.concat(frames, ignore_index=True)
    rows = rows.merge(
        metadata[["official_order", "symbol", "asset_type",
                  "sector_or_etf_type", "asset_class"]],
        on="symbol", how="left", validate="many_to_one",
    )

    # Assets that did not yet exist at the origin are not rows at all.
    rows["asset_inception_date"] = rows["symbol"].map(inception)
    rows = rows.loc[rows["asset_inception_date"] <= rows["origin_date"]].copy()

    rows["variant"] = variant
    rows["n_missing_features"] = rows[FEATURE_COLUMNS].isna().sum(axis=1)
    rows["features_complete"] = rows["n_missing_features"] == 0
    rows["hist_target_available"] = rows["hist_target_quintile"].notna()
    rows["eligible_for_training_from"] = rows["target_end_date"]

    ordered = (["variant", "origin_date", "target_start_date", "target_end_date",
                "eligible_for_training_from", "origin_role", "m6_round",
                "symbol", "official_order", "asset_type", "sector_or_etf_type",
                "asset_class", "asset_inception_date"]
               + FEATURE_COLUMNS
               + ["features_complete", "n_missing_features"]
               + TARGET_COLUMNS
               + ["hist_target_cross_section_size", "hist_target_available"])
    rows = rows[ordered].sort_values(["origin_date", "official_order"])
    return rows.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def validate_variant(variant: str, rows: pd.DataFrame, data: dict[str, object],
                     grid: pd.DataFrame) -> dict[str, object]:
    """Focused correctness checks; every failure raises."""
    returns: pd.DataFrame = data["returns"]
    calendar: pd.DatetimeIndex = data["calendar"]
    facts: dict[str, object] = {"variant": variant}

    origins = pd.to_datetime(rows["origin_date"].unique())
    steps = set(np.diff(np.sort(origins)).astype("timedelta64[D]").astype(int))
    if steps != {28}:
        raise ValidationError(f"{variant}: classifier origins are not four-week "
                              f"spaced (found steps {sorted(steps)} days).")
    if not (pd.DatetimeIndex(origins).dayofweek == 4).all():
        raise ValidationError(f"{variant}: a classifier origin is not a Friday.")

    position = {d: i for i, d in enumerate(calendar)}
    for record in rows.drop_duplicates("origin_date").itertuples():
        i = position[pd.Timestamp(record.origin_date)]
        if position[pd.Timestamp(record.target_start_date)] != i + 1:
            raise ValidationError(f"{variant}: target_start is not the weekday "
                                  "immediately after the origin.")
        if position[pd.Timestamp(record.target_end_date)] != i + HORIZON:
            raise ValidationError(f"{variant}: target_end is not 20 weekdays "
                                  "after the origin.")

    # Direction of the cross-sectional quintile: the lowest realised return must
    # sit in quintile 0 and the highest in quintile 4, in every origin.
    graded = rows.loc[rows["hist_target_available"]]
    for _, block in graded.groupby("origin_date"):
        lowest = block.loc[block["hist_target_log_return"].idxmin()]
        highest = block.loc[block["hist_target_log_return"].idxmax()]
        if lowest["hist_target_quintile"] != 0 or highest["hist_target_quintile"] != 4:
            raise ValidationError(f"{variant}: quintile direction is inverted.")
        ordered = block.sort_values("hist_target_log_return")["hist_target_quintile"]
        if not ordered.is_monotonic_increasing:
            raise ValidationError(f"{variant}: target quintile is not monotone "
                                  "non-decreasing in the realised return.")

    # The target must be a 20-day CUMULATIVE return, not a lagged daily return.
    sample = graded.sample(min(400, len(graded)), random_state=0)
    for record in sample.itertuples():
        i = position[pd.Timestamp(record.origin_date)]
        window = returns[record.symbol].iloc[i + 1: i + HORIZON + 1]
        if window.notna().all():
            expected = float(window.sum())
            if not np.isclose(record.hist_target_log_return, expected, atol=1e-9):
                if variant == "no_knn":
                    raise ValidationError(
                        f"{variant}: target for {record.symbol} at "
                        f"{record.origin_date} is not the cumulative return over "
                        "the next 20 weekdays.")
    facts["target_spot_checks"] = int(len(sample))

    # No target may be usable before its window has ended.
    if (rows["eligible_for_training_from"] != rows["target_end_date"]).any():
        raise ValidationError(f"{variant}: training eligibility is not "
                              "target_end.")
    if (pd.to_datetime(rows["eligible_for_training_from"])
            <= pd.to_datetime(rows["origin_date"])).any():
        raise ValidationError(f"{variant}: a target is eligible at or before its "
                              "own origin.")

    facts["rows"] = int(len(rows))
    facts["origins"] = int(rows["origin_date"].nunique())
    facts["complete_feature_rows"] = int(rows["features_complete"].sum())
    facts["rows_with_target"] = int(rows["hist_target_available"].sum())
    facts["trainable_rows"] = int(
        (rows["features_complete"] & rows["hist_target_available"]).sum())
    facts["missing_by_feature"] = {
        c: int(rows[c].isna().sum()) for c in FEATURE_COLUMNS}
    return facts


def spot_check_feature_windows(rows: pd.DataFrame, data: dict[str, object],
                               n_checks: int = 60) -> dict[str, int]:
    """Recompute each no-KNN feature from first principles for random rows.

    This is the numerical verification that the released ``shift(20)``-style
    indexing was translated correctly rather than copied: every window below is
    written out explicitly in origin coordinates.
    """
    returns: pd.DataFrame = data["returns"]
    calendar: pd.DatetimeIndex = data["calendar"]
    position = {d: i for i, d in enumerate(calendar)}
    ldv = log_dollar_volume(data["close"], data["volume"])
    ldv40 = dollar_volume_aggregate(ldv)

    checked = {c: 0 for c in FEATURE_COLUMNS}
    sample = rows.loc[rows["features_complete"]].sample(
        min(n_checks, int(rows["features_complete"].sum())), random_state=7)
    for record in sample.itertuples():
        i = position[pd.Timestamp(record.origin_date)]
        s = record.symbol
        r = returns[s]

        pairs = [
            ("feat_ret_4w_recent", record.feat_ret_4w_recent,
             float(r.iloc[i - 19: i + 1].sum())),
            ("feat_ret_4w_seasonal_11m", record.feat_ret_4w_seasonal_11m,
             float(r.iloc[i - 219: i - 199].sum())),
            ("feat_vol_3m", record.feat_vol_3m,
             float(r.iloc[i - 59: i + 1].std(ddof=1))),
            ("feat_max_ret_3m", record.feat_max_ret_3m,
             float(r.iloc[i - 59: i + 1].max())),
            ("feat_dollar_volume_2m", record.feat_dollar_volume_2m,
             float(ldv[s].iloc[i - 39: i + 1].sum(min_count=40))),
            ("feat_abs_ret_to_volume_3m", record.feat_abs_ret_to_volume_3m,
             float((r.iloc[i - 59: i + 1].abs()
                    / ldv40[s].iloc[i - 59: i + 1]).sum(min_count=60))),
        ]
        for name, produced, expected in pairs:
            if not np.isclose(produced, expected, rtol=1e-9, atol=1e-12):
                raise ValidationError(
                    f"Feature window mismatch for {name} / {s} @ "
                    f"{record.origin_date}: {produced!r} vs {expected!r}")
            checked[name] += 1

        # Rank frequencies: recompute the 20-day trailing cross-sectional
        # daily-quintile histogram directly.
        totals = np.zeros(5)
        for j in range(i - 19, i + 1):
            day = returns.iloc[j].dropna()
            totals += rank_to_quintiles(day.to_numpy())[day.index.get_loc(s)]
        for name, q in (("feat_rank1_freq_4w", 0), ("feat_rank2_freq_4w", 1),
                        ("feat_rank4_freq_4w", 3), ("feat_rank5_freq_4w", 4)):
            produced = getattr(record, name)
            if not np.isclose(produced, totals[q] / 20.0, atol=1e-12):
                raise ValidationError(
                    f"Feature window mismatch for {name} / {s} @ "
                    f"{record.origin_date}: {produced!r} vs {totals[q] / 20.0!r}")
            checked[name] += 1
    return checked


def verify_knn_causality(data: dict[str, object], grid: pd.DataFrame,
                         knn_rows: pd.DataFrame) -> dict[str, object]:
    """Positive leakage test for the KNN branch.

    Corrupts every return AFTER a chosen cutoff to an extreme value, rebuilds
    the causal imputation and the features at that origin, and requires the
    result to be bit-identical to the uncorrupted build. If any future
    observation could reach the predictors, this fails.
    """
    returns: pd.DataFrame = data["returns"]
    dollar_volume_40 = dollar_volume_aggregate(
        log_dollar_volume(data["close"], data["volume"]))
    record = grid.loc[grid["origin_role"] == "tuning"].iloc[0]
    i = int(record.origin_index)

    clean = features_at_origin(knn_impute_returns(returns, i), dollar_volume_40, i)
    poisoned_returns = returns.copy()
    poisoned_returns.iloc[i + 1:] = 9.0
    poisoned = features_at_origin(
        knn_impute_returns(poisoned_returns, i), dollar_volume_40, i)

    if not clean.equals(poisoned):
        raise ValidationError("KNN predictor construction is contaminated by "
                              "post-origin observations.")

    # Negative control: corrupting information the predictors are entitled to
    # see MUST change them, otherwise the test above proves nothing.
    within = returns.copy()
    within.iloc[i - 10: i + 1] = 9.0
    sensitive = features_at_origin(
        knn_impute_returns(within, i), dollar_volume_40, i)
    if sensitive["feat_ret_4w_recent"].equals(clean["feat_ret_4w_recent"]):
        raise ValidationError("Predictors do not respond to in-window data - "
                              "the leakage test is not meaningful.")

    # And the same test for the historical target: corrupting data after
    # target_end must not change the label, while corrupting data inside the
    # target window must.
    end = i + HORIZON
    clean_target = target_at_origin(knn_impute_returns(returns, end), i)
    after = returns.copy()
    after.iloc[end + 1:] = 9.0
    poisoned_target = target_at_origin(knn_impute_returns(after, end), i)
    if not clean_target.equals(poisoned_target):
        raise ValidationError("KNN target construction is contaminated by data "
                              "after target_end.")

    inside = returns.copy()
    inside.iloc[i + 1: end + 1] = 9.0
    changed_target = target_at_origin(knn_impute_returns(inside, end), i)
    if changed_target["hist_target_log_return"].equals(
            clean_target["hist_target_log_return"]):
        raise ValidationError("The historical target does not depend on its own "
                              "target window - the test is not meaningful.")

    return {
        "leakage_test_origin": record.origin_date.date().isoformat(),
        "predictor_test": "identical under post-origin corruption",
        "target_test": "identical under post-target_end corruption; "
                       "changes under in-window corruption",
    }


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def build_report(grid: pd.DataFrame, built: dict[str, pd.DataFrame],
                 facts: dict[str, dict], knn_facts: dict[str, object],
                 window_checks: dict[str, int],
                 volume_facts: dict[str, object]) -> str:
    tuning = grid.loc[grid["origin_role"] == "tuning", "origin_date"]
    m6 = grid.loc[grid["origin_role"] == "m6", "origin_date"]
    no_knn, knn = built["no_knn"], built["knn"]

    merged = no_knn.merge(
        knn[["origin_date", "symbol", "hist_target_quintile"]],
        on=["origin_date", "symbol"], how="inner", suffixes=("_no_knn", "_knn"))
    both = merged.dropna(subset=["hist_target_quintile_no_knn",
                                 "hist_target_quintile_knn"])
    label_agreement = float((both["hist_target_quintile_no_knn"]
                             == both["hist_target_quintile_knn"]).mean())

    lines = [
        "# Feature-Based Baseline Dataset Report (Stage 6B)",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "## 1. Classifier origin grid",
        "",
        f"- {len(grid)} four-week-spaced Friday forecast origins, "
        f"{grid['origin_date'].min().date()} .. {grid['origin_date'].max().date()}.",
        "- Every step is exactly 28 calendar days = 20 shared weekdays. No "
        "origin is generated on any other weekday, so no two supervised rows "
        "share an overlapping four-week target window.",
        f"- Roles: {int((grid['origin_role'] == 'historical').sum())} historical, "
        f"{len(tuning)} tuning, {len(m6)} M6.",
        f"- Tuning origins: {', '.join(d.date().isoformat() for d in tuning)}",
        f"- M6 origins: {', '.join(d.date().isoformat() for d in m6)} "
        "(identical to Data/metadata/m6_round_schedule.csv).",
        "",
        "## 2. Row counts per variant",
        "",
        "| variant | rows | origins | rows with complete features | rows with a "
        "target | trainable rows |",
        "|---|---|---|---|---|---|",
        *[f"| {v} | {facts[v]['rows']} | {facts[v]['origins']} | "
          f"{facts[v]['complete_feature_rows']} | {facts[v]['rows_with_target']} | "
          f"{facts[v]['trainable_rows']} |" for v in VARIANTS],
        "",
        f"- Historical label agreement where BOTH branches produce a label: "
        f"{label_agreement:.4f} over {len(both)} shared rows. The branches are "
        "deliberately not forced to match.",
        "",
        "## 3. Missing predictor values per variant",
        "",
        "| feature | no_knn missing | knn missing |",
        "|---|---|---|",
        *[f"| {c} | {facts['no_knn']['missing_by_feature'][c]} | "
          f"{facts['knn']['missing_by_feature'][c]} |" for c in FEATURE_COLUMNS],
        "",
        "## 4. Volume treatment",
        "",
        f"- Asset-days with nonpositive dollar volume (log undefined): "
        f"{volume_facts['undefined_days']} of {volume_facts['total_days']} "
        f"({volume_facts['undefined_share']:.4%}), concentrated in "
        f"{volume_facts['affected_assets']} assets, overwhelmingly the "
        "London-listed ETFs plus DRE's forward-filled post-acquisition tail.",
        "- Treatment: the released Samartzis behaviour exactly. "
        "`log(volume*close/1000)` is undefined on such a day, so the "
        f"{DOLLAR_VOLUME_WINDOW}-day rolling SUM of any window containing one "
        "is undefined too (`.replace([inf,-inf], nan)` in the released code), "
        f"and the {RATIO_WINDOW}-day absolute-return-to-volume sum is strict, "
        "so one undefined denominator removes it as well. A single "
        f"zero-volume day therefore removes up to {DOLLAR_VOLUME_WINDOW} "
        f"consecutive aggregates and up to {DOLLAR_VOLUME_WINDOW + RATIO_WINDOW - 1} "
        "consecutive ratio values.",
        "- These two features are the only ones affected; the aggregate also "
        f"stays undefined until {DOLLAR_VOLUME_WINDOW} weekdays have elapsed "
        "since inception, which falls out of the same rule.",
        "- Raw Dataset D is not modified in any way.",
        "",
        "## 5. Feature-window numerical spot checks",
        "",
        "Each feature was recomputed from first principles, in explicit "
        "origin coordinates, for randomly sampled rows:",
        "",
        "| feature | released name | window relative to origin t | independent "
        "recomputations |",
        "|---|---|---|---|",
        *[f"| {c} | {FEATURE_NOTES[c][0]} | {FEATURE_NOTES[c][1]} | "
          f"{window_checks[c]} |" for c in FEATURE_COLUMNS],
        "",
        "## 6. KNN causality",
        "",
        f"- Leakage test origin: {knn_facts['leakage_test_origin']}",
        f"- Predictors: {knn_facts['predictor_test']}",
        f"- Historical target: {knn_facts['target_test']}",
        "",
        "## 7. Evaluation separation",
        "",
        "- Every target column in these files is prefixed `hist_target_` and is "
        "a BRANCH-SPECIFIC historical supervised label built from Dataset D.",
        "- The official M6 evaluation ground truth is NOT defined here. It "
        "remains `scripts/evaluate_m6_rps.py` built from `Data/assets_m6.csv`, "
        "which this stage neither reads for labels nor modifies.",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def volume_diagnostics(data: dict[str, object]) -> dict[str, object]:
    ldv = log_dollar_volume(data["close"], data["volume"])
    observed = data["adjusted_close"].notna()
    undefined = int((ldv.isna() & observed).sum().sum())
    per_asset = (ldv.isna() & observed).sum()
    return {
        "undefined_days": undefined,
        "total_days": int(observed.sum().sum()),
        "undefined_share": undefined / int(observed.sum().sum()),
        "affected_assets": int((per_asset > 0).sum()),
        "per_asset": per_asset[per_asset > 0].sort_values(ascending=False),
    }


def run(variants: tuple[str, ...] = VARIANTS) -> dict[str, pd.DataFrame]:
    data = load_inputs()
    grid = build_origin_grid(data["calendar"], data["m6_schedule"])

    schedule_out = grid.copy()
    for column in ("origin_date", "target_start_date", "target_end_date",
                   "eligible_for_training_from"):
        schedule_out[column] = schedule_out[column].dt.strftime("%Y-%m-%d")
    OUT_SCHEDULE.parent.mkdir(parents=True, exist_ok=True)
    schedule_out.to_csv(OUT_SCHEDULE, index=False)
    logger.info("Wrote %s (%d origins)", OUT_SCHEDULE, len(schedule_out))

    built: dict[str, pd.DataFrame] = {}
    facts: dict[str, dict] = {}
    for variant in variants:
        logger.info("Building variant %s ...", variant)
        rows = build_variant(variant, data, grid)
        facts[variant] = validate_variant(variant, rows, data, grid)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUT_DIR / f"supervised_rows_{variant}.csv"
        out = rows.copy()
        for column in ("origin_date", "target_start_date", "target_end_date",
                       "eligible_for_training_from", "asset_inception_date"):
            out[column] = pd.to_datetime(out[column]).dt.strftime("%Y-%m-%d")
        out.to_csv(path, index=False)
        logger.info("Wrote %s (%d rows)", path, len(out))
        built[variant] = rows

    if set(variants) == set(VARIANTS):
        window_checks = spot_check_feature_windows(built["no_knn"], data)
        knn_facts = verify_knn_causality(data, grid, built["knn"])
        volume_facts = volume_diagnostics(data)

        summary = pd.DataFrame([
            {"variant": v,
             "rows": facts[v]["rows"],
             "origins": facts[v]["origins"],
             "complete_feature_rows": facts[v]["complete_feature_rows"],
             "rows_with_target": facts[v]["rows_with_target"],
             "trainable_rows": facts[v]["trainable_rows"],
             **{f"missing_{c}": facts[v]["missing_by_feature"][c]
                for c in FEATURE_COLUMNS}}
            for v in VARIANTS
        ])
        OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(OUT_SUMMARY, index=False)
        logger.info("Wrote %s", OUT_SUMMARY)

        OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
        OUT_REPORT.write_text(
            build_report(grid, built, facts, knn_facts, window_checks,
                         volume_facts),
            encoding="utf-8")
        logger.info("Wrote %s", OUT_REPORT)
    return built


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, default=None,
                        help="Build only one pipeline (default: both).")
    args = parser.parse_args()
    variants = VARIANTS if args.variant is None else (args.variant,)
    try:
        run(variants)
    except ValidationError as exc:
        logger.error("Validation failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
