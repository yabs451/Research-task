"""Stage 6A: preprocess raw Dataset D (OHLCV) onto the project's shared weekday
calendar and build adjusted-close daily log returns.

This is the COMMON preparation shared by both feature-engineering variants
(KNN and no-KNN) built in scripts/build_feature_baseline_dataset.py.

Input (read-only, never modified, never re-downloaded):
    Data/raw/dataset_d_eodhd/dataset_d_ohlcv.csv

Substantive transformations - and ONLY these:
    1. reindex each of the six raw fields onto ONE shared Monday-to-Friday
       calendar (2009-01-02 .. 2023-02-03, 3676 dates), reproducing the
       convention established for Dataset A in scripts/preprocess_dataset_a.py
       (which applied the same rule over its own shorter span);
    2. forward-fill each asset/field only AFTER that asset's first genuine
       observation (pandas ffill never fills before the first valid value, so
       pre-inception history is preserved as missing);
    3. daily log returns from ADJUSTED CLOSE: log(P[t] / P[t-1]).

Deliberately NOT done here (see documentation_part2.txt):
    no backward filling, no interpolation, no KNN imputation, no ECOD, no
    winsorising, no clipping, no normalisation, no outlier replacement, no
    correction of the known KR adjusted-close provider break, no special-casing
    of any asset.

Outputs (all new; nothing existing is overwritten):
    Data/processed/dataset_d/dataset_d_open_weekday.csv
    Data/processed/dataset_d/dataset_d_high_weekday.csv
    Data/processed/dataset_d/dataset_d_low_weekday.csv
    Data/processed/dataset_d/dataset_d_close_weekday.csv
    Data/processed/dataset_d/dataset_d_adjusted_close_weekday.csv
    Data/processed/dataset_d/dataset_d_volume_weekday.csv
    Data/processed/dataset_d/dataset_d_daily_log_returns.csv
    Data/metadata/dataset_d_preprocessing_summary.csv
    reports/dataset_d_preprocessing_report.md

Usage:
    python scripts/preprocess_dataset_d.py
"""

from __future__ import annotations

import hashlib
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_OHLCV = PROJECT_ROOT / "Data" / "raw" / "dataset_d_eodhd" / "dataset_d_ohlcv.csv"
OFFICIAL_M6_CSV = PROJECT_ROOT / "Data" / "raw" / "m6_official" / "assets_m6.csv"

OUT_DIR = PROJECT_ROOT / "Data" / "processed" / "dataset_d"
OUT_SUMMARY = PROJECT_ROOT / "Data" / "metadata" / "dataset_d_preprocessing_summary.csv"
OUT_REPORT = PROJECT_ROOT / "reports" / "dataset_d_preprocessing_report.md"

START_DATE = pd.Timestamp("2009-01-02")   # first genuine observation in Dataset D
END_DATE = pd.Timestamp("2023-02-03")     # final M6 evaluation date
EXPECTED_ASSETS = 100
EXPECTED_WEEKDAYS = 3676

PRICE_FIELDS = ("open", "high", "low", "close", "adjusted_close")
ALL_FIELDS = PRICE_FIELDS + ("volume",)
RETURN_BASIS = "adjusted_close"

DRE_LAST_GENUINE = pd.Timestamp("2022-10-03")
KR_BREAK_DATE = pd.Timestamp("2022-10-21")

# The 13 Friday anchors of the established M6 round schedule (12 origins plus
# the final evaluation date). Verified to exist on the shared calendar.
M6_FRIDAY_ANCHORS = [pd.Timestamp(d) for d in (
    "2022-03-04", "2022-04-01", "2022-04-29", "2022-05-27", "2022-06-24",
    "2022-07-22", "2022-08-19", "2022-09-16", "2022-10-14", "2022-11-11",
    "2022-12-09", "2023-01-06", "2023-02-03",
)]

logger = logging.getLogger("preprocess_dataset_d")


class ValidationError(RuntimeError):
    """A structural expectation about the input or output was violated."""


# --------------------------------------------------------------------------- #
# Loading and validation
# --------------------------------------------------------------------------- #

def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_official_order(csv_path: Path = OFFICIAL_M6_CSV) -> list[str]:
    """Official 100-asset order (names/order only - prices are never used)."""
    df = pd.read_csv(csv_path)
    symbols = list(dict.fromkeys(df["symbol"].astype(str).str.strip()))
    if len(symbols) != EXPECTED_ASSETS:
        raise ValidationError(
            f"Official file {csv_path} yields {len(symbols)} unique symbols, "
            f"expected {EXPECTED_ASSETS}."
        )
    return symbols


def load_raw_long(csv_path: Path = RAW_OHLCV,
                  official: list[str] | None = None) -> pd.DataFrame:
    """Load and structurally validate the raw long-format Dataset D file."""
    if not csv_path.is_file():
        raise ValidationError(f"Input not found: {csv_path}")
    official = official or load_official_order()

    df = pd.read_csv(csv_path, parse_dates=["date"])
    required = {"symbol", "eodhd_identifier", "date", *ALL_FIELDS}
    if not required.issubset(df.columns):
        raise ValidationError(f"{csv_path} is missing columns "
                              f"{sorted(required - set(df.columns))}.")
    if df[sorted(required)].isna().any().any():
        raise ValidationError("Raw Dataset D contains missing values.")
    if df.duplicated(["symbol", "date"]).any():
        raise ValidationError("Raw Dataset D contains duplicate symbol/date rows.")
    if sorted(df["symbol"].unique()) != sorted(official):
        raise ValidationError("Raw Dataset D symbols differ from the official "
                              "M6 universe.")
    if (df["date"].dt.dayofweek > 4).any():
        raise ValidationError("Raw Dataset D contains weekend observations.")
    if df["date"].min() != START_DATE or df["date"].max() != END_DATE:
        raise ValidationError(
            f"Raw date span is {df['date'].min().date()}..{df['date'].max().date()}, "
            f"expected {START_DATE.date()}..{END_DATE.date()}."
        )
    for field in PRICE_FIELDS:
        if (df[field] <= 0).any():
            raise ValidationError(f"Non-positive values in raw field {field!r}.")
    if (df["volume"] < 0).any():
        raise ValidationError("Negative raw volume observations.")

    logger.info("Raw Dataset D validated: %d rows, %d assets, %s .. %s",
                df.shape[0], df["symbol"].nunique(),
                df["date"].min().date(), df["date"].max().date())
    return df


# --------------------------------------------------------------------------- #
# Calendar and forward-filling
# --------------------------------------------------------------------------- #

def weekday_calendar(start: pd.Timestamp = START_DATE,
                     end: pd.Timestamp = END_DATE) -> pd.DatetimeIndex:
    """Shared Monday-to-Friday calendar (public holidays retained as dates)."""
    cal = pd.date_range(start, end, freq="B", name="date")
    if len(cal) != EXPECTED_WEEKDAYS:
        raise ValidationError(
            f"Weekday calendar has {len(cal)} dates, expected {EXPECTED_WEEKDAYS}."
        )
    return cal


def build_panels(raw: pd.DataFrame, official: list[str],
                 calendar: pd.DatetimeIndex,
                 ) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Pivot each raw field to date x symbol, reindex onto the shared calendar
    and forward-fill after inception.

    Returns (raw_on_calendar, filled) keyed by field name. Column order is the
    project's official M6 asset order.
    """
    raw_on_calendar: dict[str, pd.DataFrame] = {}
    filled: dict[str, pd.DataFrame] = {}
    for field in ALL_FIELDS:
        wide = raw.pivot(index="date", columns="symbol", values=field)
        wide = wide.reindex(columns=official).sort_index()
        wide = wide.reindex(calendar).astype(float)
        wide.columns.name = None
        raw_on_calendar[field] = wide
        filled[field] = wide.ffill()
    return raw_on_calendar, filled


def daily_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """log_return[t] = log(price[t] / price[t-1]); NaN where either is missing."""
    returns = np.log(prices).diff()
    if np.isinf(returns.to_numpy()).any():
        raise ValidationError("Infinite values found in log returns.")
    return returns


def load_processed_panels(
    out_dir: Path = OUT_DIR,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Read back the processed Dataset D panels and the log-return panel."""
    panels: dict[str, pd.DataFrame] = {}
    for field in ALL_FIELDS:
        path = out_dir / f"dataset_d_{field}_weekday.csv"
        frame = pd.read_csv(path, parse_dates=["date"]).set_index("date")
        frame.index.name = "date"
        panels[field] = frame
    returns = pd.read_csv(out_dir / "dataset_d_daily_log_returns.csv",
                          parse_dates=["date"]).set_index("date")
    returns.index.name = "date"
    return panels, returns


# --------------------------------------------------------------------------- #
# Summary and validation
# --------------------------------------------------------------------------- #

def build_summary(raw_on_calendar: dict[str, pd.DataFrame],
                  filled: dict[str, pd.DataFrame],
                  returns: pd.DataFrame) -> pd.DataFrame:
    """One row per asset (official order)."""
    base_raw = raw_on_calendar[RETURN_BASIS]
    base_filled = filled[RETURN_BASIS]
    raw_vol = raw_on_calendar["volume"]
    filled_vol = filled["volume"]

    rows = []
    for order, symbol in enumerate(base_raw.columns, start=1):
        genuine = base_raw[symbol].dropna()
        first_gen, last_gen = genuine.index.min(), genuine.index.max()
        ret_s = returns[symbol]
        after = base_filled[symbol].loc[first_gen:]
        rows.append({
            "official_order": order,
            "symbol": symbol,
            "first_genuine_date": first_gen.date().isoformat(),
            "last_genuine_date": last_gen.date().isoformat(),
            "genuine_observation_count": int(genuine.shape[0]),
            "leading_missing_weekdays": int((base_raw.index < first_gen).sum()),
            "forward_filled_weekdays": int(after.shape[0] - genuine.shape[0]),
            "first_valid_log_return_date": ret_s.first_valid_index().date().isoformat(),
            "valid_log_return_count": int(ret_s.notna().sum()),
            "missing_log_return_count": int(ret_s.isna().sum()),
            "raw_zero_volume_days": int((raw_vol[symbol] == 0).sum()),
            "processed_zero_volume_days": int((filled_vol[symbol] == 0).sum()),
        })
    return pd.DataFrame(rows)


def validate_outputs(raw_on_calendar: dict[str, pd.DataFrame],
                     filled: dict[str, pd.DataFrame],
                     returns: pd.DataFrame,
                     calendar: pd.DatetimeIndex) -> dict[str, object]:
    facts: dict[str, object] = {}

    if (calendar.dayofweek > 4).any():
        raise ValidationError("Weekend dates present in the shared calendar.")

    for field, frame in filled.items():
        if not frame.index.equals(calendar):
            raise ValidationError(f"{field}: index is not the shared calendar.")
        raw_frame = raw_on_calendar[field]
        for symbol in frame.columns:
            first_gen = raw_frame[symbol].first_valid_index()
            if first_gen is None:
                raise ValidationError(f"{field}/{symbol}: no genuine observation.")
            # Nothing may exist strictly before the first genuine observation.
            before = frame[symbol].loc[:first_gen].iloc[:-1]
            if before.notna().any():
                raise ValidationError(
                    f"{field}/{symbol}: value created before inception "
                    "(backward filling detected)."
                )
            if frame[symbol].loc[first_gen:].isna().any():
                raise ValidationError(f"{field}/{symbol}: missing after inception.")

    # Every asset's inception must be identical across fields (same source rows).
    base = raw_on_calendar[RETURN_BASIS].apply(lambda s: s.first_valid_index())
    for field in ALL_FIELDS:
        series = raw_on_calendar[field].apply(lambda s: s.first_valid_index())
        if not series.equals(base):
            raise ValidationError(f"{field}: inception dates differ from "
                                  f"{RETURN_BASIS}.")

    # Established DRE treatment (Stage 2 convention, reproduced here).
    dre_raw = raw_on_calendar[RETURN_BASIS]["DRE"].dropna()
    if dre_raw.index.max() != DRE_LAST_GENUINE:
        raise ValidationError(
            f"DRE final genuine observation is {dre_raw.index.max().date()}, "
            f"expected {DRE_LAST_GENUINE.date()}."
        )
    dre_after = filled[RETURN_BASIS]["DRE"].loc[DRE_LAST_GENUINE:]
    if not np.allclose(dre_after.to_numpy(), float(dre_raw.iloc[-1])):
        raise ValidationError("DRE price is not constant after 2022-10-03.")
    dre_ret = returns["DRE"].loc[DRE_LAST_GENUINE + pd.Timedelta(days=1):]
    if not (dre_ret == 0.0).all():
        raise ValidationError("DRE has a non-zero return after 2022-10-03.")
    facts["dre_final_price"] = float(dre_raw.iloc[-1])
    facts["dre_forward_filled_weekdays"] = int(dre_after.shape[0] - 1)

    missing_anchors = [d.date().isoformat() for d in M6_FRIDAY_ANCHORS
                       if d not in calendar]
    if missing_anchors:
        raise ValidationError(f"Missing M6 anchor dates: {missing_anchors}.")

    if np.isinf(returns.to_numpy()).any():
        raise ValidationError("Infinite log returns.")

    # The KR provider break must still be present: this stage does not repair it.
    kr = float(returns["KR"].loc[KR_BREAK_DATE])
    facts["kr_2022_10_21_log_return"] = kr
    if kr < 0.15:
        raise ValidationError(
            "KR's 2022-10-21 adjusted-close return is no longer the known "
            "provider anomaly - Dataset D appears to have been altered."
        )

    facts["dates_added_by_reindexing"] = int(
        len(calendar) - raw_on_calendar[RETURN_BASIS].dropna(how="all").shape[0]
    )
    return facts


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def build_report(summary: pd.DataFrame, facts: dict[str, object],
                 raw_on_calendar: dict[str, pd.DataFrame],
                 filled: dict[str, pd.DataFrame], returns: pd.DataFrame,
                 raw_hash_unchanged: bool) -> str:
    zero_vol = summary.loc[summary["raw_zero_volume_days"] > 0,
                           ["symbol", "raw_zero_volume_days",
                            "processed_zero_volume_days"]]
    late = summary.loc[summary["leading_missing_weekdays"] > 0,
                       ["symbol", "first_genuine_date",
                        "leading_missing_weekdays"]]
    total_filled = sum(
        int(filled[f].notna().sum().sum() - raw_on_calendar[f].notna().sum().sum())
        for f in ALL_FIELDS
    )
    lines = [
        "# Dataset D Preprocessing Report (Stage 6A)",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "## 1. Input",
        "",
        "- `Data/raw/dataset_d_eodhd/dataset_d_ohlcv.csv` (SHA-256 unchanged by "
        f"this run: {'CONFIRMED' if raw_hash_unchanged else 'FAILED'})",
        "- Raw Dataset D was neither re-downloaded, repaired, nor modified. The "
        "known KR adjusted-close provider break on 2022-10-21 is deliberately "
        "retained: this run measures its log return as "
        f"{facts['kr_2022_10_21_log_return']:.4f} and fails if it has "
        "disappeared.",
        "",
        "## 2. Shared weekday calendar",
        "",
        f"- {EXPECTED_WEEKDAYS} Monday-to-Friday dates, {START_DATE.date()} .. "
        f"{END_DATE.date()}; weekends excluded, public holidays retained as "
        "calendar dates.",
        "- Dates added by reindexing (weekdays absent from the raw union "
        f"calendar): {facts['dates_added_by_reindexing']}",
        "- All 13 M6 Friday anchors are present.",
        "",
        "## 3. Forward filling",
        "",
        "- Applied per asset per field, only after that asset's first genuine "
        "observation. No backward filling, interpolation or cross-asset "
        "substitution.",
        f"- Total cells forward-filled across all six fields: {total_filled}",
        f"- Assets with genuine later inception ({late.shape[0]}): pre-inception "
        "weekdays remain missing.",
        "",
        "| symbol | first genuine date | leading missing weekdays |",
        "|---|---|---|",
        *[f"| {r.symbol} | {r.first_genuine_date} | {r.leading_missing_weekdays} |"
          for r in late.itertuples(index=False)],
        "",
        f"- DRE: final genuine price {facts['dre_final_price']:.4f} on "
        f"{DRE_LAST_GENUINE.date()}, carried forward over "
        f"{facts['dre_forward_filled_weekdays']} subsequent weekdays; every "
        "later log return is exactly zero (the established project treatment).",
        "",
        "## 4. Returns",
        "",
        "- Basis: ADJUSTED CLOSE. `r[t] = log(P[t] / P[t-1])`.",
        "- Raw close is NOT used for returns because it contains mechanical "
        "split jumps that are not investment returns.",
        f"- Valid log returns: {int(returns.notna().sum().sum())}; missing: "
        f"{int(returns.isna().sum().sum())} (pre-inception plus each asset's "
        "first calendar row).",
        "- No infinities. No clipping, winsorising, ECOD or normalisation.",
        "",
        "## 5. Zero-volume observations (recorded, not modified)",
        "",
        f"- Raw zero-volume asset-days: {int(summary['raw_zero_volume_days'].sum())}; "
        "after calendar alignment and forward filling: "
        f"{int(summary['processed_zero_volume_days'].sum())}.",
        "- Handled locally in feature construction only (see "
        "`scripts/build_feature_baseline_dataset.py`); raw values are untouched.",
        "",
        "| symbol | raw zero-volume days | zero-volume days after filling |",
        "|---|---|---|",
        *[f"| {r.symbol} | {r.raw_zero_volume_days} | "
          f"{r.processed_zero_volume_days} |"
          for r in zero_vol.itertuples(index=False)],
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def write_wide(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame.copy()
    out.insert(0, "date", out.index.strftime("%Y-%m-%d"))
    out.to_csv(path, index=False)
    logger.info("Wrote %s (%d x %d)", path, out.shape[0], out.shape[1] - 1)


def run() -> dict[str, object]:
    raw_hash_before = sha256_of(RAW_OHLCV)
    official = load_official_order()
    raw = load_raw_long(RAW_OHLCV, official)

    calendar = weekday_calendar()
    raw_on_calendar, filled = build_panels(raw, official, calendar)
    returns = daily_log_returns(filled[RETURN_BASIS])

    summary = build_summary(raw_on_calendar, filled, returns)
    facts = validate_outputs(raw_on_calendar, filled, returns, calendar)

    if sha256_of(RAW_OHLCV) != raw_hash_before:
        raise ValidationError("Raw Dataset D changed during processing.")

    for field in ALL_FIELDS:
        write_wide(filled[field], OUT_DIR / f"dataset_d_{field}_weekday.csv")
    write_wide(returns, OUT_DIR / "dataset_d_daily_log_returns.csv")

    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_SUMMARY, index=False)
    logger.info("Wrote %s (%d assets)", OUT_SUMMARY, summary.shape[0])

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text(
        build_report(summary, facts, raw_on_calendar, filled, returns, True),
        encoding="utf-8",
    )
    logger.info("Wrote %s", OUT_REPORT)
    return facts


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        run()
    except ValidationError as exc:
        logger.error("Validation failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
