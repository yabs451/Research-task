"""Dataset D - competition-baseline OHLCV data: raw EODHD acquisition.

Raw daily OHLCV history for the 100 official M6 assets over
2009-01-01 .. 2023-02-03 (inclusive), collected to support the feature-based
competition baselines (LightGBM, Random Forest) adapted from Samartzis.

Why the range starts in 2009 even though modelling begins in 2010: the
reference feature set includes a four-week return shifted by 20 x 11 = 220
business days (``feat_6``), so a training row dated early 2010 needs roughly 240
business days of prior history. The 2009 observations are feature warm-up, not
training or evaluation observations.

This is ACQUISITION ONLY. No calendar alignment, filling, imputation, return
construction, outlier handling, feature engineering, labelling, training or
evaluation happens here.

Shared EODHD infrastructure (client, token loading, secret-safety checks,
subscription verification, identifier resolution, official universe) is imported
from the Dataset C acquisition script rather than duplicated.

Outputs:
    data/raw/dataset_d_eodhd/dataset_d_ohlcv.csv     long format, one row per
                                                     asset-date, raw fields
    data/raw/dataset_d_eodhd/dataset_d_status.csv    per-asset provenance

Resuming: the consolidated CSV is appended per asset and the status file records
what succeeded, so re-running skips assets already retrieved. Use --refresh to
re-download everything.

Usage:
    python scripts/download_dataset_d_eodhd.py [--refresh] [--limit N]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from download_dataset_c_eodhd import (  # noqa: E402
    MAX_MEDIAN_ABS_RETURN_DIFF, MIN_OVERLAP_OBS, MIN_RETURN_CORR,
    DATASET_A, EodhdClient, identifier_candidates, load_token,
    official_asset_order, verify_git_secret_safety, verify_subscription,
)

OUT_DIR = PROJECT_ROOT / "data" / "raw" / "dataset_d_eodhd"
OUT_CSV = OUT_DIR / "dataset_d_ohlcv.csv"
STATUS_CSV = OUT_DIR / "dataset_d_status.csv"

FROM_DATE = "2009-01-01"
TO_DATE = "2023-02-03"          # EODHD `to` is inclusive; M6 evaluation end

# Raw fields kept exactly as EODHD supplies them.
FIELDS = ["open", "high", "low", "close", "adjusted_close", "volume"]

logger = logging.getLogger("download_dataset_d_eodhd")


def fetch_asset(client: EodhdClient, symbol: str) -> tuple[pd.DataFrame, str, list[str]]:
    """Download one asset's raw OHLCV history.

    Returns (frame, identifier_used, identifiers_tried). An empty frame means
    every candidate identifier failed - the asset is recorded as a problem, and
    no substitute security is ever used in its place.
    """
    tried: list[str] = []
    for candidate in identifier_candidates(symbol):
        tried.append(candidate)
        records = client.get(f"eod/{candidate}",
                             **{"from": FROM_DATE, "to": TO_DATE,
                                "period": "d", "order": "a"})
        if isinstance(records, list) and records:
            frame = pd.DataFrame(records)
            missing = [f for f in FIELDS if f not in frame.columns]
            if missing:
                raise RuntimeError(f"{candidate}: response lacks field(s) {missing}")
            frame = frame[["date"] + FIELDS].copy()
            frame["date"] = pd.to_datetime(frame["date"])
            frame.insert(0, "symbol", symbol)
            frame.insert(1, "eodhd_identifier", candidate)
            return frame.sort_values("date").drop_duplicates("date"), candidate, tried
    return pd.DataFrame(), "", tried


def check_identity(symbol: str, frame: pd.DataFrame,
                   dataset_a: pd.DataFrame | None) -> tuple[str, float | None, int]:
    """Sanity-check that the retrieved security is the intended asset.

    Compares adjusted-close daily returns with Dataset A over their overlap.
    Dataset A is a reference only - it never fills or patches Dataset D.
    """
    if dataset_a is None or symbol not in dataset_a.columns:
        return "no Dataset A reference", None, 0
    series = frame.set_index("date")["adjusted_close"].astype(float)
    reference = dataset_a[symbol].dropna()
    common = series.index.intersection(reference.index)
    if len(common) < MIN_OVERLAP_OBS:
        return f"insufficient overlap ({len(common)} obs)", None, len(common)
    joint = pd.concat(
        [series.reindex(common).sort_index().pct_change(),
         reference.reindex(common).sort_index().pct_change()],
        axis=1, join="inner").dropna()
    corr = float(joint.iloc[:, 0].corr(joint.iloc[:, 1]))
    median_abs = float((joint.iloc[:, 0] - joint.iloc[:, 1]).abs().median())
    verdict = ("match" if corr >= MIN_RETURN_CORR
               and median_abs <= MAX_MEDIAN_ABS_RETURN_DIFF
               else "MISMATCH - check identifier")
    return verdict, round(corr, 6), len(common)


def load_progress() -> tuple[pd.DataFrame, set[str]]:
    """Previously retrieved rows and the symbols already completed."""
    if not (OUT_CSV.is_file() and STATUS_CSV.is_file()):
        return pd.DataFrame(), set()
    status = pd.read_csv(STATUS_CSV)
    done = set(status.loc[status["status"] == "ok", "symbol"])
    existing = pd.read_csv(OUT_CSV, parse_dates=["date"])
    existing = existing[existing["symbol"].isin(done)]      # drop partial writes
    return existing, done


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true",
                        help="re-download every asset, ignoring saved progress")
    parser.add_argument("--limit", type=int, default=None,
                        help="only process the first N assets (smoke testing)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    symbols = official_asset_order()
    if len(symbols) != 100:
        raise RuntimeError(f"Expected 100 official M6 assets, found {len(symbols)}")
    if args.limit:
        symbols = symbols[: args.limit]

    verify_git_secret_safety()
    client = EodhdClient(load_token())
    subscription = verify_subscription(client, from_date=FROM_DATE)
    print(f"\nSubscription OK (type={subscription.get('subscription_type')}, "
          f"requests remaining today="
          f"{subscription.get('requests_remaining_today', 'n/a')})")
    print(f"Requesting {FROM_DATE} .. {TO_DATE} for {len(symbols)} assets, "
          f"fields: {', '.join(FIELDS)}\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    existing, done = (pd.DataFrame(), set()) if args.refresh else load_progress()
    if done:
        print(f"Resuming: {len(done)} asset(s) already retrieved.\n")

    dataset_a = (pd.read_csv(DATASET_A, parse_dates=["date"]).set_index("date")
                 if DATASET_A.is_file() else None)

    blocks = [existing] if not existing.empty else []
    status_rows: list[dict] = []
    stopped = False

    for i, symbol in enumerate(symbols, 1):
        if symbol in done:
            block = existing[existing["symbol"] == symbol]
            status_rows.append(summarise(symbol, block, block["eodhd_identifier"].iloc[0],
                                         [], "ok", "", "reused saved download", None, 0))
            continue
        try:
            frame, identifier, tried = fetch_asset(client, symbol)
        except RuntimeError as exc:
            print(f"\nSTOPPED at {symbol}: {exc}")
            print("Assets already retrieved are saved; re-run to resume.")
            stopped = True
            break

        if frame.empty:
            status_rows.append(summarise(symbol, frame, "", tried, "failed",
                                         f"no data from any candidate ({', '.join(tried)})",
                                         "", None, 0))
            print(f"  [{i:3d}/{len(symbols)}] {symbol:<8} FAILED - tried {tried}")
            continue

        verdict, corr, overlap = check_identity(symbol, frame, dataset_a)
        blocks.append(frame)
        status_rows.append(summarise(symbol, frame, identifier, tried, "ok", "",
                                     verdict, corr, overlap))
        print(f"  [{i:3d}/{len(symbols)}] {symbol:<8} {identifier:<10} "
              f"{len(frame):>5} rows  {frame['date'].min().date()} .. "
              f"{frame['date'].max().date()}  {verdict}")
        write_outputs(blocks, status_rows)          # persist after each asset

    write_outputs(blocks, status_rows)
    return report(blocks, status_rows, symbols, stopped)


def summarise(symbol, frame, identifier, tried, status, note, verdict, corr, overlap):
    return {
        "symbol": symbol,
        "eodhd_identifier": identifier,
        "identifiers_tried": " | ".join(tried),
        "status": status,
        "n_observations": 0 if frame.empty else len(frame),
        "first_date": None if frame.empty else str(frame["date"].min().date()),
        "last_date": None if frame.empty else str(frame["date"].max().date()),
        "dataset_a_identity_check": verdict,
        "dataset_a_return_corr": corr,
        "dataset_a_overlap_obs": overlap,
        "note": note,
    }


def write_outputs(blocks: list[pd.DataFrame], status_rows: list[dict]) -> None:
    if status_rows:
        pd.DataFrame(status_rows).to_csv(STATUS_CSV, index=False)
    if blocks:
        table = pd.concat(blocks, ignore_index=True)
        table = table.drop_duplicates(["symbol", "date"]).sort_values(["symbol", "date"])
        table.to_csv(OUT_CSV, index=False, date_format="%Y-%m-%d")


def report(blocks, status_rows, symbols, stopped) -> int:
    status = pd.DataFrame(status_rows)
    ok = status[status["status"] == "ok"]
    failed = status[status["status"] != "ok"]

    if blocks:
        table = pd.concat(blocks, ignore_index=True).drop_duplicates(["symbol", "date"])
        latest, earliest = table["date"].max(), table["date"].min()
        if latest > pd.Timestamp(TO_DATE):
            raise RuntimeError(f"Observation after {TO_DATE}: {latest.date()}")
        if earliest < pd.Timestamp(FROM_DATE):
            raise RuntimeError(f"Observation before {FROM_DATE}: {earliest.date()}")
        print(f"\nDataset D: {OUT_CSV.relative_to(PROJECT_ROOT)}")
        print(f"  {len(table):,} rows, {table['symbol'].nunique()} assets, "
              f"{earliest.date()} .. {latest.date()}")

    print(f"\nRetrieved {len(ok)}/{len(symbols)} assets; {len(failed)} failed.")
    mismatches = list(ok.loc[ok["dataset_a_identity_check"].astype(str)
                             .str.startswith("MISMATCH"), "symbol"])
    if mismatches:
        print(f"  IDENTITY MISMATCHES to investigate: {mismatches}")
    for row in failed.itertuples():
        print(f"  FAILED {row.symbol}: {row.note}")
    print(f"Status file: {STATUS_CSV.relative_to(PROJECT_ROOT)}")
    return 2 if stopped else (0 if failed.empty else 1)


if __name__ == "__main__":
    raise SystemExit(main())
