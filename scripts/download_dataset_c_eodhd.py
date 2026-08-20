"""Dataset C - pre-M6 adaptation data: raw EODHD acquisition.

Downloads daily history for the 100 official M6 assets from EODHD over
2012-03-05 .. 2022-03-04 (inclusive). The end date is deliberately the first M6
forecast origin, so nothing in Dataset C can overlap the competition period.

This is ACQUISITION ONLY. The script does not compute returns, build training
windows or contexts, forward/backward fill, interpolate, standardise, normalise
or clip anything, and it never uses M6-period or Dataset A values to patch a
gap. Assets that listed after 2012-03-05 simply have shorter genuine history.

Outputs (all new, nothing existing is written to):
    data/raw/dataset_c_eodhd/raw_responses/<SYMBOL>.json   preserved responses
    data/raw/dataset_c_eodhd/dataset_c_adjusted_close.csv  date + 100 assets
    data/raw/dataset_c_eodhd/dataset_c_acquisition_manifest.csv  per-asset status

Safety:
  - The EODHD token is loaded from the local env file and is never printed,
    logged, saved or embedded in any recorded URL; all error text is redacted.
  - The subscription and the historical EOD endpoint are verified before any
    bulk download.
  - Each asset's raw response is saved as soon as it arrives, so a later failure
    never costs a completed download; re-running resumes and skips what is
    already stored (use --refresh to force re-download).

Usage:
    python scripts/download_dataset_c_eodhd.py [--refresh] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ENV_FILES = (PROJECT_ROOT / "proj.env", PROJECT_ROOT / ".env")
TOKEN_ENV_NAMES = ("EODHD_API_TOKEN", "EODHD_API_KEY")

OUT_DIR = PROJECT_ROOT / "data" / "raw" / "dataset_c_eodhd"
RAW_DIR = OUT_DIR / "raw_responses"
OUT_CSV = OUT_DIR / "dataset_c_adjusted_close.csv"
MANIFEST_CSV = OUT_DIR / "dataset_c_acquisition_manifest.csv"

# Read-only references.
OFFICIAL_ORDER_SOURCE = (PROJECT_ROOT / "data" / "processed" / "rolling_origins"
                         / "round_01_context.csv")
DATASET_A = PROJECT_ROOT / "data" / "raw" / "yahoo" / "dataset_a_adjusted_close_repaired.csv"

API_BASE = "https://eodhd.com/api"
FROM_DATE = "2012-03-05"
TO_DATE = "2022-03-04"          # EODHD `to` is inclusive; first M6 forecast origin

MIN_REMAINING_REQUESTS = 120    # 100 assets + verification headroom

# Transient-network resilience for a ~100-request job.
NETWORK_RETRIES = 4
RETRY_BACKOFF_SECONDS = 15

# Overlap validation against Dataset A (a sanity check that we fetched the same
# security; Dataset A is NEVER used to fill or patch Dataset C values).
MIN_OVERLAP_OBS = 60
MIN_RETURN_CORR = 0.98
MAX_MEDIAN_ABS_RETURN_DIFF = 0.003

# Identifiers validated by the Stage 1B repair work: Everest Re trades on EODHD
# under its post-rename code, so the canonical M6 ticker does NOT map directly.
VALIDATED_IDENTIFIERS = {
    "DRE": "DRE.US",
    "RE": "EG.US",
    "WRK": "WRK.US",
}

logger = logging.getLogger("download_dataset_c_eodhd")


def identifier_candidates(symbol: str) -> list[str]:
    """EODHD identifiers to try for a canonical M6 ticker, best first."""
    if symbol in VALIDATED_IDENTIFIERS:
        primary = VALIDATED_IDENTIFIERS[symbol]
        return [primary] + [c for c in (f"{symbol}.US",) if c != primary]
    if symbol.endswith(".L"):                 # London-listed ETFs
        stem = symbol[:-2]
        return [f"{stem}.LSE", f"{stem}.L", f"{stem}.XETRA"]
    return [f"{symbol}.US"]


@dataclass
class AssetResult:
    symbol: str
    identifier: str = ""
    tried: list[str] = field(default_factory=list)
    status: str = "pending"        # ok | failed | short-history
    n_obs: int = 0
    first_date: Optional[str] = None
    last_date: Optional[str] = None
    overlap_obs: int = 0
    overlap_return_corr: Optional[float] = None
    overlap_median_abs_diff: Optional[float] = None
    overlap_verdict: str = "not checked"
    note: str = ""
    from_cache: bool = False


class EodhdClient:
    """Thin EODHD client that keeps the token out of logs and errors."""

    def __init__(self, token: str) -> None:
        self._token = token
        self._session = requests.Session()

    def _redact(self, text: str) -> str:
        return text.replace(self._token, "***REDACTED***")

    def get(self, path: str, **params: Any) -> Any:
        params = {**params, "api_token": self._token, "fmt": "json"}
        url = f"{API_BASE}/{path}"
        logger.info("GET /%s", path)
        # Transient network errors are retried a few times; a persistent failure
        # still raises so the caller can stop safely with progress preserved.
        last_error: Optional[str] = None
        for attempt in range(1, NETWORK_RETRIES + 1):
            try:
                resp = self._session.get(url, params=params, timeout=60)
                break
            except requests.RequestException as exc:
                last_error = self._redact(str(exc))
                if attempt < NETWORK_RETRIES:
                    wait = RETRY_BACKOFF_SECONDS * attempt
                    logger.info("  network error on /%s (attempt %d/%d); "
                                "retrying in %ds", path, attempt, NETWORK_RETRIES, wait)
                    time.sleep(wait)
        else:
            raise RuntimeError(
                f"Request to /{path} failed after {NETWORK_RETRIES} attempts: "
                f"{last_error}"
            ) from None
        if resp.status_code == 404:
            return None
        if resp.status_code == 429:
            raise RuntimeError(
                f"EODHD rate limit hit on /{path}; stopping so completed "
                "downloads are preserved. Re-run later to resume."
            )
        if resp.status_code != 200:
            raise RuntimeError(
                f"EODHD returned HTTP {resp.status_code} for /{path}: "
                f"{self._redact(resp.text[:300])}"
            )
        try:
            return resp.json()
        except ValueError:
            raise RuntimeError(
                f"EODHD returned non-JSON for /{path}: "
                f"{self._redact(resp.text[:200])}"
            ) from None


def load_token() -> str:
    for env_file in ENV_FILES:
        if env_file.is_file():
            load_dotenv(env_file, override=False)
    for name in TOKEN_ENV_NAMES:
        token = os.getenv(name)
        if token and token.strip():
            logger.info("Loaded EODHD token from %s (value not shown).", name)
            return token.strip()
    raise RuntimeError(
        "EODHD API token not found. Define EODHD_API_TOKEN (or EODHD_API_KEY) "
        f"in one of: {', '.join(str(p) for p in ENV_FILES)}. "
        "The token value is never printed."
    )


def verify_git_secret_safety() -> None:
    """Fail if an env file is tracked by git; require .gitignore coverage."""
    import subprocess

    tracked = subprocess.run(["git", "ls-files"], cwd=PROJECT_ROOT,
                             capture_output=True, text=True).stdout.splitlines()
    for env_file in ENV_FILES:
        if env_file.name in tracked:
            raise RuntimeError(
                f"SECURITY: {env_file.name} is tracked by git. Remove it from "
                "the index and rotate the token before continuing."
            )
    gitignore = PROJECT_ROOT / ".gitignore"
    patterns = (gitignore.read_text(encoding="utf-8").splitlines()
                if gitignore.is_file() else [])
    for env_file in ENV_FILES:
        if env_file.is_file() and not any(
            p.strip() in (env_file.name, "*.env", ".env") for p in patterns
        ):
            raise RuntimeError(
                f"SECURITY: {env_file.name} exists but is not covered by "
                ".gitignore. Add it before running."
            )
    logger.info("Secret-safety checks passed: env files untracked and ignored.")


def verify_subscription(client: EodhdClient, from_date: str = FROM_DATE,
                        probe_to: Optional[str] = None) -> dict[str, Any]:
    """Confirm the token works and the historical EOD endpoint is available.

    ``from_date``/``probe_to`` default to this dataset's own range; a companion
    acquisition (Dataset D) passes its earlier start date so the probe checks
    that history actually reaches that far back.
    """
    probe_to = probe_to or (pd.Timestamp(from_date) + pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    user = client.get("user")
    if not isinstance(user, dict):
        raise RuntimeError("Unexpected /user response; cannot verify subscription.")
    safe = {
        "token_valid": True,
        "subscription_type": user.get("subscriptionType"),
        "daily_rate_limit": user.get("dailyRateLimit"),
        "api_requests_used_today": user.get("apiRequests"),
    }
    limit, used = safe["daily_rate_limit"], safe["api_requests_used_today"]
    if isinstance(limit, (int, float)) and isinstance(used, (int, float)):
        remaining = limit - used
        safe["requests_remaining_today"] = remaining
        if remaining < MIN_REMAINING_REQUESTS:
            raise RuntimeError(
                f"Only {remaining} EODHD requests remain today; at least "
                f"{MIN_REMAINING_REQUESTS} are needed for the 100 assets. "
                "Re-run when the quota resets; completed downloads are kept."
            )

    # Probe the deep-history endpoint at the actual start of the requested range.
    probe = client.get("eod/AAPL.US", **{"from": from_date, "to": probe_to,
                                         "period": "d", "order": "a"})
    if not probe or not isinstance(probe, list):
        raise RuntimeError(
            "The EOD endpoint returned no data for a known symbol at "
            f"{from_date}; the subscription may not include history that far "
            "back. Nothing was downloaded."
        )
    if "adjusted_close" not in probe[0]:
        raise RuntimeError("EOD probe did not return adjusted_close; the "
                           "subscription may lack historical EOD access.")
    safe["eod_endpoint_accessible"] = True
    safe["eod_history_reaches_from_date"] = True
    logger.info("Subscription verified: type=%s, daily limit=%s, used today=%s",
                safe["subscription_type"], limit, used)
    return safe


def official_asset_order() -> list[str]:
    return list(pd.read_csv(OFFICIAL_ORDER_SOURCE, nrows=0).columns[1:])


def raw_path(symbol: str) -> Path:
    return RAW_DIR / f"{symbol.replace('.', '_')}.json"


def save_raw(symbol: str, identifier: str, records: list[dict]) -> None:
    """Preserve the raw response verbatim, with token-free provenance."""
    payload = {
        "canonical_m6_symbol": symbol,
        "eodhd_identifier": identifier,
        "endpoint": f"{API_BASE}/eod/{identifier}",
        "parameters": {"from": FROM_DATE, "to": TO_DATE, "period": "d",
                       "order": "a", "fmt": "json"},
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "n_records": len(records),
        "records": records,
    }
    tmp = raw_path(symbol).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    tmp.replace(raw_path(symbol))       # atomic: a crash cannot truncate a file


def load_cached(symbol: str) -> Optional[dict]:
    path = raw_path(symbol)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None
    params = payload.get("parameters", {})
    if params.get("from") != FROM_DATE or params.get("to") != TO_DATE:
        return None                      # different range: re-download
    return payload


def fetch_asset(client: EodhdClient, symbol: str, refresh: bool) -> AssetResult:
    """Download (or reuse) one asset's daily history."""
    result = AssetResult(symbol=symbol)

    if not refresh:
        cached = load_cached(symbol)
        if cached is not None:
            result.identifier = cached["eodhd_identifier"]
            result.from_cache = True
            records = cached["records"]
            _summarise(result, records)
            return result

    for candidate in identifier_candidates(symbol):
        result.tried.append(candidate)
        records = client.get(f"eod/{candidate}",
                             **{"from": FROM_DATE, "to": TO_DATE,
                                "period": "d", "order": "a"})
        if isinstance(records, list) and records:
            result.identifier = candidate
            save_raw(symbol, candidate, records)
            _summarise(result, records)
            return result

    result.status = "failed"
    result.note = f"no data from any candidate identifier ({', '.join(result.tried)})"
    return result


def _summarise(result: AssetResult, records: list[dict]) -> None:
    frame = records_to_frame(records)
    result.n_obs = int(frame["adjusted_close"].notna().sum())
    if frame.empty:
        result.status = "failed"
        result.note = "empty response"
        return
    result.first_date = str(frame["date"].min().date())
    result.last_date = str(frame["date"].max().date())
    result.status = "ok"


def records_to_frame(records: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    if frame.empty:
        return pd.DataFrame(columns=["date", "adjusted_close"])
    if "adjusted_close" not in frame.columns:
        raise RuntimeError("EODHD response has no adjusted_close field")
    frame = frame[["date", "adjusted_close"]].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["adjusted_close"] = pd.to_numeric(frame["adjusted_close"], errors="coerce")
    return frame.sort_values("date").drop_duplicates("date")


def validate_against_dataset_a(result: AssetResult, series: pd.Series,
                               dataset_a: Optional[pd.DataFrame]) -> None:
    """Confirm we fetched the same security. Never patches Dataset C values."""
    if dataset_a is None or result.symbol not in dataset_a.columns:
        result.overlap_verdict = "no Dataset A reference"
        return
    reference = dataset_a[result.symbol].dropna()
    common = series.dropna().index.intersection(reference.index)
    result.overlap_obs = len(common)
    if len(common) < MIN_OVERLAP_OBS:
        result.overlap_verdict = f"insufficient overlap ({len(common)} obs)"
        return
    # Returns are computed here only to compare two price series; nothing
    # derived is stored in Dataset C.
    c_ret = series.reindex(common).sort_index().pct_change().dropna()
    a_ret = reference.reindex(common).sort_index().pct_change().dropna()
    joint = pd.concat([c_ret, a_ret], axis=1, join="inner").dropna()
    if len(joint) < MIN_OVERLAP_OBS:
        result.overlap_verdict = f"insufficient overlap ({len(joint)} return obs)"
        return
    corr = float(joint.iloc[:, 0].corr(joint.iloc[:, 1]))
    median_abs = float((joint.iloc[:, 0] - joint.iloc[:, 1]).abs().median())
    result.overlap_return_corr = round(corr, 6)
    result.overlap_median_abs_diff = round(median_abs, 6)
    result.overlap_verdict = (
        "match" if corr >= MIN_RETURN_CORR and median_abs <= MAX_MEDIAN_ABS_RETURN_DIFF
        else "MISMATCH - check identifier"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true",
                        help="re-download even if a stored response exists")
    parser.add_argument("--limit", type=int, default=None,
                        help="only process the first N assets (smoke testing)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        stream=sys.stdout)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    symbols = official_asset_order()
    if len(symbols) != 100:
        raise RuntimeError(f"Expected 100 official assets, found {len(symbols)}")
    if args.limit:
        symbols = symbols[: args.limit]

    verify_git_secret_safety()
    client = EodhdClient(load_token())
    subscription = verify_subscription(client)
    print(f"\nSubscription OK (type={subscription.get('subscription_type')}, "
          f"requests remaining today={subscription.get('requests_remaining_today', 'n/a')})")
    print(f"Requesting {FROM_DATE} .. {TO_DATE} for {len(symbols)} assets\n")

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    dataset_a = None
    if DATASET_A.is_file():
        dataset_a = pd.read_csv(DATASET_A, parse_dates=["date"]).set_index("date")

    results: list[AssetResult] = []
    series_by_symbol: dict[str, pd.Series] = {}
    try:
        for i, symbol in enumerate(symbols, 1):
            result = fetch_asset(client, symbol, args.refresh)
            if result.status == "ok":
                payload = load_cached(symbol)
                frame = records_to_frame(payload["records"])
                series = frame.set_index("date")["adjusted_close"]
                series_by_symbol[symbol] = series
                validate_against_dataset_a(result, series, dataset_a)
            results.append(result)
            flag = "cache" if result.from_cache else "fetched"
            print(f"  [{i:3d}/{len(symbols)}] {symbol:<8} {result.identifier or '-':<10} "
                  f"{result.status:<7} {result.n_obs:>5} obs  "
                  f"{result.first_date or '-'} .. {result.last_date or '-'}  "
                  f"{result.overlap_verdict:<24} ({flag})")
    except RuntimeError as exc:
        print(f"\nSTOPPED: {exc}")
        print(f"Completed downloads are preserved in {RAW_DIR}; re-run to resume.")
        write_manifest(results)
        return 2

    return finalise(results, series_by_symbol, symbols)


def write_manifest(results: list[AssetResult]) -> None:
    if not results:
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "symbol": r.symbol,
        "eodhd_identifier": r.identifier,
        "identifiers_tried": " | ".join(r.tried),
        "status": r.status,
        "n_observations": r.n_obs,
        "first_date": r.first_date,
        "last_date": r.last_date,
        "dataset_a_overlap_obs": r.overlap_obs,
        "dataset_a_return_corr": r.overlap_return_corr,
        "dataset_a_median_abs_return_diff": r.overlap_median_abs_diff,
        "dataset_a_verdict": r.overlap_verdict,
        "note": r.note,
    } for r in results]).to_csv(MANIFEST_CSV, index=False)


def finalise(results: list[AssetResult], series_by_symbol: dict[str, pd.Series],
             symbols: list[str]) -> int:
    write_manifest(results)

    ok = [r for r in results if r.status == "ok"]
    failed = [r for r in results if r.status != "ok"]

    # --- consolidated raw CSV: union of dates, official column order --------
    if series_by_symbol:
        table = pd.concat(series_by_symbol, axis=1)
        table.columns = list(series_by_symbol)
        table = table.reindex(columns=symbols)      # official order; gaps stay NaN
        table = table.sort_index()
        table.index.name = "date"

        latest = table.index.max()
        if latest > pd.Timestamp(TO_DATE):
            raise RuntimeError(f"Observation after {TO_DATE} found ({latest.date()})")
        earliest = table.index.min()
        if earliest < pd.Timestamp(FROM_DATE):
            raise RuntimeError(f"Observation before {FROM_DATE} found ({earliest.date()})")

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        table.to_csv(OUT_CSV, date_format="%Y-%m-%d")

        print(f"\nConsolidated CSV: {OUT_CSV.relative_to(PROJECT_ROOT)}")
        print(f"  {table.shape[0]} dates x {table.shape[1]} assets, "
              f"{earliest.date()} .. {latest.date()}")

    print(f"\nRetrieved {len(ok)}/{len(symbols)} assets; {len(failed)} failed.")
    mismatches = [r.symbol for r in ok if r.overlap_verdict.startswith("MISMATCH")]
    if mismatches:
        print(f"  IDENTIFIER MISMATCHES to investigate: {mismatches}")
    for r in failed:
        print(f"  FAILED {r.symbol}: {r.note}")
    print(f"Manifest: {MANIFEST_CSV.relative_to(PROJECT_ROOT)}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
