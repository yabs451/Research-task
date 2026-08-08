"""Stage 5 - M6 quintile post-processing and RPS evaluation.

Converts the RAW sampled log-return trajectories produced by the inference stage
into M6-format quintile probability forecasts, builds the realised M6 ground
truth independently from the official price file, and scores both with the
Ranked Probability Score (RPS).

The same functions evaluate every model: models are DISCOVERED from
``Results/*/round_outputs/*_round??_samples.npz`` rather than hard-coded.

Methodology follows the official M6 evaluator (``RPS Reference/RPS and IR
calculation.py`` from Mcompetitions/M6-methods). Only its RPS-related logic is
used: the information ratio is deliberately not implemented (see README of this
module's report output). Two documented departures from that script are made and
are explained in ``rank_to_quintiles``.

The raw NPZ files are read-only research artifacts: this script hashes them
before and after the run and fails if any byte changed.

Run:  python scripts/evaluate_m6_rps.py
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = REPO_ROOT / "Results"
EVAL_DIR = RESULTS_DIR / "Evaluation"
DERIVED_DIR = EVAL_DIR / "derived_dre_adjusted"

PRICE_PATH = REPO_ROOT / "Data" / "assets_m6.csv"
SCHEDULE_PATH = REPO_ROOT / "Data" / "metadata" / "m6_round_schedule.csv"
OFFICIAL_RPS_REFERENCE = REPO_ROOT / "RPS Reference" / "RPS and IR calculation.py"

N_QUINTILES = 5
EXPECTED_ROUNDS = list(range(1, 13))
EXPECTED_ASSETS = 100
EXPECTED_SAMPLES = 100
EXPECTED_HORIZON = 20

NAIVE_LABEL = "Naive equal-probability benchmark"
DRE_ACQUISITION_LAST_TRADING_DAY = "2022-10-03"
DRE_ADJUSTABLE_ROUNDS = (9, 10, 11, 12)  # forecast origin already post-acquisition

RANK_COLUMNS = [f"Rank{i}" for i in range(1, N_QUINTILES + 1)]


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------
@dataclass
class ModelArtifacts:
    """One model's raw inference outputs, as found on disk."""

    key: str                 # directory name, e.g. Chronos_T5_Base_200M
    label: str               # human label used in tables/reports
    directory: Path          # <model>/round_outputs
    reports_dir: Path | None  # <model>/reports if it already exists
    files: dict[int, Path]   # round number -> NPZ path


def discover_models(results_dir: Path = RESULTS_DIR) -> list[ModelArtifacts]:
    """Find every model folder that contains per-round sample NPZ files."""
    pattern = re.compile(r"round(\d{2})_samples\.npz$", re.IGNORECASE)
    models: list[ModelArtifacts] = []

    for round_outputs in sorted(results_dir.glob("*/round_outputs")):
        files: dict[int, Path] = {}
        for path in sorted(round_outputs.glob("*.npz")):
            match = pattern.search(path.name)
            if match:
                files[int(match.group(1))] = path
        if not files:
            continue
        model_dir = round_outputs.parent
        reports_dir = model_dir / "reports"
        models.append(
            ModelArtifacts(
                key=model_dir.name,
                label=model_dir.name.replace("_", " "),
                directory=round_outputs,
                reports_dir=reports_dir if reports_dir.is_dir() else None,
                files=dict(sorted(files.items())),
            )
        )
    return models


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Ranking / quintile construction (official M6 tie principle)
# ---------------------------------------------------------------------------
def rank_to_quintiles(values: np.ndarray, n_quintiles: int = N_QUINTILES) -> np.ndarray:
    """Cross-sectional rank -> fractional quintile membership, tie-aware.

    Implements the official M6 tie principle: assets are ranked from lowest to
    highest return with ``rank(method="min")``; a group of k tied assets
    occupies k consecutive rank slots, and each tied asset receives the AVERAGE
    quintile membership of those slots. An untied asset therefore gets a
    one-hot row, and a tied group that sits inside one quintile also gets a
    one-hot row - both identical to the official script.

    Two documented departures from ``RPS and IR calculation.py``:

    1. Quintile boundaries are computed from the number of ranked assets
       (100 here). The official script uses ``total_ranks = max(min-rank)``,
       which shrinks below 100 whenever ties exist and would shift every
       boundary. The two agree exactly when there are no ties.
    2. The official script's if/elif chain assigns membership only for the
       FIRST quintile a tied block touches, so a block straddling a quintile
       boundary yields a row summing to less than 1. Here membership is spread
       over every quintile the block spans, so all rows sum to 1 - which is
       what the "Handle Ties" averaging clearly intends and what the M6
       probability format requires.

    Returns an (n_assets, n_quintiles) array of memberships summing to 1 per row.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("rank_to_quintiles expects a 1-D array of returns")
    n = values.size
    if np.isnan(values).any():
        raise ValueError("rank_to_quintiles received NaN returns")

    positions = pd.Series(values).rank(method="min").to_numpy()

    # Official slot boundaries: quintile q covers slots int(n*(q-1)/5)+1 .. int(n*q/5)
    edges = [int(n * q / n_quintiles) for q in range(n_quintiles + 1)]
    slot_quintile = np.empty(n + 1, dtype=int)  # 1-based slots
    for q in range(n_quintiles):
        slot_quintile[edges[q] + 1: edges[q + 1] + 1] = q

    membership = np.zeros((n, n_quintiles), dtype=float)
    for position in np.unique(positions):
        tied = np.flatnonzero(positions == position)
        slots = np.arange(int(position), int(position) + tied.size)
        counts = np.bincount(slot_quintile[slots], minlength=n_quintiles) / tied.size
        membership[tied] = counts
    return membership


def official_reference_quintiles(values: np.ndarray) -> np.ndarray:
    """Faithful port of the official script's tie block, used only to cross-check.

    Reproduces ``RPS and IR calculation.py`` exactly, including the
    ``total_ranks = max(min-rank)`` definition and the if/elif chain. Rows may
    therefore sum to less than 1 when a tied block straddles a boundary. Never
    used to produce results - only to measure agreement with
    :func:`rank_to_quintiles`.
    """
    values = np.asarray(values, dtype=float)
    n = values.size
    positions = pd.Series(values).rank(method="min").to_numpy()
    unique_positions = np.unique(positions)
    total_ranks = unique_positions[-1]

    ranges = []
    for q in range(N_QUINTILES):
        low = int(0.2 * q * total_ranks + 1) if q else 1
        high = int(0.2 * (q + 1) * total_ranks + 1)
        ranges.append(set(range(low, high)))

    membership = np.zeros((n, N_QUINTILES), dtype=float)
    for position in unique_positions:
        tied = np.flatnonzero(positions == position)
        slots = list(range(int(position), int(position) + tied.size))
        block = np.zeros((len(slots), N_QUINTILES), dtype=float)
        for q, allowed in enumerate(ranges):
            hits = [i for i, slot in enumerate(slots) if slot in allowed]
            if hits:                      # elif semantics: first hit wins, then stop
                block[hits, q] = 1.0
                break
        membership[tied] = block.mean(axis=0)
    return membership


# ---------------------------------------------------------------------------
# Ground truth from the official M6 price file
# ---------------------------------------------------------------------------
def load_prices(path: Path = PRICE_PATH) -> pd.DataFrame:
    """Load assets_m6.csv into a date x symbol price matrix (no filling yet)."""
    raw = pd.read_csv(path)
    expected = {"symbol", "date", "price"}
    if not expected.issubset(raw.columns):
        raise ValueError(f"{path} must contain columns {sorted(expected)}")
    raw["date"] = pd.to_datetime(raw["date"], format="%Y/%m/%d")
    matrix = raw.pivot(index="date", columns="symbol", values="price").sort_index()
    return matrix


def price_on_or_before(prices: pd.DataFrame, date: str) -> pd.Series:
    """Most recent available price at or before ``date``, per the official rule.

    The official evaluator inserts, for any asset missing on a date, "the most
    recent available price" for that asset. This is the vectorised equivalent,
    carried out over the full price history so that an asset which stops
    trading mid-competition (DRE) keeps its last traded price rather than
    disappearing. No interpolation of any other kind is applied.
    """
    window = prices.loc[prices.index <= pd.Timestamp(date)]
    if window.empty:
        raise ValueError(f"No prices at or before {date}")
    return window.ffill().iloc[-1]


def build_ground_truth(prices: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Realised four-week returns and official quintile targets, per round."""
    rows = []
    for record in schedule.itertuples():
        start, end = record.origin_date, record.forecast_end_date
        open_price = price_on_or_before(prices, start)
        close_price = price_on_or_before(prices, end)
        returns = (close_price - open_price) / open_price
        returns = returns.sort_index()
        if returns.isna().any():
            missing = sorted(returns.index[returns.isna()])
            raise ValueError(f"Round {record.round}: missing realised returns for {missing}")

        membership = rank_to_quintiles(returns.to_numpy())
        official = official_reference_quintiles(returns.to_numpy())
        positions = pd.Series(returns.to_numpy()).rank(method="min").to_numpy()
        n_tied = int(len(positions) - len(np.unique(positions)))

        for i, symbol in enumerate(returns.index):
            rows.append({
                "round": int(record.round),
                "symbol": symbol,
                "start_date": start,
                "end_date": end,
                "actual_return": float(returns.iloc[i]),
                **{col: float(membership[i, q]) for q, col in enumerate(RANK_COLUMNS)},
                "tied_assets_in_round": n_tied,
                "matches_official_tie_port": bool(
                    np.allclose(membership[i], official[i], atol=1e-12)
                ),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Forecast post-processing
# ---------------------------------------------------------------------------
def sampled_four_week_returns(samples: np.ndarray) -> np.ndarray:
    """(assets, samples, 20) daily log returns -> (assets, samples) simple returns.

    L = sum of the 20 predicted daily log returns; R = exp(L) - 1.
    """
    if samples.ndim != 3:
        raise ValueError(f"Expected a 3-D sample array, got shape {samples.shape}")
    log_returns = samples.astype(np.float64).sum(axis=2)
    return np.expm1(log_returns)


def quintile_probabilities(samples: np.ndarray) -> np.ndarray:
    """(assets, samples, 20) raw samples -> (assets, 5) M6 quintile probabilities.

    Each of the sampled futures is ranked CROSS-SECTIONALLY across all assets
    and converted to quintile membership with the same tie-aware function used
    for the realised outcome; the probabilities are the average membership over
    the sampled futures.
    """
    four_week = sampled_four_week_returns(samples)
    n_assets, n_samples = four_week.shape
    totals = np.zeros((n_assets, N_QUINTILES), dtype=float)
    for s in range(n_samples):
        totals += rank_to_quintiles(four_week[:, s])
    return totals / n_samples


# ---------------------------------------------------------------------------
# RPS (official M6 definition)
# ---------------------------------------------------------------------------
def rps_scores(actual: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    """Per-asset RPS: mean squared error of the cumulative quintile vectors."""
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    if actual.shape != forecast.shape:
        raise ValueError(f"Shape mismatch: actual {actual.shape} vs forecast {forecast.shape}")
    diff = np.cumsum(actual, axis=-1) - np.cumsum(forecast, axis=-1)
    return np.mean(diff ** 2, axis=-1)


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------
def load_round_samples(path: Path) -> tuple[np.ndarray, list[str], list[str]]:
    with np.load(path, allow_pickle=False) as data:
        missing = {"forecast_samples", "asset_symbols", "forecast_dates"} - set(data.files)
        if missing:
            raise ValueError(f"{path.name} is missing array(s): {sorted(missing)}")
        samples = data["forecast_samples"]
        symbols = [str(s) for s in data["asset_symbols"]]
        dates = [str(d) for d in data["forecast_dates"]]
    return samples, symbols, dates


def evaluate_model(
    model: ModelArtifacts,
    ground_truth: pd.DataFrame,
    schedule: pd.DataFrame,
    checks: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (probabilities, per-asset RPS, DRE inspection, dispersion) frames."""
    prob_rows, rps_rows, dre_rows, diag_rows = [], [], [], []
    schedule_by_round = schedule.set_index("round")

    for round_number in EXPECTED_ROUNDS:
        if round_number not in model.files:
            raise FileNotFoundError(f"{model.key}: no NPZ for round {round_number}")
        path = model.files[round_number]
        samples, symbols, dates = load_round_samples(path)

        # --- artifact-level checks ---------------------------------------
        if samples.shape != (EXPECTED_ASSETS, EXPECTED_SAMPLES, EXPECTED_HORIZON):
            raise ValueError(f"{path.name}: shape {samples.shape} != "
                             f"({EXPECTED_ASSETS}, {EXPECTED_SAMPLES}, {EXPECTED_HORIZON})")
        if not np.isfinite(samples).all():
            raise ValueError(f"{path.name}: contains non-finite forecast values")
        expected = schedule_by_round.loc[round_number]
        if dates[0] != expected.forecast_start_date or dates[-1] != expected.forecast_end_date:
            raise ValueError(
                f"{path.name}: forecast dates {dates[0]}..{dates[-1]} do not match round "
                f"{round_number} ({expected.forecast_start_date}..{expected.forecast_end_date})"
            )

        truth_round = ground_truth[ground_truth["round"] == round_number]
        if sorted(symbols) != sorted(truth_round["symbol"]):
            raise ValueError(f"{path.name}: asset universe differs from the ground truth")

        # --- forecast post-processing ------------------------------------
        probabilities = quintile_probabilities(samples)
        row_sums = probabilities.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-9):
            raise ValueError(f"{path.name}: quintile probabilities do not sum to 1")
        if probabilities.min() < -1e-12 or probabilities.max() > 1 + 1e-12:
            raise ValueError(f"{path.name}: quintile probabilities outside [0, 1]")

        # Align the ground truth to the NPZ's own asset ordering.
        truth_aligned = truth_round.set_index("symbol").loc[symbols]
        actual = truth_aligned[RANK_COLUMNS].to_numpy(dtype=float)
        scores = rps_scores(actual, probabilities)

        for i, symbol in enumerate(symbols):
            prob_rows.append({
                "model": model.label, "round": round_number, "symbol": symbol,
                **{col: float(probabilities[i, q]) for q, col in enumerate(RANK_COLUMNS)},
            })
            rps_rows.append({
                "model": model.label, "round": round_number,
                "symbol": symbol, "RPS": float(scores[i]),
            })

        # --- DRE inspection (raw values, never modified) -------------------
        dre_index = symbols.index("DRE")
        dre_samples = samples[dre_index].astype(np.float64)
        dre_four_week = np.expm1(dre_samples.sum(axis=1))
        dre_rows.append({
            "model": model.label,
            "round": round_number,
            "npz_file": path.name,
            "forecast_start": dates[0],
            "forecast_end": dates[-1],
            "origin_after_acquisition": dates[0] > DRE_ACQUISITION_LAST_TRADING_DAY,
            "all_zero": bool(np.all(dre_samples == 0.0)),
            "n_exact_zero_values": int((dre_samples == 0.0).sum()),
            "n_values": int(dre_samples.size),
            "min_daily_log_return": float(dre_samples.min()),
            "max_daily_log_return": float(dre_samples.max()),
            "mean_daily_log_return": float(dre_samples.mean()),
            "min_four_week_return": float(dre_four_week.min()),
            "max_four_week_return": float(dre_four_week.max()),
            "median_four_week_return": float(np.median(dre_four_week)),
        })

        # --- forecast dispersion diagnostic --------------------------------
        # Explains how confident the resulting probabilities are: when the
        # spread of the assets' median predicted returns is large relative to
        # the spread within each asset's own samples, the cross-sectional
        # ordering barely changes between samples and the probabilities become
        # near one-hot. Diagnostic only - it feeds no result.
        four_week = sampled_four_week_returns(samples)
        cross_asset = float(np.std(np.median(four_week, axis=1)))
        within_asset = float(np.mean(np.std(four_week, axis=1)))
        safe = np.where(probabilities > 0, probabilities, 1.0)
        diag_rows.append({
            "model": model.label,
            "round": round_number,
            "mean_max_probability": float(probabilities.max(axis=1).mean()),
            "mean_probability_entropy": float((-(safe * np.log(safe)).sum(axis=1)).mean()),
            "max_possible_entropy": float(np.log(N_QUINTILES)),
            "cross_asset_median_spread": cross_asset,
            "within_asset_sample_spread": within_asset,
            "dispersion_ratio": cross_asset / within_asset if within_asset else np.nan,
        })

        checks.append(
            f"{model.key} round {round_number:02d}: shape OK, finite, dates match schedule, "
            f"universe matches ground truth, probabilities sum to 1 "
            f"(max |sum-1| = {np.max(np.abs(row_sums - 1)):.2e})"
        )

    return (pd.DataFrame(prob_rows), pd.DataFrame(rps_rows),
            pd.DataFrame(dre_rows), pd.DataFrame(diag_rows))


def naive_benchmark_rps(ground_truth: pd.DataFrame) -> pd.DataFrame:
    """Score the flat [0.2]*5 forecast through the same evaluator."""
    rows = []
    flat = np.full(N_QUINTILES, 1.0 / N_QUINTILES)
    for record in ground_truth.itertuples():
        actual = np.array([getattr(record, col) for col in RANK_COLUMNS], dtype=float)
        rows.append({
            "model": NAIVE_LABEL, "round": record.round,
            "symbol": record.symbol, "RPS": float(rps_scores(actual, flat)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# DRE-adjusted derived copies (never overwrite the raw NPZ files)
# ---------------------------------------------------------------------------
def write_dre_adjusted_copies(model: ModelArtifacts, dre_frame: pd.DataFrame) -> list[Path]:
    """Write clearly labelled DRE-zeroed reference copies for rounds 9-12.

    These are derived artifacts for later sensitivity work only. They are never
    used for the primary RPS results in this stage, and the raw NPZ files are
    opened read-only.
    """
    written: list[Path] = []
    subset = dre_frame[dre_frame["round"].isin(DRE_ADJUSTABLE_ROUNDS)]
    if subset.empty or bool(subset["all_zero"].all()):
        return written                      # already zero - no copies needed

    out_dir = DERIVED_DIR / model.key
    out_dir.mkdir(parents=True, exist_ok=True)
    for round_number in DRE_ADJUSTABLE_ROUNDS:
        path = model.files[round_number]
        samples, symbols, dates = load_round_samples(path)
        adjusted = samples.copy()
        adjusted[symbols.index("DRE")] = 0.0
        target = out_dir / f"{path.stem}_dre_zeroed.npz"
        np.savez_compressed(
            target,
            forecast_samples=adjusted,
            asset_symbols=np.array(symbols),
            forecast_dates=np.array(dates),
        )
        written.append(target)
    return written


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def markdown_table(frame: pd.DataFrame, float_format: str = "{:.6f}") -> str:
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    divider = "|" + "|".join(["---"] * len(frame.columns)) + "|"
    lines = [header, divider]
    for row in frame.itertuples(index=False):
        cells = [float_format.format(v) if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    started = datetime.now(timezone.utc)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    checks: list[str] = []

    # --- inputs -----------------------------------------------------------
    for required in (PRICE_PATH, SCHEDULE_PATH, OFFICIAL_RPS_REFERENCE):
        if not required.is_file():
            raise FileNotFoundError(f"Required input not found: {required}")

    schedule = pd.read_csv(SCHEDULE_PATH)
    if sorted(schedule["round"]) != EXPECTED_ROUNDS:
        raise ValueError(f"Round schedule does not contain rounds 1-12: {list(schedule['round'])}")

    models = discover_models()
    if not models:
        raise FileNotFoundError(f"No model round_outputs folders found under {RESULTS_DIR}")
    for model in models:
        if sorted(model.files) != EXPECTED_ROUNDS:
            raise FileNotFoundError(
                f"{model.key}: expected 12 round NPZ files, found rounds {sorted(model.files)}"
            )
    print(f"Discovered {len(models)} model(s):")
    for model in models:
        print(f"  {model.key}: {len(model.files)} rounds in {model.directory.relative_to(REPO_ROOT)}")

    # Hash the raw artifacts so we can prove they were not modified.
    hashes_before = {p: sha256_of(p) for m in models for p in m.files.values()}

    # --- ground truth (built once, reused by every model) ------------------
    prices = load_prices()
    ground_truth = build_ground_truth(prices, schedule)
    gt_sums = ground_truth[RANK_COLUMNS].sum(axis=1)
    if not np.allclose(gt_sums, 1.0, atol=1e-12):
        raise ValueError("Ground-truth quintile rows do not sum to 1")
    if len(ground_truth) != len(EXPECTED_ROUNDS) * EXPECTED_ASSETS:
        raise ValueError(f"Ground truth has {len(ground_truth)} rows, expected 1200")
    ties_per_round = ground_truth.groupby("round")["tied_assets_in_round"].first()
    checks.append(
        f"Ground truth: 1200 rows (12 rounds x 100 assets), all quintile rows sum to 1 "
        f"(max |sum-1| = {np.max(np.abs(gt_sums - 1)):.2e})"
    )
    checks.append(
        f"Ground-truth tied assets per round: {ties_per_round.to_dict()}; "
        f"agreement with the faithful port of the official tie block: "
        f"{int(ground_truth['matches_official_tie_port'].sum())}/1200 rows"
    )

    gt_path = EVAL_DIR / "m6_ground_truth_quintiles.csv"
    ground_truth.drop(columns=["matches_official_tie_port"]).to_csv(gt_path, index=False)

    # --- per-model evaluation ---------------------------------------------
    all_rps, all_dre, all_diag, derived_files = [], [], [], {}
    generated: list[Path] = [gt_path]

    for model in models:
        probabilities, rps_frame, dre_frame, diag_frame = evaluate_model(
            model, ground_truth, schedule, checks
        )
        if len(rps_frame) != len(EXPECTED_ROUNDS) * EXPECTED_ASSETS:
            raise ValueError(f"{model.key}: expected 1200 asset-round scores, got {len(rps_frame)}")
        if rps_frame["RPS"].isna().any():
            raise ValueError(f"{model.key}: RPS contains missing values")

        prob_path = EVAL_DIR / f"predicted_quintile_probabilities_{model.key}.csv"
        rps_path = EVAL_DIR / f"rps_by_asset_{model.key}.csv"
        probabilities.to_csv(prob_path, index=False)
        rps_frame.to_csv(rps_path, index=False)
        generated += [prob_path, rps_path]

        all_rps.append(rps_frame)
        all_dre.append(dre_frame)
        all_diag.append(diag_frame)
        derived_files[model.key] = write_dre_adjusted_copies(model, dre_frame)
        generated += derived_files[model.key]

    naive = naive_benchmark_rps(ground_truth)
    all_rps.append(naive)

    # --- aggregation -------------------------------------------------------
    rps_all = pd.concat(all_rps, ignore_index=True)
    by_round = (rps_all.groupby(["model", "round"], as_index=False)["RPS"]
                .mean().rename(columns={"RPS": "mean_RPS"}))
    overall = (by_round.groupby("model", as_index=False)["mean_RPS"]
               .mean().rename(columns={"mean_RPS": "overall_mean_RPS"}))

    # Equal round sizes => mean of round means must equal mean of all 1200 scores.
    flat_mean = rps_all.groupby("model")["RPS"].mean()
    for row in overall.itertuples():
        if not np.isclose(row.overall_mean_RPS, flat_mean[row.model], atol=1e-12):
            raise ValueError(f"{row.model}: mean of round means != mean of asset scores")
    checks.append(
        "For every model the mean of the 12 round means equals the mean of all 1,200 "
        f"asset-round scores (max difference "
        f"{max(abs(r.overall_mean_RPS - flat_mean[r.model]) for r in overall.itertuples()):.2e})"
    )
    recomputed = rps_all.groupby(["model", "round"])["RPS"].apply(lambda s: s.mean())
    if not np.allclose(by_round.set_index(["model", "round"])["mean_RPS"], recomputed, atol=1e-15):
        raise ValueError("Round means do not equal the mean of their 100 asset scores")
    checks.append("Every round mean equals the mean of that round's 100 asset RPS values")

    by_round_path = EVAL_DIR / "rps_by_round.csv"
    by_round.to_csv(by_round_path, index=False)

    comparison = by_round.pivot(index="model", columns="round", values="mean_RPS")
    comparison.columns = [f"Round {c}" for c in comparison.columns]
    comparison = comparison.merge(
        overall.set_index("model"), left_index=True, right_index=True
    ).sort_values("overall_mean_RPS")
    comparison = comparison.rename(columns={"overall_mean_RPS": "Overall Mean RPS"})
    comparison.index.name = "Model (lower RPS is better)"
    comparison_path = EVAL_DIR / "model_comparison_rps.csv"
    comparison.to_csv(comparison_path)

    long_comparison = by_round.pivot(index="round", columns="model", values="mean_RPS")
    long_comparison.index.name = "Round"
    long_path = EVAL_DIR / "rps_round_comparison_long.csv"
    long_comparison.to_csv(long_path)

    dre_all = pd.concat(all_dre, ignore_index=True)
    dre_path = EVAL_DIR / "dre_raw_forecast_inspection.csv"
    dre_all.to_csv(dre_path, index=False)

    diagnostics = pd.concat(all_diag, ignore_index=True)
    diag_path = EVAL_DIR / "forecast_dispersion_diagnostics.csv"
    diagnostics.to_csv(diag_path, index=False)
    generated += [by_round_path, comparison_path, long_path, dre_path, diag_path]

    # --- raw artifacts unchanged ------------------------------------------
    changed = [p.name for p, digest in hashes_before.items() if sha256_of(p) != digest]
    if changed:
        raise RuntimeError(f"Raw NPZ files were modified during evaluation: {changed}")
    checks.append(
        f"All {len(hashes_before)} raw NPZ files are byte-identical before and after the run "
        "(SHA-256 verified); they were opened read-only"
    )

    # --- reports -----------------------------------------------------------
    report_path = write_central_report(
        models, comparison, long_comparison, ground_truth, dre_all, diagnostics,
        derived_files, checks, started, generated,
    )
    generated.append(report_path)
    for model in models:
        model_report = write_model_report(
            model, by_round, overall, dre_all, derived_files.get(model.key, []),
            checks, started,
        )
        if model_report:
            generated.append(model_report)

    # --- console summary ---------------------------------------------------
    print("\nOverall mean RPS (lower is better):")
    for row in overall.sort_values("overall_mean_RPS").itertuples():
        print(f"  {row.model:<45} {row.overall_mean_RPS:.6f}")
    print(f"\nGenerated {len(generated)} file(s) under {EVAL_DIR.relative_to(REPO_ROOT)} "
          "and the model report folders.")


def write_central_report(models, comparison, long_comparison, ground_truth, dre_all,
                         diagnostics, derived_files, checks, started, generated) -> Path:
    path = EVAL_DIR / "m6_rps_evaluation_report.md"

    comparison_md = comparison.reset_index()
    long_md = long_comparison.reset_index()
    dre_md = dre_all[[
        "model", "round", "all_zero", "min_daily_log_return", "max_daily_log_return",
        "min_four_week_return", "median_four_week_return", "max_four_week_return",
    ]]

    derived_lines = []
    for key, paths in derived_files.items():
        if paths:
            derived_lines.append(
                f"- `{key}`: {len(paths)} DRE-zeroed reference copies for rounds "
                f"{', '.join(str(r) for r in DRE_ADJUSTABLE_ROUNDS)} in "
                f"`{DERIVED_DIR.relative_to(REPO_ROOT).as_posix()}/{key}/`"
            )
        else:
            derived_lines.append(f"- `{key}`: no copies needed (DRE forecasts already zero)")

    text = f"""# M6 quintile post-processing and RPS evaluation

Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
Evaluator: `scripts/evaluate_m6_rps.py` (run started {started.strftime('%Y-%m-%d %H:%M:%S UTC')})

## 1. Purpose

Convert the raw sampled trajectories produced by the inference stage into
M6-format quintile probability forecasts, build the realised M6 outcome
independently from the official price file, and score both models with the
Ranked Probability Score (RPS). Lower RPS is better.

## 2. Models evaluated

{chr(10).join(f"- **{m.label}** - 12 raw NPZ rounds in `{m.directory.relative_to(REPO_ROOT).as_posix()}`" for m in models)}
- **{NAIVE_LABEL}** - the flat forecast [0.20, 0.20, 0.20, 0.20, 0.20] for every
  asset in every round, scored through exactly the same evaluator as a reference
  point. It is a validation benchmark, not a model.

## 3. Input artifacts

- Raw forecasts: one NPZ per model per round, each containing `forecast_samples`
  with shape (100 assets, 100 sampled trajectories, 20 forecast weekdays),
  `asset_symbols` and `forecast_dates`. The NPZ files are read-only research
  artifacts; the evaluator hashes them before and after the run.
- Realised prices: `Data/assets_m6.csv` (official M6 daily adjusted closes,
  100 symbols, {len(ground_truth) // EXPECTED_ASSETS} evaluation windows used here).
- Round anchors: `Data/metadata/m6_round_schedule.csv` (Stage 3).
- Methodological reference: `RPS Reference/RPS and IR calculation.py` from
  Mcompetitions/M6-methods.

## 4. Forecast post-processing

For each asset and each sampled trajectory the 20 predicted daily log returns
are summed and converted to a simple four-week return:

    sampled four-week return = exp(sum of the 20 predicted daily log returns) - 1

This yields 100 assets x 100 sampled four-week returns per round.

## 5. Cross-sectional ranking of each sampled future

The M6 task is cross-sectional, so each sampled future is treated as one
complete scenario for the whole universe: within sample *s* the 100 assets are
ranked from lowest to highest four-week return and split into five equal
quintiles (lowest 20% = Rank1 ... highest 20% = Rank5).

## 6. Quintile probability estimation

An asset's probability for a quintile is its average membership of that quintile
across the 100 sampled futures. Every probability row is checked to sum to 1
(to within 1e-9) and to lie inside [0, 1].

## 7. Ground truth from assets_m6.csv

For round *r* the realised return uses the official round anchors - the round's
forecast origin as the opening date and the next anchor as the closing date, the
same interval the model forecast:

    actual_return = (close_price - open_price) / open_price

Missing prices follow the official evaluator's rule: the most recent available
price at or before the date is carried forward. The carry-forward is applied
over the full price history rather than only within the round window, because
DRE stops appearing in the file after 2022-11-28; the official script's
in-window lookup has no price to fall back on for the last rounds. No other
interpolation is used. Ground truth is built once and reused for both models.

## 8. Official M6 tie handling

Ranking uses `rank(method="min")`, and a block of k tied assets receives the
average quintile membership of the k consecutive rank slots it occupies - the
official "Handle Ties" principle. Untied assets therefore get one-hot targets
such as [0, 0, 0, 1, 0], and fractional membership is retained where a tied
block crosses a quintile boundary.

Two departures from `RPS and IR calculation.py` are made deliberately and are
implemented in `rank_to_quintiles`:

1. Quintile boundaries are derived from the number of ranked assets (100). The
   official script derives them from `max(min-rank)`, which falls below 100 when
   ties exist and shifts every boundary. The two definitions agree exactly when
   there are no ties.
2. The official if/elif chain assigns membership only for the first quintile a
   tied block touches, so a boundary-straddling block yields a row summing to
   less than 1. Here the membership is spread across every quintile the block
   spans, so all rows sum to 1, as the M6 probability format requires.

A faithful port of the official tie block (`official_reference_quintiles`) is
kept in the evaluator purely to measure agreement; it never produces results.

Applying this tie-aware ranking to the *sampled forecasts* is this project's
post-processing extension: the official script defines the treatment only for
realised outcomes. It matters here because Chronos outputs are quantised and can
repeat values.

## 9. RPS

For each asset, with `actual` and `forecast` the five quintile values:

    RPS = mean( (cumsum(actual) - cumsum(forecast))^2 )

over the five cumulative positions, matching the official implementation.
Aggregation: one RPS per asset per round, the round score is the mean of its 100
asset scores, and a model's final score is the mean of its 12 round scores.
Because every round holds exactly 100 assets this equals the mean of all 1,200
asset-round values, which the evaluator verifies.

## 10. Information ratio - excluded

IR is out of scope for this research. `IR_calculation()` is not called, ported or
reproduced, and no investment weights, portfolio returns, return standard
deviations, information ratios or overall M6 competition rankings are computed.

## 11. Round-by-round comparison

{markdown_table(long_md)}

## 12. Overall comparison (lower is better)

{markdown_table(comparison_md)}

## 13. Naive equal-probability benchmark

The flat [0.2]*5 forecast is scored through the evaluator itself, not asserted to
be 0.16. Its computed value appears in the tables above; with one-hot targets
spread evenly over the five quintiles the theoretical value is 0.16, and any
deviation reflects the realised tie structure rather than an adjustment.

## 14. DRE raw-output inspection

DRE was acquired and stopped trading on {DRE_ACQUISITION_LAST_TRADING_DAY}; its
official M6 price is carried forward afterwards, so its realised competition
return is exactly zero in the affected rounds. The raw model forecasts were
inspected, not modified (values shown with significant digits, so a genuinely
tiny non-zero forecast is not displayed as an exact zero):

{markdown_table(dre_md, '{:.6g}')}

The primary RPS results above use these raw forecasts exactly as generated.
Clearly labelled DRE-zeroed reference copies were written for rounds
{', '.join(str(r) for r in DRE_ADJUSTABLE_ROUNDS)} (origins already after the
acquisition) as separate derived artifacts; they are not used for any result in
this report. Round 8 is deliberately left alone: its forecast window crosses the
acquisition date and needs date-specific treatment.

{chr(10).join(derived_lines)}

## 15. Validation checks

{chr(10).join('- ' + c for c in checks)}

## 16. Forecast sharpness diagnostic (interpretation aid)

RPS rewards being both correct and appropriately uncertain, so how *confident*
each model's quintile probabilities are matters as much as where they point.
`mean_max_probability` is the average largest probability an asset receives
(0.2 = maximally diffuse, 1.0 = a one-hot bet) and `dispersion_ratio` is the
spread of the assets' median predicted four-week returns divided by the typical
spread within a single asset's own samples. A ratio above 1 means the
cross-sectional ordering is nearly the same in every sampled future, which
produces near-one-hot probabilities; below 1 means sampling noise reshuffles the
ordering and the probabilities stay diffuse. Diagnostic only - it feeds no result.

{markdown_table(diagnostics.groupby('model', as_index=False)[['mean_max_probability', 'mean_probability_entropy', 'dispersion_ratio']].mean(), '{:.3f}')}

Per-round values are in `Results/Evaluation/forecast_dispersion_diagnostics.csv`.

## 17. Limitations and interpretation notes

- The evaluation scores 12 rounds x 100 assets per model; with 12 observations
  per model, round-to-round differences are noisy and no significance testing is
  performed here.
- Quintile probabilities are estimated from 100 sampled trajectories, so each
  probability is resolved to 0.01 and carries Monte Carlo error of roughly
  0.02-0.05; a model whose samples are nearly deterministic will produce
  near-degenerate probability rows.
- Forecast dates are the shared weekday calendar of Stages 2-3, while realised
  returns come from the official price file's own trading calendar. Both are
  anchored on the same round dates, so the four-week interval matches even
  though intermediate days need not.
- The raw forecasts are used unchanged, including DRE, so both models are
  penalised for any non-zero DRE prediction in the post-acquisition rounds
  exactly as the competition would have penalised a live participant.
- A model can score worse than the flat 0.16 benchmark: confident probabilities
  that point the wrong way are penalised more heavily than a diffuse forecast.
  The sharpness diagnostic above is the way to tell an informative model from a
  merely confident one.

## 18. Generated files

{chr(10).join('- `' + p.relative_to(REPO_ROOT).as_posix() + '`' for p in sorted(set(generated)))}
"""
    path.write_text(text, encoding="utf-8")
    return path


def write_model_report(model, by_round, overall, dre_all, derived_paths, checks, started):
    if model.reports_dir is None:
        return None
    rounds = by_round[by_round["model"] == model.label][["round", "mean_RPS"]]
    naive_rounds = by_round[by_round["model"] == NAIVE_LABEL][["round", "mean_RPS"]]
    merged = rounds.merge(naive_rounds, on="round", suffixes=("_model", "_naive"))
    merged.columns = ["Round", "Model RPS", "Naive RPS"]
    merged["Difference (model - naive)"] = merged["Model RPS"] - merged["Naive RPS"]

    model_overall = float(overall.loc[overall["model"] == model.label, "overall_mean_RPS"].iloc[0])
    naive_overall = float(overall.loc[overall["model"] == NAIVE_LABEL, "overall_mean_RPS"].iloc[0])
    dre_model = dre_all[dre_all["model"] == model.label]
    model_checks = [c for c in checks if c.startswith(model.key)]

    text = f"""# {model.label} - M6 RPS evaluation

Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
Evaluator: `scripts/evaluate_m6_rps.py`

## Inputs

- Raw NPZ forecasts (unchanged, read-only): 12 files in
  `{model.directory.relative_to(REPO_ROOT).as_posix()}`
{chr(10).join(f"  - `{p.name}`" for p in model.files.values())}
- Realised prices: `Data/assets_m6.csv`; round anchors:
  `Data/metadata/m6_round_schedule.csv`.
- Rounds evaluated: 12 (1-12), 100 assets each = 1,200 asset-round scores.

## Methodology

Each sampled trajectory's 20 predicted daily log returns are summed and
converted with `exp(sum) - 1` to a four-week simple return. Each of the 100
sampled futures is ranked cross-sectionally across the 100 assets and split into
five quintiles using the official M6 tie-aware rule; an asset's quintile
probabilities are its average membership across those samples. The realised
target is built independently from `Data/assets_m6.csv` using the official
anchor-to-anchor return and the same tie-aware quintile construction.

RPS per asset is `mean((cumsum(actual) - cumsum(forecast))^2)` over the five
cumulative positions; the round score is the mean of its 100 asset scores and the
final score is the mean of the 12 round scores. The information ratio was not
calculated.

## Results (lower is better)

**Final mean RPS: {model_overall:.6f}** (naive equal-probability benchmark:
{naive_overall:.6f}; difference {model_overall - naive_overall:+.6f})

{markdown_table(merged)}

## DRE observation

{markdown_table(dre_model[['round', 'all_zero', 'min_daily_log_return', 'max_daily_log_return', 'min_four_week_return', 'median_four_week_return', 'max_four_week_return']], '{:.6g}')}

The results above use these raw forecasts unchanged. {'DRE-zeroed reference copies for rounds ' + ', '.join(str(r) for r in DRE_ADJUSTABLE_ROUNDS) + ' were written to `' + DERIVED_DIR.relative_to(REPO_ROOT).as_posix() + '/' + model.key + '/` as separate derived artifacts and are not used in any result here.' if derived_paths else 'No DRE-adjusted copies were needed.'}

## Validation

{chr(10).join('- ' + c for c in model_checks)}

## Generated evaluation artifacts

- `Results/Evaluation/m6_ground_truth_quintiles.csv`
- `Results/Evaluation/predicted_quintile_probabilities_{model.key}.csv`
- `Results/Evaluation/rps_by_asset_{model.key}.csv`
- `Results/Evaluation/rps_by_round.csv`
- `Results/Evaluation/model_comparison_rps.csv`
- `Results/Evaluation/rps_round_comparison_long.csv`
- `Results/Evaluation/dre_raw_forecast_inspection.csv`
- `Results/Evaluation/m6_rps_evaluation_report.md`
"""
    path = model.reports_dir / f"{model.key.lower()}_rps_evaluation_report.md"
    path.write_text(text, encoding="utf-8")
    return path


if __name__ == "__main__":
    main()
