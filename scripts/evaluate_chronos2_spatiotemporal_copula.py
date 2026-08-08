"""Chronos-2 post-processing method 1: spatiotemporal Gaussian copula.

Chronos-2 outputs MARGINAL daily quantiles per asset and day, not joint
trajectories. The M6 task is cross-sectional and cumulative over 20 days, so the
marginals alone are not enough: a joint distribution over (asset, day) is needed
before four-week returns can be ranked. This script supplies that joint structure
with a separable Gaussian copula whose dependence is estimated ONLY from each
round's own 512-day Stage 3 context:

    latent Z ~ N(0, C_time (x) C_asset)      (Kronecker / separable)
    U = Phi(Z), clipped to [0.01, 0.99]
    scenario returns = Chronos-2 marginal quantile curve evaluated at U

The resulting (100, 1000, 20) array holds SYNTHETIC COPULA SCENARIOS DERIVED
FROM CHRONOS-2 - they are not native Chronos-2 samples and must never be
described as such. Only the marginals come from the model; all dependence is
imposed here from history.

Scoring reuses scripts/evaluate_m6_rps.py unchanged (four-week return conversion,
cross-sectional ranking, tie handling, ground truth, RPS), so this method is
directly comparable with the two sampled-trajectory models.

The raw Chronos-2 NPZ files are read-only: they are hashed before and after.

Run:  python scripts/evaluate_chronos2_spatiotemporal_copula.py
"""

from __future__ import annotations

import hashlib
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.covariance import LedoitWolf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from evaluate_m6_rps import (  # noqa: E402
    EXPECTED_ROUNDS, RANK_COLUMNS, build_ground_truth, load_prices,
    naive_benchmark_rps, quintile_probabilities, rps_scores,
)

MODEL_DIR = REPO_ROOT / "Results" / "Chronos_2_120M"
ROUND_OUTPUTS = MODEL_DIR / "round_outputs"
CONTEXT_DIR = REPO_ROOT / "Data" / "processed" / "rolling_origins"
SCHEDULE_PATH = REPO_ROOT / "Data" / "metadata" / "m6_round_schedule.csv"
MAIN_ROUND_RPS = REPO_ROOT / "Results" / "Evaluation" / "rps_by_round.csv"

PREFIX = "chronos2_spatiotemporal_copula"
PROBS_CSV = MODEL_DIR / f"{PREFIX}_quintile_probabilities.csv"
RPS_CSV = MODEL_DIR / f"{PREFIX}_rps.csv"
DIAGNOSTICS_CSV = MODEL_DIR / f"{PREFIX}_diagnostics.csv"

N_SCENARIOS = 1000
RANDOM_SEED = 42
N_ASSETS = 100
HORIZON = 20
N_QUANTILES = 21
CONTEXT_LENGTH = 512
NAIVE_RPS = 0.16
MODEL_LABEL = "Chronos-2 120M (spatiotemporal Gaussian copula)"

EIGENVALUE_FLOOR = 1e-8
MAX_ABS_RHO = 0.99


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def discover_quantile_files(directory: Path = ROUND_OUTPUTS) -> dict[int, Path]:
    """Locate the raw Chronos-2 quantile NPZ for each round (filenames parsed)."""
    pattern = re.compile(r"round(\d{2}).*\.npz$", re.IGNORECASE)
    found: dict[int, list[Path]] = {}
    for path in sorted(directory.glob("*.npz")):
        match = pattern.search(path.name)
        if match:
            found.setdefault(int(match.group(1)), []).append(path)

    problems = []
    duplicates = {r: [p.name for p in ps] for r, ps in found.items() if len(ps) > 1}
    if duplicates:
        problems.append(f"duplicate files for round(s) {duplicates}")
    missing = [r for r in EXPECTED_ROUNDS if r not in found]
    if missing:
        problems.append(f"missing round(s) {missing}")
    if problems:
        raise FileNotFoundError(
            f"Chronos-2 quantile files in {directory} are incomplete:\n  - "
            + "\n  - ".join(problems)
        )
    return {r: found[r][0] for r in EXPECTED_ROUNDS}


def load_quantile_forecasts(path: Path, round_number: int, schedule: pd.DataFrame,
                            official_order: list[str]):
    """Load and validate one raw Chronos-2 NPZ. Never modifies the file."""
    with np.load(path, allow_pickle=False) as data:
        expected_keys = {"quantile_forecasts", "quantile_levels",
                         "asset_symbols", "forecast_dates"}
        missing = expected_keys - set(data.files)
        if missing:
            raise ValueError(f"{path.name}: missing array(s) {sorted(missing)}")
        forecasts = data["quantile_forecasts"].astype(np.float64)
        levels = data["quantile_levels"].astype(np.float64)
        symbols = [str(s) for s in data["asset_symbols"]]
        dates = [str(d) for d in data["forecast_dates"]]

    if forecasts.shape != (N_ASSETS, HORIZON, N_QUANTILES):
        raise ValueError(f"{path.name}: shape {forecasts.shape} != "
                         f"({N_ASSETS}, {HORIZON}, {N_QUANTILES})")
    if levels.shape != (N_QUANTILES,) or not np.all(np.diff(levels) > 0):
        raise ValueError(f"{path.name}: quantile levels are not 21 increasing values")
    if not np.isfinite(forecasts).all():
        raise ValueError(f"{path.name}: non-finite quantile forecasts")
    if symbols != official_order:
        raise ValueError(f"{path.name}: asset_symbols differ from the official M6 order")

    row = schedule.set_index("round").loc[round_number]
    expected_dates = [d.strftime("%Y-%m-%d")
                      for d in pd.bdate_range(row.forecast_start_date, row.forecast_end_date)]
    if dates != expected_dates:
        raise ValueError(f"{path.name}: forecast dates do not match round {round_number}")

    return forecasts, levels, symbols, dates


def load_context(round_number: int, schedule: pd.DataFrame,
                 official_order: list[str]) -> np.ndarray:
    """Round's Stage 3 context as (512 days, 100 assets); genuine NaNs preserved.

    Asserts the context ends on the round's forecast origin and holds nothing
    after it, so no future information can enter the dependence estimates.
    """
    row = schedule.set_index("round").loc[round_number]
    frame = pd.read_csv(CONTEXT_DIR / f"round_{round_number:02d}_context.csv",
                        parse_dates=["date"])
    if frame.shape[0] != CONTEXT_LENGTH:
        raise ValueError(f"Round {round_number}: context has {frame.shape[0]} rows")
    if list(frame.columns) != ["date"] + official_order:
        raise ValueError(f"Round {round_number}: context columns are not the official order")

    origin = pd.Timestamp(row.origin_date)
    if frame["date"].iloc[-1] != origin or (frame["date"] > origin).any():
        raise ValueError(
            f"Round {round_number}: context does not end on the origin {row.origin_date} "
            "- refusing to estimate dependence from data that could include the future"
        )
    return frame[official_order].to_numpy(dtype=np.float64)


# ---------------------------------------------------------------------------
# A. Marginal quantile curves
# ---------------------------------------------------------------------------
def marginal_curves(forecasts: np.ndarray) -> np.ndarray:
    """Enforce non-decreasing quantile values per (asset, day).

    Chronos-2's output is already monotone; the cumulative maximum only guards
    against numerical crossing so the curve is a valid inverse CDF. The raw NPZ
    is untouched - this works on the loaded copy.
    """
    return np.maximum.accumulate(forecasts, axis=2)


def invert_marginal(curve: np.ndarray, levels: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Monotonic (piecewise-linear) interpolation of one (asset, day) curve.

    Percentiles outside the model's own 0.01-0.99 range are clipped to those
    boundaries rather than extrapolated, so no tail behaviour is invented.
    """
    clipped = np.clip(u, levels[0], levels[-1])
    return np.interp(clipped, levels, curve)


# ---------------------------------------------------------------------------
# B/C. Historical dependence from the round's own context
# ---------------------------------------------------------------------------
def gaussian_scores(context: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Complete-case history -> standard-normal scores, (n_usable, 100).

    Each asset's observed returns are converted to empirical percentile ranks
    and then to normal scores. Rows are restricted to dates where every asset
    has a genuine observation: leading missing history (OGN in every round, CARR
    in round 1) is never backfilled, zero-filled or imputed - those rows are
    simply not usable for a joint estimate.
    """
    usable_mask = ~np.isnan(context).any(axis=1)
    usable = context[usable_mask]
    n_usable, n_series = usable.shape
    if n_usable <= n_series:
        raise ValueError(
            f"Only {n_usable} complete historical rows for {n_series} assets - "
            "too few to estimate dependence"
        )
    ranks = pd.DataFrame(usable).rank(axis=0, method="average").to_numpy()
    uniforms = ranks / (n_usable + 1.0)
    return norm.ppf(uniforms), usable_mask


def asset_correlation(scores: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """Ledoit-Wolf shrunk 100x100 correlation matrix, made positive definite."""
    estimator = LedoitWolf().fit(scores)
    covariance = estimator.covariance_
    shrinkage = float(estimator.shrinkage_)

    sd = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(sd, sd)
    correlation = (correlation + correlation.T) / 2.0

    eigenvalues = np.linalg.eigvalsh(correlation)
    min_before = float(eigenvalues.min())
    if min_before < EIGENVALUE_FLOOR:
        values, vectors = np.linalg.eigh(correlation)
        values = np.clip(values, EIGENVALUE_FLOOR, None)
        correlation = vectors @ np.diag(values) @ vectors.T
        sd = np.sqrt(np.diag(correlation))
        correlation = correlation / np.outer(sd, sd)
        correlation = (correlation + correlation.T) / 2.0
    min_after = float(np.linalg.eigvalsh(correlation).min())
    if min_after <= 0:
        raise ValueError("Asset correlation matrix is not positive definite")
    return correlation, shrinkage, min_before, min_after


def temporal_rho(scores: np.ndarray) -> float:
    """Pooled lag-one dependence across assets, on the Gaussian score series.

    Deliberately not floored away from zero: if the daily return series show no
    lag-one structure the estimate stays near zero and the time correlation
    matrix is effectively the identity.
    """
    current, lagged = scores[:-1], scores[1:]
    denominator = float(np.sum(current * current))
    if denominator == 0.0:
        return 0.0
    rho = float(np.sum(current * lagged) / denominator)
    return float(np.clip(rho, -MAX_ABS_RHO, MAX_ABS_RHO))


def ar1_time_correlation(rho: float, horizon: int = HORIZON) -> np.ndarray:
    lags = np.abs(np.subtract.outer(np.arange(horizon), np.arange(horizon)))
    return np.power(float(rho), lags)


# ---------------------------------------------------------------------------
# D. Scenario generation
# ---------------------------------------------------------------------------
def generate_scenarios(curve: np.ndarray, levels: np.ndarray,
                       asset_corr: np.ndarray, time_corr: np.ndarray,
                       n_scenarios: int, seed: int) -> np.ndarray:
    """Separable Gaussian copula scenarios, returned as (100, n_scenarios, 20).

    Z = L_asset @ G @ L_time.T gives Cov(vec Z) = time_corr (x) asset_corr with
    unit marginal variances, so Phi(Z) is uniform on each (asset, day).
    """
    n_assets = asset_corr.shape[0]
    horizon = time_corr.shape[0]
    if curve.shape[:2] != (n_assets, horizon):
        raise ValueError(
            f"Marginal curves {curve.shape[:2]} do not match the dependence matrices "
            f"({n_assets} assets, {horizon} days)"
        )

    chol_asset = np.linalg.cholesky(asset_corr)
    chol_time = np.linalg.cholesky(time_corr)

    rng = np.random.default_rng(seed)
    white = rng.standard_normal((n_scenarios, n_assets, horizon))
    latent = np.einsum("ij,sjk,lk->sil", chol_asset, white, chol_time, optimize=True)
    uniforms = norm.cdf(latent)

    scenarios = np.empty((n_assets, n_scenarios, horizon), dtype=np.float64)
    for asset in range(n_assets):
        for day in range(horizon):
            scenarios[asset, :, day] = invert_marginal(
                curve[asset, day], levels, uniforms[:, asset, day]
            )
    return scenarios


# ---------------------------------------------------------------------------
def main() -> None:
    started = datetime.now(timezone.utc)
    schedule = pd.read_csv(SCHEDULE_PATH)
    official_order = list(pd.read_csv(
        CONTEXT_DIR / "round_01_context.csv", nrows=0).columns[1:])

    files = discover_quantile_files()
    hashes_before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files.values()}
    print(f"Found {len(files)} raw Chronos-2 quantile files in "
          f"{ROUND_OUTPUTS.relative_to(REPO_ROOT)}")

    ground_truth = build_ground_truth(load_prices(), schedule)

    prob_rows, rps_rows, diagnostics = [], [], []
    for round_number, path in files.items():
        forecasts, levels, symbols, _ = load_quantile_forecasts(
            path, round_number, schedule, official_order)
        context = load_context(round_number, schedule, official_order)

        scores, usable_mask = gaussian_scores(context)
        n_usable = int(usable_mask.sum())
        asset_corr, shrinkage, min_before, min_after = asset_correlation(scores)
        rho = temporal_rho(scores)
        time_corr = ar1_time_correlation(rho)

        curve = marginal_curves(forecasts)
        scenarios = generate_scenarios(curve, levels, asset_corr, time_corr,
                                       N_SCENARIOS, RANDOM_SEED)
        if scenarios.shape != (N_ASSETS, N_SCENARIOS, HORIZON):
            raise ValueError(f"Round {round_number}: scenarios {scenarios.shape}")
        if not np.isfinite(scenarios).all():
            raise ValueError(f"Round {round_number}: non-finite scenario returns")

        # --- M6 probabilities and RPS, via the existing evaluator -----------
        probabilities = quintile_probabilities(scenarios)
        row_sums = probabilities.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-9):
            raise ValueError(f"Round {round_number}: probabilities do not sum to 1")
        if probabilities.min() < -1e-12 or probabilities.max() > 1 + 1e-12:
            raise ValueError(f"Round {round_number}: probabilities outside [0, 1]")
        # Each scenario must distribute exactly 20 assets per quintile.
        expected_mass = N_ASSETS / 5.0
        if not np.allclose(probabilities.sum(axis=0), expected_mass, atol=1e-9):
            raise ValueError(
                f"Round {round_number}: cross-sectional quintile mass is not "
                f"{expected_mass} per quintile"
            )

        truth = ground_truth[ground_truth["round"] == round_number].set_index("symbol")
        truth = truth.loc[symbols]
        scores_rps = rps_scores(truth[RANK_COLUMNS].to_numpy(dtype=float), probabilities)
        if not np.isfinite(scores_rps).all():
            raise ValueError(f"Round {round_number}: non-finite RPS")

        for i, symbol in enumerate(symbols):
            prob_rows.append({
                "model": MODEL_LABEL, "round": round_number, "symbol": symbol,
                **{col: float(probabilities[i, q]) for q, col in enumerate(RANK_COLUMNS)},
                "RPS": float(scores_rps[i]),
            })
        rps_rows.append({"round": round_number, "mean_RPS": float(scores_rps.mean())})
        diagnostics.append({
            "round": round_number,
            "context_rows": CONTEXT_LENGTH,
            "usable_historical_rows": n_usable,
            "assets_with_leading_gaps": ", ".join(
                f"{official_order[a]}:{int(np.isnan(context[:, a]).sum())}"
                for a in range(N_ASSETS) if np.isnan(context[:, a]).any()
            ) or "none",
            "temporal_rho": round(rho, 6),
            "ledoit_wolf_shrinkage": round(shrinkage, 6),
            "min_eigenvalue_before_repair": round(min_before, 10),
            "min_eigenvalue_after_repair": round(min_after, 10),
        })
        print(f"  round {round_number:02d}: usable rows {n_usable:>3} | rho {rho:+.4f} | "
              f"LW shrinkage {shrinkage:.4f} | RPS {scores_rps.mean():.6f}")

    # --- aggregation --------------------------------------------------------
    rps_frame = pd.DataFrame(rps_rows).sort_values("round")
    overall = float(rps_frame["mean_RPS"].mean())
    probabilities_frame = pd.DataFrame(prob_rows)
    if len(probabilities_frame) != len(EXPECTED_ROUNDS) * N_ASSETS:
        raise ValueError(f"Expected 1200 asset-round rows, got {len(probabilities_frame)}")
    if not np.isclose(overall, probabilities_frame["RPS"].mean(), atol=1e-12):
        raise ValueError("Mean of round means != mean of the 1,200 asset scores")

    # --- validation ---------------------------------------------------------
    naive_overall = float(
        naive_benchmark_rps(ground_truth).groupby("round")["RPS"].mean().mean()
    )
    if not np.isclose(naive_overall, NAIVE_RPS, atol=1e-12):
        raise ValueError(f"Naive benchmark sanity check failed: {naive_overall}")

    changed = [p.name for p, digest in hashes_before.items()
               if hashlib.sha256(p.read_bytes()).hexdigest() != digest]
    if changed:
        raise RuntimeError(f"Raw Chronos-2 NPZ files were modified: {changed}")

    # --- outputs ------------------------------------------------------------
    output = rps_frame.copy()
    output["naive_RPS"] = NAIVE_RPS
    output["difference_vs_naive"] = (output["mean_RPS"] - NAIVE_RPS).round(6)
    output["mean_RPS"] = output["mean_RPS"].round(6)
    output = pd.concat([
        output,
        pd.DataFrame([{
            "round": pd.NA, "mean_RPS": round(overall, 6), "naive_RPS": NAIVE_RPS,
            "difference_vs_naive": round(overall - NAIVE_RPS, 6),
        }]),
    ], ignore_index=True)
    output["round"] = output["round"].astype("Int64")
    output["scope"] = ["round"] * len(rps_frame) + ["overall"]
    output = output[["scope", "round", "mean_RPS", "naive_RPS", "difference_vs_naive"]]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    probabilities_frame.to_csv(PROBS_CSV, index=False)
    output.to_csv(RPS_CSV, index=False)
    pd.DataFrame(diagnostics).to_csv(DIAGNOSTICS_CSV, index=False)

    # --- comparison with the existing models --------------------------------
    print(f"\nOverall RPS (lower is better), {N_SCENARIOS} scenarios, seed {RANDOM_SEED}:")
    print(f"  {MODEL_LABEL:<52} {overall:.6f}")
    if MAIN_ROUND_RPS.is_file():
        existing = pd.read_csv(MAIN_ROUND_RPS).groupby("model")["mean_RPS"].mean()
        for model, value in existing.sort_values().items():
            print(f"  {model:<52} {value:.6f}")
    print(f"\nRuntime {(datetime.now(timezone.utc) - started).total_seconds():.1f}s")
    for path in (PROBS_CSV, RPS_CSV, DIAGNOSTICS_CSV):
        print(f"Wrote {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
