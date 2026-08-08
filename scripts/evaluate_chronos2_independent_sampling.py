"""Chronos-2 post-processing method 2: independent quantile sampling.

A deliberate ablation of method 1. The spatiotemporal copula
(scripts/evaluate_chronos2_spatiotemporal_copula.py) imposes historical
cross-asset and lag-one dependence on the percentile draws. This script changes
exactly one thing: every percentile is drawn INDEPENDENTLY for every
(asset, scenario, forecast day). There is no correlation matrix, no Ledoit-Wolf
estimation, no temporal rho, no AR(1) matrix, no Gaussian copula and no
historical dependence fitting of any kind - the round's Stage 3 context is not
even read, because nothing here is estimated from history.

Everything else is imported from the copula script or the primary evaluator so
the two experiments cannot drift apart: the same NPZ discovery and validation,
the same monotone quantile interpolation and 0.01-0.99 clipping, the same 1,000
scenarios and seed 42, and the same four-week conversion, cross-sectional
ranking, tie handling, ground truth and RPS.

The generated array holds INDEPENDENTLY SAMPLED SCENARIOS DERIVED FROM
CHRONOS-2 MARGINAL QUANTILE FORECASTS. They are not native Chronos-2
trajectories: only the marginals come from the model, and here nothing at all
links one asset, day or scenario to another.

The raw Chronos-2 NPZ files are read-only: they are hashed before and after.

Run:  python scripts/evaluate_chronos2_independent_sampling.py
"""

from __future__ import annotations

import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import evaluate_chronos2_spatiotemporal_copula as copula  # noqa: E402
from evaluate_chronos2_spatiotemporal_copula import (  # noqa: E402
    HORIZON, MODEL_DIR, N_ASSETS, NAIVE_RPS, ROUND_OUTPUTS, SCHEDULE_PATH,
    discover_quantile_files, invert_marginal, load_quantile_forecasts, marginal_curves,
)
from evaluate_m6_rps import (  # noqa: E402
    EXPECTED_ROUNDS, RANK_COLUMNS, build_ground_truth, load_prices,
    naive_benchmark_rps, quintile_probabilities, rps_scores,
)

CONTEXT_DIR = REPO_ROOT / "Data" / "processed" / "rolling_origins"
MAIN_ROUND_RPS = REPO_ROOT / "Results" / "Evaluation" / "rps_by_round.csv"
COPULA_RPS_CSV = MODEL_DIR / "chronos2_spatiotemporal_copula_rps.csv"

PREFIX = "chronos2_independent_sampling"
PROBS_CSV = MODEL_DIR / f"{PREFIX}_quintile_probabilities.csv"
RPS_CSV = MODEL_DIR / f"{PREFIX}_rps.csv"

# Must match the copula experiment exactly - asserted in main().
N_SCENARIOS = 1000
RANDOM_SEED = 42

MODEL_LABEL = "Chronos-2 120M (independent quantile sampling)"


def generate_independent_scenarios(curve: np.ndarray, levels: np.ndarray,
                                   n_scenarios: int, seed: int) -> np.ndarray:
    """Independent percentile draws mapped through the marginal curves.

    One uniform draw per (scenario, asset, day), all mutually independent - the
    ablation of the copula's correlated latent normals. The draws are then
    mapped through the SAME monotone interpolation and 0.01-0.99 clipping used
    by the copula method, so the marginals of the two experiments agree and only
    the dependence differs.

    Returns (n_assets, n_scenarios, horizon).
    """
    n_assets, horizon = curve.shape[0], curve.shape[1]

    rng = np.random.default_rng(seed)
    uniforms = rng.random((n_scenarios, n_assets, horizon))

    scenarios = np.empty((n_assets, n_scenarios, horizon), dtype=np.float64)
    for asset in range(n_assets):
        for day in range(horizon):
            scenarios[asset, :, day] = invert_marginal(
                curve[asset, day], levels, uniforms[:, asset, day]
            )
    return scenarios


def load_copula_round_rps() -> dict[int, float]:
    """Existing method-1 per-round results, read rather than recomputed."""
    if not COPULA_RPS_CSV.is_file():
        raise FileNotFoundError(
            f"Copula results not found at {COPULA_RPS_CSV}. Run "
            "scripts/evaluate_chronos2_spatiotemporal_copula.py first."
        )
    frame = pd.read_csv(COPULA_RPS_CSV)
    rounds = frame[frame["scope"] == "round"]
    return {int(r.round): float(r.mean_RPS) for r in rounds.itertuples()}


def main() -> None:
    started = datetime.now(timezone.utc)

    # --- comparability: every shared setting must match method 1 -----------
    differences = []
    if N_SCENARIOS != copula.N_SCENARIOS:
        differences.append(f"N_SCENARIOS {N_SCENARIOS} vs copula {copula.N_SCENARIOS}")
    if RANDOM_SEED != copula.RANDOM_SEED:
        differences.append(f"RANDOM_SEED {RANDOM_SEED} vs copula {copula.RANDOM_SEED}")
    if differences:
        raise ValueError(
            "This experiment must differ from the copula ONLY in the percentile "
            "dependence mechanism, but these settings differ:\n  - "
            + "\n  - ".join(differences)
        )
    print("Comparability with the copula experiment:")
    print(f"  scenarios {N_SCENARIOS}, seed {RANDOM_SEED}, "
          "interpolation/clipping/quintile/RPS functions imported from the same modules")
    print("  only difference: percentiles drawn independently, no dependence estimated")

    schedule = pd.read_csv(SCHEDULE_PATH)
    official_order = list(pd.read_csv(CONTEXT_DIR / "round_01_context.csv",
                                      nrows=0).columns[1:])

    files = discover_quantile_files()
    hashes_before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files.values()}
    print(f"\nFound {len(files)} raw Chronos-2 quantile files in "
          f"{ROUND_OUTPUTS.relative_to(REPO_ROOT)}")

    ground_truth = build_ground_truth(load_prices(), schedule)
    copula_rps = load_copula_round_rps()

    prob_rows, rps_rows = [], []
    for round_number, path in files.items():
        forecasts, levels, symbols, _ = load_quantile_forecasts(
            path, round_number, schedule, official_order)

        curve = marginal_curves(forecasts)
        scenarios = generate_independent_scenarios(curve, levels, N_SCENARIOS, RANDOM_SEED)
        if scenarios.shape != (N_ASSETS, N_SCENARIOS, HORIZON):
            raise ValueError(f"Round {round_number}: scenarios {scenarios.shape}")
        if not np.isfinite(scenarios).all():
            raise ValueError(f"Round {round_number}: non-finite scenario returns")

        # --- M6 probabilities and RPS, via the existing evaluator -----------
        probabilities = quintile_probabilities(scenarios)
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-9):
            raise ValueError(f"Round {round_number}: probabilities do not sum to 1")
        if probabilities.min() < -1e-12 or probabilities.max() > 1 + 1e-12:
            raise ValueError(f"Round {round_number}: probabilities outside [0, 1]")
        if not np.allclose(probabilities.sum(axis=0), N_ASSETS / 5.0, atol=1e-9):
            raise ValueError(
                f"Round {round_number}: cross-sectional quintile mass is not "
                f"{N_ASSETS / 5.0} per quintile"
            )

        truth = ground_truth[ground_truth["round"] == round_number].set_index("symbol")
        truth = truth.loc[symbols]
        scores = rps_scores(truth[RANK_COLUMNS].to_numpy(dtype=float), probabilities)
        if not np.isfinite(scores).all():
            raise ValueError(f"Round {round_number}: non-finite RPS")

        for i, symbol in enumerate(symbols):
            prob_rows.append({
                "model": MODEL_LABEL, "round": round_number, "symbol": symbol,
                **{col: float(probabilities[i, q]) for q, col in enumerate(RANK_COLUMNS)},
                "RPS": float(scores[i]),
            })
        mean_rps = float(scores.mean())
        rps_rows.append({"round": round_number, "mean_RPS": mean_rps})
        delta = mean_rps - copula_rps[round_number]
        print(f"  round {round_number:02d}: RPS {mean_rps:.6f} | copula "
              f"{copula_rps[round_number]:.6f} | {delta:+.6f} "
              f"({'independent better' if delta < 0 else 'copula better'})")

    # --- aggregation --------------------------------------------------------
    rps_frame = pd.DataFrame(rps_rows).sort_values("round")
    overall = float(rps_frame["mean_RPS"].mean())
    copula_overall = float(np.mean([copula_rps[r] for r in EXPECTED_ROUNDS]))
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
    output["copula_RPS"] = output["round"].map(copula_rps)
    output["difference_vs_copula"] = (output["mean_RPS"] - output["copula_RPS"]).round(6)
    output["naive_RPS"] = NAIVE_RPS
    output["difference_vs_naive"] = (output["mean_RPS"] - NAIVE_RPS).round(6)
    output["mean_RPS"] = output["mean_RPS"].round(6)
    output["copula_RPS"] = output["copula_RPS"].round(6)
    output = pd.concat([
        output,
        pd.DataFrame([{
            "round": pd.NA, "mean_RPS": round(overall, 6),
            "copula_RPS": round(copula_overall, 6),
            "difference_vs_copula": round(overall - copula_overall, 6),
            "naive_RPS": NAIVE_RPS,
            "difference_vs_naive": round(overall - NAIVE_RPS, 6),
        }]),
    ], ignore_index=True)
    output["round"] = output["round"].astype("Int64")
    output["scope"] = ["round"] * len(rps_frame) + ["overall"]
    output = output[["scope", "round", "mean_RPS", "copula_RPS", "difference_vs_copula",
                     "naive_RPS", "difference_vs_naive"]]

    probabilities_frame.to_csv(PROBS_CSV, index=False)
    output.to_csv(RPS_CSV, index=False)

    # --- comparison ---------------------------------------------------------
    wins_vs_copula = int((rps_frame["mean_RPS"].to_numpy()
                          < np.array([copula_rps[r] for r in rps_frame["round"]])).sum())
    wins_vs_naive = int((rps_frame["mean_RPS"] < NAIVE_RPS).sum())

    print(f"\nOverall RPS (lower is better), {N_SCENARIOS} scenarios, seed {RANDOM_SEED}:")
    ranking = {
        "Naive equal-probability benchmark": NAIVE_RPS,
        "Chronos-2 120M + spatiotemporal copula": copula_overall,
        MODEL_LABEL: overall,
    }
    if MAIN_ROUND_RPS.is_file():
        existing = pd.read_csv(MAIN_ROUND_RPS).groupby("model")["mean_RPS"].mean()
        for model, value in existing.items():
            if "Naive" not in model:
                ranking[model] = float(value)
    for model, value in sorted(ranking.items(), key=lambda kv: kv[1]):
        print(f"  {model:<48} {value:.6f}")

    print(f"\nIndependent vs copula: {overall - copula_overall:+.6f} overall; "
          f"independent wins {wins_vs_copula}/12 rounds")
    print(f"Independent vs naive : {overall - NAIVE_RPS:+.6f} overall; "
          f"independent wins {wins_vs_naive}/12 rounds")
    print(f"\nRuntime {(datetime.now(timezone.utc) - started).total_seconds():.1f}s")
    for path in (PROBS_CSV, RPS_CSV):
        print(f"Wrote {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
