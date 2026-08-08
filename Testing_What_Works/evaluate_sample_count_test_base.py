"""Test 3 - does a larger sample count improve Chronos-T5 Base's RPS?

The Chronos Base counterpart of evaluate_sample_count_test.py (which did this
for Financial Chronos). It scores the 300/500/1000-sample Base experiment
against the same ground truth and the same RPS methodology as the primary
evaluation, and compares with the existing 100-sample Base result.

Discovery, per-file validation and scoring are IMPORTED from
evaluate_sample_count_test.py, which in turn imports the M6 methodology from
scripts/evaluate_m6_rps.py - nothing about the scoring is reimplemented here.
Only the experiment folder, the baseline model label and the output filenames
differ, so the Financial experiment's own inputs and outputs are untouched.

Reads the experimental NPZ files read-only, writes only into Testing_What_Works/,
and never touches Results/.

Run:  python Testing_What_Works/evaluate_sample_count_test_base.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

TEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEST_DIR.parent
sys.path.insert(0, str(TEST_DIR))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from evaluate_sample_count_test import (  # noqa: E402
    BASELINE_SAMPLES, MAIN_ROUND_RPS, NAIVE_RPS, SCHEDULE_PATH,
    discover_experiment_files, score_configuration, validate_file,
)
from evaluate_m6_rps import (  # noqa: E402
    EXPECTED_ROUNDS, build_ground_truth, load_prices,
)

MODEL_LABEL = "Chronos T5 Base 200M"
# Cross-check only: the primary 100-sample Base result recorded in Results/.
EXPECTED_BASELINE_OVERALL = 0.226899

# Two Base sample-count experiments are prepared. The 300/500 run is the one
# currently intended to be executed (compute constraints); the 300/500/1000 run
# remains available. Whichever folder actually holds NPZ files is scored, and
# each writes its own output filenames so neither can overwrite the other.
EXPERIMENTS = {
    "300-500-sample-outputs-base": [300, 500],
    "300-500-1000-sample-outputs-base": [300, 500, 1000],
}


def select_experiment(name: str | None = None) -> tuple[Path, list[int]]:
    """Pick the Base experiment folder to score, failing clearly if ambiguous."""
    if name is not None:
        if name not in EXPERIMENTS:
            raise ValueError(f"Unknown experiment '{name}'; choose from {list(EXPERIMENTS)}")
        return TEST_DIR / name, EXPERIMENTS[name]

    populated = [n for n in EXPERIMENTS
                 if (TEST_DIR / n).is_dir() and any((TEST_DIR / n).rglob("*.npz"))]
    if not populated:
        raise FileNotFoundError(
            "No Base sample-count NPZ files found. Copy the files produced by the "
            "notebook into one of:\n  "
            + "\n  ".join(str((TEST_DIR / n).relative_to(REPO_ROOT)) for n in EXPERIMENTS)
        )
    if len(populated) > 1:
        raise ValueError(
            f"Both Base experiments contain files ({populated}); pass one explicitly, "
            f"e.g. python {Path(__file__).name} {populated[0]}"
        )
    return TEST_DIR / populated[0], EXPERIMENTS[populated[0]]


def output_paths(counts: list[int]) -> tuple[Path, Path]:
    tag = "_".join(str(c) for c in counts)
    return (TEST_DIR / f"sample_count_test_base_{tag}_results.csv",
            TEST_DIR / f"sample_count_test_base_{tag}_summary.md")


def load_baseline_round_rps() -> dict[int, float]:
    """The existing 100-sample Base per-round results, from the primary outputs."""
    if not MAIN_ROUND_RPS.is_file():
        raise FileNotFoundError(
            f"Baseline results not found at {MAIN_ROUND_RPS}. "
            "Run scripts/evaluate_m6_rps.py first."
        )
    frame = pd.read_csv(MAIN_ROUND_RPS)
    if MODEL_LABEL not in set(frame["model"]):
        raise ValueError(
            f"'{MODEL_LABEL}' not found in {MAIN_ROUND_RPS.name}; "
            f"available models: {sorted(set(frame['model']))}"
        )
    baseline = frame[frame["model"] == MODEL_LABEL]
    if sorted(baseline["round"]) != EXPECTED_ROUNDS:
        raise ValueError(f"Baseline is missing rounds: found {sorted(baseline['round'])}")
    return dict(zip(baseline["round"], baseline["mean_RPS"]))


def main() -> None:
    experiment_dir, expected_counts = select_experiment(
        sys.argv[1] if len(sys.argv) > 1 else None)
    results_csv, summary_md = output_paths(expected_counts)
    n_expected = len(expected_counts) * len(EXPECTED_ROUNDS)
    print(f"Experiment: {experiment_dir.name}  (sample counts {expected_counts})")

    schedule = pd.read_csv(SCHEDULE_PATH)
    prices = load_prices()
    ground_truth = build_ground_truth(prices, schedule)
    official_order = list(pd.read_csv(
        REPO_ROOT / "Data" / "processed" / "rolling_origins" / "round_01_context.csv",
        nrows=0).columns[1:])

    files = discover_experiment_files(experiment_dir, expected_counts)
    print(f"Found {n_expected} experimental NPZ files in "
          f"{experiment_dir.relative_to(REPO_ROOT)}")
    for num_samples, rounds in files.items():
        for round_number, path in rounds.items():
            validate_file(path, round_number, num_samples, schedule, official_order)
        print(f"  {num_samples:>4} samples: 12/12 rounds valid (100, {num_samples}, 20), "
              "finite, official asset order, dates match the round schedule")

    # --- score each configuration ------------------------------------------
    by_config = {BASELINE_SAMPLES: load_baseline_round_rps()}
    baseline_overall = float(np.mean(list(by_config[BASELINE_SAMPLES].values())))
    if not np.isclose(baseline_overall, EXPECTED_BASELINE_OVERALL, atol=5e-6):
        raise ValueError(
            f"Baseline overall RPS {baseline_overall:.6f} does not match the recorded "
            f"primary Base result {EXPECTED_BASELINE_OVERALL}"
        )
    print(f"\nBaseline: {BASELINE_SAMPLES} samples = {baseline_overall:.6f}, read from "
          f"{MAIN_ROUND_RPS.relative_to(REPO_ROOT)}")

    for num_samples, rounds in files.items():
        by_config[num_samples] = score_configuration(rounds, ground_truth)
        print(f"  scored {num_samples} samples: overall "
              f"{np.mean(list(by_config[num_samples].values())):.6f}")

    # --- comparison table ---------------------------------------------------
    rows = []
    for num_samples in sorted(by_config):
        round_rps = by_config[num_samples]
        overall = float(np.mean([round_rps[r] for r in EXPECTED_ROUNDS]))
        row = {"Sample Count": num_samples}
        row.update({f"Round {r} RPS": round(round_rps[r], 6) for r in EXPECTED_ROUNDS})
        row["Overall RPS"] = round(overall, 6)
        row["Difference vs 100 Samples"] = round(overall - baseline_overall, 6)
        row["Difference vs Naive 0.160000"] = round(overall - NAIVE_RPS, 6)
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(results_csv, index=False)

    overall_by_config = {int(r["Sample Count"]): r["Overall RPS"] for r in rows}
    best = min(overall_by_config, key=overall_by_config.get)
    write_summary(table, overall_by_config, best, baseline_overall, by_config,
                  experiment_dir, results_csv, summary_md)

    print("\nOverall RPS (lower is better):")
    for num_samples in sorted(overall_by_config):
        mark = "  <-- best" if num_samples == best else ""
        print(f"  {num_samples:>4} samples: {overall_by_config[num_samples]:.6f} "
              f"({overall_by_config[num_samples] - baseline_overall:+.6f} vs 100){mark}")
    print(f"\nWrote {results_csv.relative_to(REPO_ROOT)}")
    print(f"Wrote {summary_md.relative_to(REPO_ROOT)}")


def write_summary(table, overall_by_config, best, baseline_overall, by_config,
                  experiment_dir, results_csv, summary_md) -> None:
    counts = sorted(overall_by_config)
    spread = max(overall_by_config.values()) - min(overall_by_config.values())
    improvement = overall_by_config[best] - baseline_overall

    steps = []
    for a, b in zip(counts, counts[1:]):
        mean_abs = float(np.mean([abs(by_config[b][r] - by_config[a][r])
                                  for r in EXPECTED_ROUNDS]))
        steps.append((a, b, overall_by_config[b] - overall_by_config[a], mean_abs))
    step_lines = "\n".join(
        f"| {a} → {b} | {delta:+.6f} | {mean_abs:.6f} |" for a, b, delta, mean_abs in steps
    )

    nested = [n for n in counts if n != BASELINE_SAMPLES]
    nested_spread = (max(overall_by_config[n] for n in nested)
                     - min(overall_by_config[n] for n in nested))
    rounds_improved = sum(1 for r in EXPECTED_ROUNDS
                          if by_config[best][r] < by_config[BASELINE_SAMPLES][r])

    compare_cols = ["Sample Count", "Overall RPS", "Difference vs 100 Samples",
                    "Difference vs Naive 0.160000"]
    header = "| " + " | ".join(compare_cols) + " |\n|" + "|".join(["---"] * len(compare_cols)) + "|"
    body = "\n".join(
        "| " + " | ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in row) + " |"
        for row in table[compare_cols].itertuples(index=False)
    )
    per_round = ("| Round | " + " | ".join(f"{n} samples" for n in counts) + " |\n|"
                 + "|".join(["---"] * (len(counts) + 1)) + "|\n"
                 + "\n".join(f"| {r} | " + " | ".join(f"{by_config[n][r]:.6f}" for n in counts)
                             + " |" for r in EXPECTED_ROUNDS))

    text = f"""# Test 3 — Does a larger sample count improve Chronos-T5 Base's RPS?

Exploratory sensitivity check, run {datetime.now(timezone.utc).strftime('%Y-%m-%d')}.
Model: Chronos-T5 Base 200M (`amazon/chronos-t5-base`). Companion to Test 2,
which asked the same question of Financial Chronos.

## Files checked

All {12 * (len(counts) - 1)} experimental NPZ files were present and valid
(12 rounds x {len(counts) - 1} sample counts, in
`{experiment_dir.name}`): expected shape (100, n, 20), the three
required arrays, official M6 asset ordering, forecast dates matching the round
schedule, finite values throughout, no missing or duplicate rounds.

## Method

Scored with the primary evaluator's own functions, imported from
`scripts/evaluate_m6_rps.py` via `evaluate_sample_count_test.py` — same four-week
return conversion (`exp(sum of 20 log returns) - 1`), same cross-sectional
ranking, same tie-aware quintile construction, same official ground truth from
`Data/assets_m6.csv`, same RPS formula and aggregation. DRE was left exactly as
generated. The 100-sample baseline is read from the existing
`Results/Evaluation/rps_by_round.csv` rather than recomputed.

The 300/500/1000 configurations are nested prefixes of one 1000-sample draw per
round, so differences among them reflect the extra trajectories rather than a
different random draw. The 100-sample baseline was a separate earlier run, so
100 → 1000 is **not** a perfectly controlled nested comparison.

## Result (lower RPS is better)

{header}
{body}

Best configuration: **{best} samples** ({overall_by_config[best]:.6f}),
{abs(improvement):.6f} {'better' if improvement < 0 else 'worse'} than the
100-sample baseline ({baseline_overall:.6f}), and better in {rounds_improved} of
the 12 rounds.

### Step-by-step movement

| Step | Change in overall RPS | Mean absolute change per round |
|---|---|---|
{step_lines}

### Per-round detail

{per_round}

## Does sample count matter materially?

The whole 100→1000 range moves the overall RPS by {spread:.6f}, against a
{baseline_overall - NAIVE_RPS:+.6f} gap between this model and the naive
benchmark. Within the nested set (300/500/1000, one shared draw) the spread is
{nested_spread:.6f}, which isolates the pure sample-size effect; the 100→300 step
also carries draw-to-draw noise because the 100-sample result came from a
separate run.

## Files

- `{results_csv.name}`
- `{summary_md.name}`
"""
    summary_md.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
