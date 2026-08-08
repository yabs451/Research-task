# Test 3 — Does a larger sample count improve Chronos-T5 Base's RPS?

Exploratory sensitivity check, run 2026-08-07.
Model: Chronos-T5 Base 200M (`amazon/chronos-t5-base`). Companion to Test 2,
which asked the same question of Financial Chronos.

## Files checked

All 24 experimental NPZ files were present and valid
(12 rounds x 2 sample counts, in
`300-500-sample-outputs-base`): expected shape (100, n, 20), the three
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

| Sample Count | Overall RPS | Difference vs 100 Samples | Difference vs Naive 0.160000 |
|---|---|---|---|
| 100 | 0.226899 | 0.000000 | 0.066899 |
| 300 | 0.226854 | -0.000045 | 0.066854 |
| 500 | 0.226920 | 0.000021 | 0.066920 |

Best configuration: **300 samples** (0.226854),
0.000045 better than the
100-sample baseline (0.226899), and better in 7 of
the 12 rounds.

### Step-by-step movement

| Step | Change in overall RPS | Mean absolute change per round |
|---|---|---|
| 100 → 300 | -0.000045 | 0.001041 |
| 300 → 500 | +0.000066 | 0.000483 |

### Per-round detail

| Round | 100 samples | 300 samples | 500 samples |
|---|---|---|---|
| 1 | 0.230891 | 0.230669 | 0.230765 |
| 2 | 0.231249 | 0.232902 | 0.232817 |
| 3 | 0.208048 | 0.207879 | 0.207071 |
| 4 | 0.230311 | 0.230176 | 0.229335 |
| 5 | 0.214248 | 0.216833 | 0.217845 |
| 6 | 0.234013 | 0.231594 | 0.232158 |
| 7 | 0.197714 | 0.198449 | 0.199183 |
| 8 | 0.185988 | 0.186696 | 0.186924 |
| 9 | 0.276527 | 0.275741 | 0.275320 |
| 10 | 0.183424 | 0.180836 | 0.181474 |
| 11 | 0.202942 | 0.202742 | 0.202764 |
| 12 | 0.327434 | 0.327730 | 0.327381 |

## Does sample count matter materially?

The whole 100→1000 range moves the overall RPS by 0.000066, against a
+0.066899 gap between this model and the naive
benchmark. Within the nested set (300/500/1000, one shared draw) the spread is
0.000066, which isolates the pure sample-size effect; the 100→300 step
also carries draw-to-draw noise because the 100-sample result came from a
separate run.

## Files

- `sample_count_test_base_300_500_results.csv`
- `sample_count_test_base_300_500_summary.md`
