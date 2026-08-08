# Test 2 — Does a larger sample count improve the RPS?

Exploratory sensitivity check, run 2026-08-07.
Model: Financial Chronos Small 46M (FinText 2021 Global). **Outcome: a small,
consistent improvement that plateaus by 300 samples and is far too small to
matter — the RPS stays roughly 0.019 above the naive
benchmark at every sample count.**

## Files checked

All 36 experimental NPZ files were present and valid: **True**
(12 rounds x 3 sample counts, in `300-500-1000-sample-outputs-fintext`).
Each file was checked for the expected shape (100, n, 20),
the presence of `forecast_samples`, `asset_symbols` and `forecast_dates`, the
official M6 asset ordering, forecast dates matching the round schedule, and
finite values throughout. No round was missing or duplicated.

## Method

Scored with the primary evaluator's own functions, imported from
`scripts/evaluate_m6_rps.py` — same four-week return conversion
(`exp(sum of 20 log returns) - 1`), same cross-sectional ranking, same tie-aware
quintile construction, same official ground truth from `Data/assets_m6.csv`, same
RPS formula and aggregation. Nothing about the methodology was changed and DRE
was left exactly as generated. The 100-sample baseline is read from the existing
`Results/Evaluation/rps_by_round.csv` rather than recomputed.

The 300/500/1000 configurations are nested prefixes of one 1000-sample draw per
round, so differences between them reflect the extra trajectories rather than a
different random draw. The 100-sample baseline was a separate earlier draw.

## Result (lower RPS is better)

| Sample Count | Overall RPS | Difference vs 100 Samples | Difference vs Naive 0.160000 |
|---|---|---|---|
| 100 | 0.179368 | 0.000000 | 0.019368 |
| 300 | 0.177916 | -0.001453 | 0.017916 |
| 500 | 0.177947 | -0.001421 | 0.017947 |
| 1000 | 0.177624 | -0.001744 | 0.017624 |

Best configuration: **1000 samples** (0.177624),
0.001744 better than the
100-sample baseline (0.179368).

### Step-by-step movement

| Step | Change in overall RPS | Mean absolute change per round |
|---|---|---|
| 100 → 300 | -0.001452 | 0.001673 |
| 300 → 500 | +0.000031 | 0.000366 |
| 500 → 1000 | -0.000323 | 0.000607 |

### Per-round detail

| Round | 100 samples | 300 samples | 500 samples | 1000 samples |
|---|---|---|---|---|
| 1 | 0.157357 | 0.155732 | 0.155646 | 0.155374 |
| 2 | 0.157182 | 0.158505 | 0.158678 | 0.158759 |
| 3 | 0.182473 | 0.181002 | 0.181261 | 0.180610 |
| 4 | 0.177037 | 0.175069 | 0.173938 | 0.173429 |
| 5 | 0.192773 | 0.189776 | 0.189838 | 0.190178 |
| 6 | 0.190969 | 0.189867 | 0.190247 | 0.188760 |
| 7 | 0.215305 | 0.213689 | 0.213594 | 0.212560 |
| 8 | 0.186139 | 0.185005 | 0.185238 | 0.185002 |
| 9 | 0.177572 | 0.176163 | 0.175762 | 0.175265 |
| 10 | 0.168564 | 0.166450 | 0.166153 | 0.166688 |
| 11 | 0.173373 | 0.172778 | 0.173300 | 0.172409 |
| 12 | 0.173675 | 0.170952 | 0.171709 | 0.172455 |

## Does sample count matter materially?

No. The whole 100→1000 range moves the overall RPS by only 0.001744, negligible next to the +0.019368 gap between this model and the naive benchmark.

The direction is consistent, though, and worth stating precisely: the best configuration improves on the baseline in 11 of the 12 rounds. But that improvement should not all be credited to the larger sample. The 100→300 step (-0.001452) mixes two things, because the 100-sample result came from a separate earlier draw. The 300/500/1000 results are nested prefixes of one draw, so their spread isolates the pure sample-size effect — and that is just 0.000323, an order of magnitude smaller than the distance to the benchmark. Sample count is not what is holding this model back.

## Do the results stabilise?

Yes. Almost all of the movement happens in the first step: after 300 samples the
overall RPS barely shifts (+0.000031 from 300 to 500, -0.000323
from 500 to 1000), and the mean absolute per-round movement
drops sharply after the first step and stays small
(see the step table). That is the behaviour expected from Monte Carlo error:
each configuration re-estimates the same quintile probabilities with more draws,
so the estimates settle rather than trend. Practically, 300 trajectories already
resolve these probabilities about as well as 1000; the overall RPS stays inside a
band of 0.001744 across the entire 100→1000 range and never approaches the
0.160000 naive benchmark.

## Conclusion

Increasing the sample count from 100 to 1000 improves Financial Chronos's RPS by
0.001744 — real and consistently signed, but roughly a tenth of the
distance to the naive benchmark, and essentially exhausted by 300 samples. The
primary evaluation keeps the 100-sample result (0.179368); this test
is exploratory and does not replace it. The model's gap to the benchmark is a
property of its forecasts, not of how finely they were sampled — so effort is
better spent on the forecasts themselves than on more trajectories. If a future
run wants marginally tighter probability estimates at negligible cost, 300
samples is the sensible setting.
