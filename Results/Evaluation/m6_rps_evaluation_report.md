# M6 quintile post-processing and RPS evaluation

Generated: 2026-08-06 15:51:38 UTC
Evaluator: `scripts/evaluate_m6_rps.py` (run started 2026-08-06 15:51:30 UTC)

## 1. Purpose

Convert the raw sampled trajectories produced by the inference stage into
M6-format quintile probability forecasts, build the realised M6 outcome
independently from the official price file, and score both models with the
Ranked Probability Score (RPS). Lower RPS is better.

## 2. Models evaluated

- **Chronos T5 Base 200M** - 12 raw NPZ rounds in `Results/Chronos_T5_Base_200M/round_outputs`
- **Financial Chronos Small 46M 2021 Global** - 12 raw NPZ rounds in `Results/Financial_Chronos_Small_46M_2021_Global/round_outputs`
- **Naive equal-probability benchmark** - the flat forecast [0.20, 0.20, 0.20, 0.20, 0.20] for every
  asset in every round, scored through exactly the same evaluator as a reference
  point. It is a validation benchmark, not a model.

## 3. Input artifacts

- Raw forecasts: one NPZ per model per round, each containing `forecast_samples`
  with shape (100 assets, 100 sampled trajectories, 20 forecast weekdays),
  `asset_symbols` and `forecast_dates`. The NPZ files are read-only research
  artifacts; the evaluator hashes them before and after the run.
- Realised prices: `Data/assets_m6.csv` (official M6 daily adjusted closes,
  100 symbols, 12 evaluation windows used here).
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

| Round | Chronos T5 Base 200M | Financial Chronos Small 46M 2021 Global | Naive equal-probability benchmark |
|---|---|---|---|
| 1 | 0.230891 | 0.157357 | 0.160000 |
| 2 | 0.231249 | 0.157182 | 0.160000 |
| 3 | 0.208048 | 0.182473 | 0.160000 |
| 4 | 0.230311 | 0.177037 | 0.160000 |
| 5 | 0.214248 | 0.192773 | 0.160000 |
| 6 | 0.234013 | 0.190969 | 0.160000 |
| 7 | 0.197714 | 0.215305 | 0.160000 |
| 8 | 0.185988 | 0.186139 | 0.160000 |
| 9 | 0.276527 | 0.177572 | 0.160000 |
| 10 | 0.183424 | 0.168564 | 0.160000 |
| 11 | 0.202942 | 0.173373 | 0.160000 |
| 12 | 0.327434 | 0.173675 | 0.160000 |

## 12. Overall comparison (lower is better)

| Model (lower RPS is better) | Round 1 | Round 2 | Round 3 | Round 4 | Round 5 | Round 6 | Round 7 | Round 8 | Round 9 | Round 10 | Round 11 | Round 12 | Overall Mean RPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Naive equal-probability benchmark | 0.160000 | 0.160000 | 0.160000 | 0.160000 | 0.160000 | 0.160000 | 0.160000 | 0.160000 | 0.160000 | 0.160000 | 0.160000 | 0.160000 | 0.160000 |
| Financial Chronos Small 46M 2021 Global | 0.157357 | 0.157182 | 0.182473 | 0.177037 | 0.192773 | 0.190969 | 0.215305 | 0.186139 | 0.177572 | 0.168564 | 0.173373 | 0.173675 | 0.179368 |
| Chronos T5 Base 200M | 0.230891 | 0.231249 | 0.208048 | 0.230311 | 0.214248 | 0.234013 | 0.197714 | 0.185988 | 0.276527 | 0.183424 | 0.202942 | 0.327434 | 0.226899 |

## 13. Naive equal-probability benchmark

The flat [0.2]*5 forecast is scored through the evaluator itself, not asserted to
be 0.16. Its computed value appears in the tables above; with one-hot targets
spread evenly over the five quintiles the theoretical value is 0.16, and any
deviation reflects the realised tie structure rather than an adjustment.

## 14. DRE raw-output inspection

DRE was acquired and stopped trading on 2022-10-03; its
official M6 price is carried forward afterwards, so its realised competition
return is exactly zero in the affected rounds. The raw model forecasts were
inspected, not modified (values shown with significant digits, so a genuinely
tiny non-zero forecast is not displayed as an exact zero):

| model | round | all_zero | min_daily_log_return | max_daily_log_return | min_four_week_return | median_four_week_return | max_four_week_return |
|---|---|---|---|---|---|---|---|
| Chronos T5 Base 200M | 1 | False | -0.0037267 | 0.0045067 | -0.0283635 | -0.00781271 | 0.0307983 |
| Chronos T5 Base 200M | 2 | False | -0.00318506 | 0.00489412 | -0.0181665 | 0.00167161 | 0.0183444 |
| Chronos T5 Base 200M | 3 | False | -0.00355603 | 0.00438829 | -0.031351 | -0.00986249 | 0.00767096 |
| Chronos T5 Base 200M | 4 | False | -0.00318774 | 0.00606448 | -0.0258614 | -0.00589155 | 0.0291852 |
| Chronos T5 Base 200M | 5 | False | -0.00370679 | 0.00536301 | -0.0246911 | -0.00871611 | 0.00887232 |
| Chronos T5 Base 200M | 6 | False | -0.00339809 | 0.00418834 | -0.024046 | -0.00779297 | 0.0157701 |
| Chronos T5 Base 200M | 7 | False | -0.00370831 | 0.00268261 | -0.0263158 | -0.00527236 | 0.0197599 |
| Chronos T5 Base 200M | 8 | False | -0.00345714 | 0.00257276 | -0.0265741 | -0.00956164 | 0.00855868 |
| Chronos T5 Base 200M | 9 | False | -0.00506511 | 0.0125836 | -0.0361323 | 3.01609e-09 | 0.0249998 |
| Chronos T5 Base 200M | 10 | False | -0.00194969 | 1.42888e-10 | -0.00194778 | 2.85777e-09 | 2.85777e-09 |
| Chronos T5 Base 200M | 11 | False | -0.00187076 | 1.37104e-10 | -0.001869 | 2.74208e-09 | 2.74208e-09 |
| Chronos T5 Base 200M | 12 | False | -0.00180281 | 1.32124e-10 | -0.00180118 | 2.64248e-09 | 2.64248e-09 |
| Financial Chronos Small 46M 2021 Global | 1 | False | -0.00728005 | 0.00693339 | -0.0287845 | -0.000822998 | 0.0215491 |
| Financial Chronos Small 46M 2021 Global | 2 | False | -0.00932213 | 0.0108758 | -0.0151108 | 0.00319025 | 0.031161 |
| Financial Chronos Small 46M 2021 Global | 3 | False | -0.00824697 | 0.00983583 | -0.0227382 | 0.00690884 | 0.0291682 |
| Financial Chronos Small 46M 2021 Global | 4 | False | -0.010885 | 0.00785272 | -0.0199358 | 0.000855613 | 0.0228035 |
| Financial Chronos Small 46M 2021 Global | 5 | False | -0.0130921 | 0.0108838 | -0.0206058 | -0.000591324 | 0.0115015 |
| Financial Chronos Small 46M 2021 Global | 6 | False | -0.009404 | 0.0109055 | -0.0273567 | -0.000553019 | 0.0422785 |
| Financial Chronos Small 46M 2021 Global | 7 | False | -0.00717992 | 0.00915242 | -0.0131678 | 1.69253e-09 | 0.0125443 |
| Financial Chronos Small 46M 2021 Global | 8 | False | -0.011095 | 0.164496 | -0.00936254 | 0.00193143 | 0.180885 |
| Financial Chronos Small 46M 2021 Global | 9 | False | -0.161925 | 0.161925 | -0.476753 | 3.01609e-09 | 0.382441 |
| Financial Chronos Small 46M 2021 Global | 10 | False | -0.153425 | 0.153425 | -0.458656 | 2.85777e-09 | 0.847255 |
| Financial Chronos Small 46M 2021 Global | 11 | False | -0.147214 | 0.147214 | -0.445038 | 2.74208e-09 | 0.158602 |
| Financial Chronos Small 46M 2021 Global | 12 | False | -0.141867 | 0.141867 | -0.433041 | 2.64248e-09 | 0.152423 |

The primary RPS results above use these raw forecasts exactly as generated.
Clearly labelled DRE-zeroed reference copies were written for rounds
9, 10, 11, 12 (origins already after the
acquisition) as separate derived artifacts; they are not used for any result in
this report. Round 8 is deliberately left alone: its forecast window crosses the
acquisition date and needs date-specific treatment.

- `Chronos_T5_Base_200M`: 4 DRE-zeroed reference copies for rounds 9, 10, 11, 12 in `Results/Evaluation/derived_dre_adjusted/Chronos_T5_Base_200M/`
- `Financial_Chronos_Small_46M_2021_Global`: 4 DRE-zeroed reference copies for rounds 9, 10, 11, 12 in `Results/Evaluation/derived_dre_adjusted/Financial_Chronos_Small_46M_2021_Global/`

## 15. Validation checks

- Ground truth: 1200 rows (12 rounds x 100 assets), all quintile rows sum to 1 (max |sum-1| = 0.00e+00)
- Ground-truth tied assets per round: {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0}; agreement with the faithful port of the official tie block: 1200/1200 rows
- Chronos_T5_Base_200M round 01: shape OK, finite, dates match schedule, universe matches ground truth, probabilities sum to 1 (max |sum-1| = 2.22e-16)
- Chronos_T5_Base_200M round 02: shape OK, finite, dates match schedule, universe matches ground truth, probabilities sum to 1 (max |sum-1| = 2.22e-16)
- Chronos_T5_Base_200M round 03: shape OK, finite, dates match schedule, universe matches ground truth, probabilities sum to 1 (max |sum-1| = 2.22e-16)
- Chronos_T5_Base_200M round 04: shape OK, finite, dates match schedule, universe matches ground truth, probabilities sum to 1 (max |sum-1| = 1.11e-16)
- Chronos_T5_Base_200M round 05: shape OK, finite, dates match schedule, universe matches ground truth, probabilities sum to 1 (max |sum-1| = 2.22e-16)
- Chronos_T5_Base_200M round 06: shape OK, finite, dates match schedule, universe matches ground truth, probabilities sum to 1 (max |sum-1| = 2.22e-16)
- Chronos_T5_Base_200M round 07: shape OK, finite, dates match schedule, universe matches ground truth, probabilities sum to 1 (max |sum-1| = 2.22e-16)
- Chronos_T5_Base_200M round 08: shape OK, finite, dates match schedule, universe matches ground truth, probabilities sum to 1 (max |sum-1| = 2.22e-16)
- Chronos_T5_Base_200M round 09: shape OK, finite, dates match schedule, universe matches ground truth, probabilities sum to 1 (max |sum-1| = 2.22e-16)
- Chronos_T5_Base_200M round 10: shape OK, finite, dates match schedule, universe matches ground truth, probabilities sum to 1 (max |sum-1| = 2.22e-16)
- Chronos_T5_Base_200M round 11: shape OK, finite, dates match schedule, universe matches ground truth, probabilities sum to 1 (max |sum-1| = 2.22e-16)
- Chronos_T5_Base_200M round 12: shape OK, finite, dates match schedule, universe matches ground truth, probabilities sum to 1 (max |sum-1| = 1.11e-16)
- Financial_Chronos_Small_46M_2021_Global round 01: shape OK, finite, dates match schedule, universe matches ground truth, probabilities sum to 1 (max |sum-1| = 2.22e-16)
- Financial_Chronos_Small_46M_2021_Global round 02: shape OK, finite, dates match schedule, universe matches ground truth, probabilities sum to 1 (max |sum-1| = 2.22e-16)
- Financial_Chronos_Small_46M_2021_Global round 03: shape OK, finite, dates match schedule, universe matches ground truth, probabilities sum to 1 (max |sum-1| = 2.22e-16)
- Financial_Chronos_Small_46M_2021_Global round 04: shape OK, finite, dates match schedule, universe matches ground truth, probabilities sum to 1 (max |sum-1| = 2.22e-16)
- Financial_Chronos_Small_46M_2021_Global round 05: shape OK, finite, dates match schedule, universe matches ground truth, probabilities sum to 1 (max |sum-1| = 2.22e-16)
- Financial_Chronos_Small_46M_2021_Global round 06: shape OK, finite, dates match schedule, universe matches ground truth, probabilities sum to 1 (max |sum-1| = 2.22e-16)
- Financial_Chronos_Small_46M_2021_Global round 07: shape OK, finite, dates match schedule, universe matches ground truth, probabilities sum to 1 (max |sum-1| = 2.22e-16)
- Financial_Chronos_Small_46M_2021_Global round 08: shape OK, finite, dates match schedule, universe matches ground truth, probabilities sum to 1 (max |sum-1| = 2.22e-16)
- Financial_Chronos_Small_46M_2021_Global round 09: shape OK, finite, dates match schedule, universe matches ground truth, probabilities sum to 1 (max |sum-1| = 2.22e-16)
- Financial_Chronos_Small_46M_2021_Global round 10: shape OK, finite, dates match schedule, universe matches ground truth, probabilities sum to 1 (max |sum-1| = 2.22e-16)
- Financial_Chronos_Small_46M_2021_Global round 11: shape OK, finite, dates match schedule, universe matches ground truth, probabilities sum to 1 (max |sum-1| = 2.22e-16)
- Financial_Chronos_Small_46M_2021_Global round 12: shape OK, finite, dates match schedule, universe matches ground truth, probabilities sum to 1 (max |sum-1| = 2.22e-16)
- For every model the mean of the 12 round means equals the mean of all 1,200 asset-round scores (max difference 2.78e-17)
- Every round mean equals the mean of that round's 100 asset RPS values
- All 24 raw NPZ files are byte-identical before and after the run (SHA-256 verified); they were opened read-only

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

| model | mean_max_probability | mean_probability_entropy | dispersion_ratio |
|---|---|---|---|
| Chronos T5 Base 200M | 0.529 | 1.064 | 1.967 |
| Financial Chronos Small 46M 2021 Global | 0.396 | 1.386 | 0.565 |

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

- `Results/Evaluation/derived_dre_adjusted/Chronos_T5_Base_200M/chronos_t5_base_round09_samples_dre_zeroed.npz`
- `Results/Evaluation/derived_dre_adjusted/Chronos_T5_Base_200M/chronos_t5_base_round10_samples_dre_zeroed.npz`
- `Results/Evaluation/derived_dre_adjusted/Chronos_T5_Base_200M/chronos_t5_base_round11_samples_dre_zeroed.npz`
- `Results/Evaluation/derived_dre_adjusted/Chronos_T5_Base_200M/chronos_t5_base_round12_samples_dre_zeroed.npz`
- `Results/Evaluation/derived_dre_adjusted/Financial_Chronos_Small_46M_2021_Global/financial_chronos_small_2021_global_round09_samples_dre_zeroed.npz`
- `Results/Evaluation/derived_dre_adjusted/Financial_Chronos_Small_46M_2021_Global/financial_chronos_small_2021_global_round10_samples_dre_zeroed.npz`
- `Results/Evaluation/derived_dre_adjusted/Financial_Chronos_Small_46M_2021_Global/financial_chronos_small_2021_global_round11_samples_dre_zeroed.npz`
- `Results/Evaluation/derived_dre_adjusted/Financial_Chronos_Small_46M_2021_Global/financial_chronos_small_2021_global_round12_samples_dre_zeroed.npz`
- `Results/Evaluation/dre_raw_forecast_inspection.csv`
- `Results/Evaluation/forecast_dispersion_diagnostics.csv`
- `Results/Evaluation/m6_ground_truth_quintiles.csv`
- `Results/Evaluation/model_comparison_rps.csv`
- `Results/Evaluation/predicted_quintile_probabilities_Chronos_T5_Base_200M.csv`
- `Results/Evaluation/predicted_quintile_probabilities_Financial_Chronos_Small_46M_2021_Global.csv`
- `Results/Evaluation/rps_by_asset_Chronos_T5_Base_200M.csv`
- `Results/Evaluation/rps_by_asset_Financial_Chronos_Small_46M_2021_Global.csv`
- `Results/Evaluation/rps_by_round.csv`
- `Results/Evaluation/rps_round_comparison_long.csv`
