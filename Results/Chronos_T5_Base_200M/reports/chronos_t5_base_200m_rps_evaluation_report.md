# Chronos T5 Base 200M - M6 RPS evaluation

Generated: 2026-08-06 15:51:38 UTC
Evaluator: `scripts/evaluate_m6_rps.py`

## Inputs

- Raw NPZ forecasts (unchanged, read-only): 12 files in
  `Results/Chronos_T5_Base_200M/round_outputs`
  - `chronos_t5_base_round01_samples.npz`
  - `chronos_t5_base_round02_samples.npz`
  - `chronos_t5_base_round03_samples.npz`
  - `chronos_t5_base_round04_samples.npz`
  - `chronos_t5_base_round05_samples.npz`
  - `chronos_t5_base_round06_samples.npz`
  - `chronos_t5_base_round07_samples.npz`
  - `chronos_t5_base_round08_samples.npz`
  - `chronos_t5_base_round09_samples.npz`
  - `chronos_t5_base_round10_samples.npz`
  - `chronos_t5_base_round11_samples.npz`
  - `chronos_t5_base_round12_samples.npz`
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

**Final mean RPS: 0.226899** (naive equal-probability benchmark:
0.160000; difference +0.066899)

| Round | Model RPS | Naive RPS | Difference (model - naive) |
|---|---|---|---|
| 1 | 0.230891 | 0.160000 | 0.070891 |
| 2 | 0.231249 | 0.160000 | 0.071249 |
| 3 | 0.208048 | 0.160000 | 0.048048 |
| 4 | 0.230311 | 0.160000 | 0.070311 |
| 5 | 0.214248 | 0.160000 | 0.054248 |
| 6 | 0.234013 | 0.160000 | 0.074013 |
| 7 | 0.197714 | 0.160000 | 0.037714 |
| 8 | 0.185988 | 0.160000 | 0.025988 |
| 9 | 0.276527 | 0.160000 | 0.116527 |
| 10 | 0.183424 | 0.160000 | 0.023424 |
| 11 | 0.202942 | 0.160000 | 0.042942 |
| 12 | 0.327434 | 0.160000 | 0.167434 |

## DRE observation

| round | all_zero | min_daily_log_return | max_daily_log_return | min_four_week_return | median_four_week_return | max_four_week_return |
|---|---|---|---|---|---|---|
| 1 | False | -0.0037267 | 0.0045067 | -0.0283635 | -0.00781271 | 0.0307983 |
| 2 | False | -0.00318506 | 0.00489412 | -0.0181665 | 0.00167161 | 0.0183444 |
| 3 | False | -0.00355603 | 0.00438829 | -0.031351 | -0.00986249 | 0.00767096 |
| 4 | False | -0.00318774 | 0.00606448 | -0.0258614 | -0.00589155 | 0.0291852 |
| 5 | False | -0.00370679 | 0.00536301 | -0.0246911 | -0.00871611 | 0.00887232 |
| 6 | False | -0.00339809 | 0.00418834 | -0.024046 | -0.00779297 | 0.0157701 |
| 7 | False | -0.00370831 | 0.00268261 | -0.0263158 | -0.00527236 | 0.0197599 |
| 8 | False | -0.00345714 | 0.00257276 | -0.0265741 | -0.00956164 | 0.00855868 |
| 9 | False | -0.00506511 | 0.0125836 | -0.0361323 | 3.01609e-09 | 0.0249998 |
| 10 | False | -0.00194969 | 1.42888e-10 | -0.00194778 | 2.85777e-09 | 2.85777e-09 |
| 11 | False | -0.00187076 | 1.37104e-10 | -0.001869 | 2.74208e-09 | 2.74208e-09 |
| 12 | False | -0.00180281 | 1.32124e-10 | -0.00180118 | 2.64248e-09 | 2.64248e-09 |

The results above use these raw forecasts unchanged. DRE-zeroed reference copies for rounds 9, 10, 11, 12 were written to `Results/Evaluation/derived_dre_adjusted/Chronos_T5_Base_200M/` as separate derived artifacts and are not used in any result here.

## Validation

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

## Generated evaluation artifacts

- `Results/Evaluation/m6_ground_truth_quintiles.csv`
- `Results/Evaluation/predicted_quintile_probabilities_Chronos_T5_Base_200M.csv`
- `Results/Evaluation/rps_by_asset_Chronos_T5_Base_200M.csv`
- `Results/Evaluation/rps_by_round.csv`
- `Results/Evaluation/model_comparison_rps.csv`
- `Results/Evaluation/rps_round_comparison_long.csv`
- `Results/Evaluation/dre_raw_forecast_inspection.csv`
- `Results/Evaluation/m6_rps_evaluation_report.md`
