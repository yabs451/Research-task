# Financial Chronos Small 46M 2021 Global - M6 RPS evaluation

Generated: 2026-08-06 15:51:38 UTC
Evaluator: `scripts/evaluate_m6_rps.py`

## Inputs

- Raw NPZ forecasts (unchanged, read-only): 12 files in
  `Results/Financial_Chronos_Small_46M_2021_Global/round_outputs`
  - `financial_chronos_small_2021_global_round01_samples.npz`
  - `financial_chronos_small_2021_global_round02_samples.npz`
  - `financial_chronos_small_2021_global_round03_samples.npz`
  - `financial_chronos_small_2021_global_round04_samples.npz`
  - `financial_chronos_small_2021_global_round05_samples.npz`
  - `financial_chronos_small_2021_global_round06_samples.npz`
  - `financial_chronos_small_2021_global_round07_samples.npz`
  - `financial_chronos_small_2021_global_round08_samples.npz`
  - `financial_chronos_small_2021_global_round09_samples.npz`
  - `financial_chronos_small_2021_global_round10_samples.npz`
  - `financial_chronos_small_2021_global_round11_samples.npz`
  - `financial_chronos_small_2021_global_round12_samples.npz`
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

**Final mean RPS: 0.179368** (naive equal-probability benchmark:
0.160000; difference +0.019368)

| Round | Model RPS | Naive RPS | Difference (model - naive) |
|---|---|---|---|
| 1 | 0.157357 | 0.160000 | -0.002643 |
| 2 | 0.157182 | 0.160000 | -0.002818 |
| 3 | 0.182473 | 0.160000 | 0.022473 |
| 4 | 0.177037 | 0.160000 | 0.017037 |
| 5 | 0.192773 | 0.160000 | 0.032773 |
| 6 | 0.190969 | 0.160000 | 0.030969 |
| 7 | 0.215305 | 0.160000 | 0.055305 |
| 8 | 0.186139 | 0.160000 | 0.026139 |
| 9 | 0.177572 | 0.160000 | 0.017572 |
| 10 | 0.168564 | 0.160000 | 0.008564 |
| 11 | 0.173373 | 0.160000 | 0.013373 |
| 12 | 0.173675 | 0.160000 | 0.013675 |

## DRE observation

| round | all_zero | min_daily_log_return | max_daily_log_return | min_four_week_return | median_four_week_return | max_four_week_return |
|---|---|---|---|---|---|---|
| 1 | False | -0.00728005 | 0.00693339 | -0.0287845 | -0.000822998 | 0.0215491 |
| 2 | False | -0.00932213 | 0.0108758 | -0.0151108 | 0.00319025 | 0.031161 |
| 3 | False | -0.00824697 | 0.00983583 | -0.0227382 | 0.00690884 | 0.0291682 |
| 4 | False | -0.010885 | 0.00785272 | -0.0199358 | 0.000855613 | 0.0228035 |
| 5 | False | -0.0130921 | 0.0108838 | -0.0206058 | -0.000591324 | 0.0115015 |
| 6 | False | -0.009404 | 0.0109055 | -0.0273567 | -0.000553019 | 0.0422785 |
| 7 | False | -0.00717992 | 0.00915242 | -0.0131678 | 1.69253e-09 | 0.0125443 |
| 8 | False | -0.011095 | 0.164496 | -0.00936254 | 0.00193143 | 0.180885 |
| 9 | False | -0.161925 | 0.161925 | -0.476753 | 3.01609e-09 | 0.382441 |
| 10 | False | -0.153425 | 0.153425 | -0.458656 | 2.85777e-09 | 0.847255 |
| 11 | False | -0.147214 | 0.147214 | -0.445038 | 2.74208e-09 | 0.158602 |
| 12 | False | -0.141867 | 0.141867 | -0.433041 | 2.64248e-09 | 0.152423 |

The results above use these raw forecasts unchanged. DRE-zeroed reference copies for rounds 9, 10, 11, 12 were written to `Results/Evaluation/derived_dre_adjusted/Financial_Chronos_Small_46M_2021_Global/` as separate derived artifacts and are not used in any result here.

## Validation

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

## Generated evaluation artifacts

- `Results/Evaluation/m6_ground_truth_quintiles.csv`
- `Results/Evaluation/predicted_quintile_probabilities_Financial_Chronos_Small_46M_2021_Global.csv`
- `Results/Evaluation/rps_by_asset_Financial_Chronos_Small_46M_2021_Global.csv`
- `Results/Evaluation/rps_by_round.csv`
- `Results/Evaluation/model_comparison_rps.csv`
- `Results/Evaluation/rps_round_comparison_long.csv`
- `Results/Evaluation/dre_raw_forecast_inspection.csv`
- `Results/Evaluation/m6_rps_evaluation_report.md`
