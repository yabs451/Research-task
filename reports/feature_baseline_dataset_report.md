# Feature-Based Baseline Dataset Report (Stage 6B)

Generated: 2026-08-21 09:20:54 UTC

## 1. Classifier origin grid

- 170 four-week-spaced Friday forecast origins, 2010-01-22 .. 2023-01-06.
- Every step is exactly 28 calendar days = 20 shared weekdays. No origin is generated on any other weekday, so no two supervised rows share an overlapping four-week target window.
- Roles: 146 historical, 12 tuning, 12 M6.
- Tuning origins: 2021-03-05, 2021-04-02, 2021-04-30, 2021-05-28, 2021-06-25, 2021-07-23, 2021-08-20, 2021-09-17, 2021-10-15, 2021-11-12, 2021-12-10, 2022-01-07
- M6 origins: 2022-03-04, 2022-04-01, 2022-04-29, 2022-05-27, 2022-06-24, 2022-07-22, 2022-08-19, 2022-09-16, 2022-10-14, 2022-11-11, 2022-12-09, 2023-01-06 (identical to Data/metadata/m6_round_schedule.csv).

## 2. Row counts per variant

| variant | rows | origins | rows with complete features | rows with a target | trainable rows |
|---|---|---|---|---|---|
| no_knn | 15363 | 170 | 14739 | 15363 | 14739 |
| knn | 15363 | 170 | 14842 | 15363 | 14842 |

- Historical label agreement where BOTH branches produce a label: 0.9974 over 15363 shared rows. The branches are deliberately not forced to match.

## 3. Missing predictor values per variant

| feature | no_knn missing | knn missing |
|---|---|---|
| feat_ret_4w_recent | 24 | 0 |
| feat_ret_4w_seasonal_11m | 284 | 0 |
| feat_vol_3m | 73 | 0 |
| feat_max_ret_3m | 73 | 0 |
| feat_dollar_volume_2m | 382 | 382 |
| feat_abs_ret_to_volume_3m | 521 | 521 |
| feat_rank1_freq_4w | 24 | 0 |
| feat_rank2_freq_4w | 24 | 0 |
| feat_rank4_freq_4w | 24 | 0 |
| feat_rank5_freq_4w | 24 | 0 |

## 4. Volume treatment

- Asset-days with nonpositive dollar volume (log undefined): 2122 of 328020 (0.6469%), concentrated in 13 assets, overwhelmingly the London-listed ETFs plus DRE's forward-filled post-acquisition tail.
- Treatment: the released Samartzis behaviour exactly. `log(volume*close/1000)` is undefined on such a day, so the 40-day rolling SUM of any window containing one is undefined too (`.replace([inf,-inf], nan)` in the released code), and the 60-day absolute-return-to-volume sum is strict, so one undefined denominator removes it as well. A single zero-volume day therefore removes up to 40 consecutive aggregates and up to 99 consecutive ratio values.
- These two features are the only ones affected; the aggregate also stays undefined until 40 weekdays have elapsed since inception, which falls out of the same rule.
- Raw Dataset D is not modified in any way.

## 5. Feature-window numerical spot checks

Each feature was recomputed from first principles, in explicit origin coordinates, for randomly sampled rows:

| feature | released name | window relative to origin t | independent recomputations |
|---|---|---|---|
| feat_ret_4w_recent | feat_0 | sum of daily log returns over weekdays [t-19, t]  ==  log(P_t / P_t-20) | 60 |
| feat_ret_4w_seasonal_11m | feat_6 | sum of daily log returns over weekdays [t-219, t-200] | 60 |
| feat_vol_3m | feat_1 | sample std (ddof=1) of daily log returns over [t-59, t] | 60 |
| feat_max_ret_3m | feat_2 | max daily log return over [t-59, t] | 60 |
| feat_dollar_volume_2m | feat_3 | sum over [t-39, t] of log(raw volume * RAW close / 1000) | 60 |
| feat_abs_ret_to_volume_3m | feat_4 | sum over j in [t-59, t] of |r_j| / D40_j, where D40_j is the 40-day log-dollar-volume aggregate ending at j | 60 |
| feat_rank1_freq_4w | feat_Rank1 | fraction of weekdays in [t-19, t] on which the asset sat in the LOWEST cross-sectional daily-return quintile | 60 |
| feat_rank2_freq_4w | feat_Rank2 | as above, quintile 2 | 60 |
| feat_rank4_freq_4w | feat_Rank4 | as above, quintile 4 | 60 |
| feat_rank5_freq_4w | feat_Rank5 | as above, quintile 5 (highest) | 60 |

## 6. KNN causality

- Leakage test origin: 2021-03-05
- Predictors: identical under post-origin corruption
- Historical target: identical under post-target_end corruption; changes under in-window corruption

## 7. Evaluation separation

- Every target column in these files is prefixed `hist_target_` and is a BRANCH-SPECIFIC historical supervised label built from Dataset D.
- The official M6 evaluation ground truth is NOT defined here. It remains `scripts/evaluate_m6_rps.py` built from `Data/assets_m6.csv`, which this stage neither reads for labels nor modifies.
