# Chronos-T5 Base (200M) - M6 Rounds 2-12 Inference Report

Generated: 2026-08-06 09:23:21 UTC
Run started: 2026-08-06 09:15:03 UTC

## Experiment settings (identical for every round)

| Setting | Value |
|---|---|
| Model | `amazon/chronos-t5-base` (Chronos-T5 Base, 200M parameters) |
| Context length | 512 weekday daily log returns |
| Prediction length | 20 weekdays |
| Sampled trajectories | 100 per asset |
| Series batch size | 10 (GPU efficiency only) |
| Random seed | 42 (reset at the start of every round) |
| Assets | 100 official M6 assets, official order |
| Runtime | Google Colab GPU (NVIDIA L4), dtype torch.bfloat16 |
| Versions | chronos-forecasting 2.3.1, torch 2.11.0+cu128 |

- The model was **loaded once** before the round loop and reused for every round;
  it was never reloaded, trained or fine-tuned (`MODEL_LOADED_ONCE = True`).
- Chronos is univariate: each of the 100 assets was forecast as its own
  independent series. Batching of 10 assets per forward pass is a
  speed/memory measure only and cannot let one asset's history inform another's forecast.
- Contexts were taken unchanged from the Stage 3 files
  `round_XX_context.csv`, each verified byte-for-byte against the repository
  SHA-256 digests before inference. No preprocessing was repeated: no normalising,
  standardising, smoothing, clipping, averaging, outlier removal or NaN-filling.
  OGN's genuine leading missing values were left as NaN and handled by Chronos's
  missing-value mask.

## Per-round results

| Round | Origin (context end) | Forecast range | Output shape | Finite check | NPZ file |
|---|---|---|---|---|---|
| 02 | 2022-04-01 | 2022-04-04 .. 2022-04-29 | 100 x 100 x 20 | all finite | `chronos_t5_base_round02_samples.npz` |
| 03 | 2022-04-29 | 2022-05-02 .. 2022-05-27 | 100 x 100 x 20 | all finite | `chronos_t5_base_round03_samples.npz` |
| 04 | 2022-05-27 | 2022-05-30 .. 2022-06-24 | 100 x 100 x 20 | all finite | `chronos_t5_base_round04_samples.npz` |
| 05 | 2022-06-24 | 2022-06-27 .. 2022-07-22 | 100 x 100 x 20 | all finite | `chronos_t5_base_round05_samples.npz` |
| 06 | 2022-07-22 | 2022-07-25 .. 2022-08-19 | 100 x 100 x 20 | all finite | `chronos_t5_base_round06_samples.npz` |
| 07 | 2022-08-19 | 2022-08-22 .. 2022-09-16 | 100 x 100 x 20 | all finite | `chronos_t5_base_round07_samples.npz` |
| 08 | 2022-09-16 | 2022-09-19 .. 2022-10-14 | 100 x 100 x 20 | all finite | `chronos_t5_base_round08_samples.npz` |
| 09 | 2022-10-14 | 2022-10-17 .. 2022-11-11 | 100 x 100 x 20 | all finite | `chronos_t5_base_round09_samples.npz` |
| 10 | 2022-11-11 | 2022-11-14 .. 2022-12-09 | 100 x 100 x 20 | all finite | `chronos_t5_base_round10_samples.npz` |
| 11 | 2022-12-09 | 2022-12-12 .. 2023-01-06 | 100 x 100 x 20 | all finite | `chronos_t5_base_round11_samples.npz` |
| 12 | 2023-01-06 | 2023-01-09 .. 2023-02-03 | 100 x 100 x 20 | all finite | `chronos_t5_base_round12_samples.npz` |

- All 11 rounds (2-12) completed successfully.

Each NPZ contains exactly `forecast_samples` (assets x sampled trajectories x
forecast weekdays), `asset_symbols` and `forecast_dates`. Every file was saved
immediately after its round finished, then reloaded and confirmed to open with the
correct shape, unchanged asset ordering, correct forecast dates and finite values.

## Context provenance (SHA-256)

- Round 02: `4b1854a641bb0229231d151301c8294c3f4cdf313dabfebff6d61e7a7d7e50fa`
- Round 03: `efef3a5bf1b1f4504761e6899d83cb26e62e097ad29e8363dcacc69c13b37798`
- Round 04: `541ce612db093729e3eee4be20f8e1a1aaa1a042cdd723cd119cd2f04effbe91`
- Round 05: `b9d4f19490cc9da3260ccce91739213a2936d3c0a238d6e29346a4d1a727421f`
- Round 06: `b3f3265fd0283e807033a330ff0679f0aaec0bea51aa3d7e67403a58cb3c2478`
- Round 07: `80846de8dacbd539012a01b0165a92a8681488e355bbffb8f2f06574827b6f4a`
- Round 08: `03e2441a200633b4ea0ed21184857ab8def4d8995feca82bdfafc2945e409e67`
- Round 09: `b9326f356a5feed2db3821125733c3ba2af4a988ad570be1e67fb8a590f4a4de`
- Round 10: `9326663904ba2bc66a2f87975169d3e6c00371469a28f8cfd557ccc4aa9b19ce`
- Round 11: `1c6ef423f44fc01e5df0e2c46dbe41b83bc24cd31c94d82a3371f2f91eeb611a`
- Round 12: `d2ac16cd5c815a4a97e836986fd5e13098015af1885e31785dcd4349d40cbcdf`

## Scope and confirmations

- Raw sampled trajectories were saved exactly as produced by the model. **No
  post-processing** of any kind was applied: no averaging or aggregation of the
  trajectories, no four-week return conversion, no rescaling.
- **No quintile probabilities and no RPS evaluation** occurred in this notebook;
  no realised/ground-truth future returns were loaded.
- No asset ranking, no forecast-accuracy evaluation.
- **DRE:** raw Chronos output was preserved for every round, including the
  post-acquisition rounds (9-12) where non-zero values may appear. DRE forecasts
  were **not** overwritten or forced to zero here, and PLD was not used. The
  official M6 zero-return rule for DRE will be applied later, during the
  competition-specific post-processing/evaluation stage.
