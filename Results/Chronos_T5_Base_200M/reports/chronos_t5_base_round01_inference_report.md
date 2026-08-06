# Chronos-T5 Base (200M) - M6 Round 1 Inference Report

Generated: 2026-08-05 13:11:45 UTC

## Experiment

- Model: `amazon/chronos-t5-base` (Chronos-T5 Base, 200M parameters), inference only - no
  training or fine-tuning.
- Chronos is univariate: each of the 100 M6 assets was forecast as its own
  independent series (batched only for speed, batch size 10).
- Round: M6 Round 1, forecast origin 2022-03-04.
- Context: 512 weekday daily log returns,
  2020-03-19 to 2022-03-04, taken
  unchanged from the Stage 3 file `round_01_context.csv`
  (SHA-256 `ba19ad0af578e6ec...`). No preprocessing was repeated: no
  normalising, standardising, smoothing, clipping, averaging or NaN-filling.
  CARR's and OGN's genuine leading missing values were left as NaN and handled
  by Chronos's missing-value mask.
- Horizon: 20 weekdays, 2022-03-07 to 2022-04-01.
- Trajectories: 100 sampled paths per asset; random seed 42.
- Runtime: Google Colab GPU (NVIDIA L4), dtype torch.bfloat16,
  chronos-forecasting 2.3.1, torch 2.11.0+cu128.

## Result

- Raw output shape: (100, 100, 20) = (assets, sampled
  trajectories, forecast weekdays) - exactly as specified.
- Validation passed: shape correct, all values finite, asset order preserved,
  saved file reloaded successfully with identical values and ordering.
- Raw forecasts saved to: `/content/drive/MyDrive/HonoursResearch/Round_1_Context/outputs/chronos_t5_base/chronos_t5_base_round01_samples.npz`
- Metadata saved to: `/content/drive/MyDrive/HonoursResearch/Round_1_Context/outputs/chronos_t5_base/chronos_t5_base_round01_metadata.json`

## Scope

The raw sampled trajectories were saved before any M6 post-processing. No
four-week return conversion, asset ranking, quintile assignment, quintile
probability construction, realised-outcome comparison or RPS evaluation was
performed in this notebook.
