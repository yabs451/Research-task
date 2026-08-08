# Test 1 — Does zeroing DRE's forecast improve RPS?

Exploratory sensitivity check, run 2026-08-06. **Outcome: no — it makes both
models very slightly worse.** The primary evaluation should stay on the raw
forecasts.

## What was tested

One variable only: DRE's forecast trajectory left **raw** versus **set to zero**
in the post-acquisition rounds 9–12, where DRE's realised M6 return is fixed at
exactly 0. Everything else — ground truth, quintile conversion, tie handling,
RPS — is unchanged.

- Rounds 1–8 use the raw forecasts in both variants. **Round 8 is deliberately
  untouched**, because its forecast window crosses the acquisition date and
  needs a date-specific rule that has not been decided.
- Verified: the round-level RPS for rounds 1–8 is identical in both variants
  (max difference 0.0), and each derived file differs from its raw counterpart
  in the DRE row only.

## Method

Reused `scripts/evaluate_m6_rps.py` (`build_ground_truth`,
`quintile_probabilities`, `rps_scores`) and the DRE-zeroed copies already in
`Results/Evaluation/derived_dre_adjusted/<model>/`. No new evaluator, no new
ground truth, no raw NPZ modified — all files opened read-only. Results in
`dre_zero_test_results.csv`.

## Result (lower RPS is better)

| Model | Original overall RPS | DRE-adjusted overall RPS | Difference |
|---|---|---|---|
| Chronos T5 Base 200M | 0.226899 | 0.227017 | **+0.000118** |
| Financial Chronos Small 46M 2021 Global | 0.179368 | 0.179398 | **+0.000029** |

Affected rounds only (rounds 1–8 unchanged):

| Model | Round | Original | Adjusted | Difference | DRE asset: original → adjusted |
|---|---|---|---|---|---|
| Chronos T5 Base 200M | 9 | 0.276527 | 0.277916 | +0.001390 | 0.4835 → 0.6034 |
| Chronos T5 Base 200M | 10 | 0.183424 | 0.183446 | +0.000023 | 0.3882 → 0.3921 |
| Chronos T5 Base 200M | 11 | 0.202942 | 0.202942 | 0.000000 | 0.2003 → 0.2003 |
| Chronos T5 Base 200M | 12 | 0.327434 | 0.327440 | +0.000006 | 0.4245 → 0.4218 |
| Financial Chronos Small 46M 2021 Global | 9 | 0.177572 | 0.177698 | +0.000126 | 0.3817 → 0.3960 |
| Financial Chronos Small 46M 2021 Global | 10 | 0.168564 | 0.168571 | +0.000007 | 0.0169 → 0.0192 |
| Financial Chronos Small 46M 2021 Global | 11 | 0.173373 | 0.173532 | +0.000159 | 0.1772 → 0.1882 |
| Financial Chronos Small 46M 2021 Global | 12 | 0.173675 | 0.173735 | +0.000060 | 0.0025 → 0.0013 |

The model ranking is unaffected: Financial Chronos still beats Chronos Base, and
both still lose to the 0.16 naive benchmark.

## Why it does not help

Zeroing DRE does not move it to a *better* quintile — it makes the existing
prediction **more confident**, and in these rounds that prediction is wrong.

M6 quintiles are cross-sectional, so a 0% forecast is ranked against the other
99 assets' *forecasts*, not against zero. In rounds 9–12 Chronos Base gives
79–92% of the other assets a negative predicted median return, so a flat 0% DRE
ranks near the **top** of the predicted cross-section. The realised outcome went
the other way: DRE's actual 0% ranked Q1, Q3, Q3 and Q2 in the *realised*
cross-section of rounds 9–12, because enough real assets rose.

Round 9, Chronos Base, is the clearest case — actual quintile Q1:

    raw     [0.04, 0.04, 0.13, 0.63, 0.16]   RPS 0.484
    zeroed  [0.00, 0.00, 0.00, 0.87, 0.13]   RPS 0.603

The adjustment sharpened a bet on Q4 when the answer was Q1. The same pattern,
weaker, holds for Financial Chronos. Round-level differences are not purely
DRE's own score either: fixing DRE's rank shifts neighbouring assets by a slot,
which is why round 12 gets marginally worse even though DRE's own RPS improves.

The effect is tiny in every case (≤0.0014 on any round, ≤0.00012 overall)
because it is one asset out of 100 in four rounds out of twelve.

## Conclusion

Do not adopt the DRE zero-adjustment as the primary evaluation. The competition
rule is about the *realised* return being zero; imposing it on the *forecast*
only removes forecast uncertainty and, here, sharpens an incorrect
cross-sectional call. Worth revisiting only if Round 8's date-specific treatment
is defined, or as a footnote showing that the DRE handling does not drive any
conclusion.
