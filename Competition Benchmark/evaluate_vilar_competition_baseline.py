"""Stage 11: competition-grade M6 benchmark - reproduction of J.M.G. Vilar's
quasi-average forecasting method.

Reference: Jose M.G. Vilar, "Quasi-average predictions and regression to the
trend: An application to the M6 financial forecasting competition",
International Journal of Forecasting 41 (2025) 1505-1513. Published M6 result:
global forecasting RPS 0.15729, rank 8 of 163 (paper Table 1; benchmark 0.16000
at rank 51; best forecasting-track performer 0.15645).

Source materials, READ-ONLY, in `Competition Benchmark Reference/`:
    m6paper.ipynb      the released reproducibility notebook (authoritative)
    Quasi Paper.pdf    the published paper
    M6_Universe.csv    the 100-asset universe with Stock/ETF classes
    AdjClose.csv.gz    adjusted closes, 2000-01-03 .. 2023-10-13, 100 assets

THIS IS A REPRODUCTION OF THE FIXED RELEASED METHODOLOGY, not of the live
competition submissions. The paper states the approach was used "with slightly
different parameters in each submission"; those per-submission parameters are
not published. The released notebook specifies one fixed configuration, and that
is what is reproduced here. Nothing is tuned, and nothing is adjusted to move the
score toward 0.15729.

THE METHOD, exactly as the released notebook implements it
----------------------------------------------------------
    lag = 20
    qf  = 1 + qcut(log(P_t) - log(P_{t-20}), 5, labels=False)   row-wise
          -> a daily panel of cross-sectional quintiles 1..5,
             Q1 = worst 20-day return, Q5 = best.
          The notebook first trims the panel to its last
          lag*((len(df)-1)//lag - 1) + 1 rows; that trim is reproduced because
          it defines the start of the "400-period" window.

    q_dists(x): drop NaN, count quintiles 1..5, return the CUMULATIVE
                distribution r.cumsum()/r.sum(); if nothing survives dropna(),
                return the uniform cumulative [0.2, 0.4, 0.6, 0.8, 1.0].

    For an information cutoff whose last usable row is `end - 1`:
      TYPE component   (per asset, from its group)
        group distribution = mean over the group's assets of q_dists applied to
        that asset's own column over qf[end - 50*lag : end].
        Groups: ETFs (49, VXX excluded), Stocks (49, DRE excluded),
                VXX (its own group), DRE (see below).
      TEMPORAL component (per asset, its own column)
        0.2 * q_dists over qf[end -   5*lag : end]
      + 0.2 * q_dists over qf[end -  10*lag : end]
      + 0.6 * q_dists over qf[end - 400*lag : end]
      MIXED (the reported method)
        0.5 * type + 0.5 * temporal            (notebook's av = 0.5)
      Ordinary probabilities follow by first-differencing the cumulative vector
      (the notebook's `desum`).

    VXX is its own asset class: the paper identifies it as the "salient
    exception" among ETFs, and the notebook removes it from `etfs` and gives it a
    one-member group, so its type component is its own 50-period distribution.

    DRE is given the UNIFORM cumulative [0.2, 0.4, 0.6, 0.8, 1.0] as its TYPE
    component - the notebook computes a DRE group distribution and then discards
    it (`1*uniform + 0*res[3]`). Its TEMPORAL component is unaffected, so DRE's
    mixed forecast is 0.5*uniform + 0.5*its own temporal distribution.

    Missing values are handled by `dropna()` INSIDE q_dists, per asset and per
    window. A date is never dropped for all assets because one asset is missing,
    and the uniform fallback applies only when a particular asset/window has no
    usable observation at all.

Evaluation uses the project's existing 12 official M6 origins and the project's
existing official ground truth and RPS scorer. Nothing about the evaluation is
redefined.

Outputs:
    Results/Evaluation/vilar_m6_predictions.csv
    Results/Evaluation/vilar_m6_rps.csv
    logs/vilar_competition_baseline.log

Usage:
    python "Competition Benchmark/evaluate_vilar_competition_baseline.py"
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_m6_rps import (  # noqa: E402
    RANK_COLUMNS,
    build_ground_truth,
    load_prices,
    rps_scores,
)

REFERENCE_DIR = PROJECT_ROOT / "Competition Benchmark Reference"
UNIVERSE_CSV = REFERENCE_DIR / "M6_Universe.csv"
ADJCLOSE_GZ = REFERENCE_DIR / "AdjClose.csv.gz"

M6_SCHEDULE = PROJECT_ROOT / "Data" / "metadata" / "m6_round_schedule.csv"
EVAL_DIR = PROJECT_ROOT / "Results" / "Evaluation"
OUT_PREDICTIONS = EVAL_DIR / "vilar_m6_predictions.csv"
OUT_RPS = EVAL_DIR / "vilar_m6_rps.csv"
LOG_PATH = PROJECT_ROOT / "logs" / "vilar_competition_baseline.log"

LAG = 20
N_QUINTILES = 5
N_ROUNDS = 12
N_ASSETS = 100
NAIVE_RPS = 0.16
UNIFORM_CUMULATIVE = np.array([0.2, 0.4, 0.6, 0.8, 1.0])

TYPE_PERIODS = 50
TEMPORAL_WEIGHTS = ((5, 0.2), (10, 0.2), (400, 0.6))
MIXED_TEMPORAL_WEIGHT = 0.5          # notebook's av; type gets (1 - av)

# Vilar's universe file uses EG (Everest Group); the project's canonical M6
# ticker for the same security is RE. Validated in project Stage 1B.
VILAR_TO_PROJECT_SYMBOL = {"EG": "RE"}

PUBLISHED_VILAR_RPS = 0.15729
PUBLISHED_VILAR_RANK = 8
PUBLISHED_BEST_RPS = 0.15645

logger = logging.getLogger("vilar_competition_baseline")


class ReproductionError(RuntimeError):
    """A structural expectation about the reference materials was violated."""


# --------------------------------------------------------------------------- #
# Reference data and the quintile panel
# --------------------------------------------------------------------------- #

def load_reference_inputs() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Load AdjClose and the universe exactly as the notebook does."""
    for path in (UNIVERSE_CSV, ADJCLOSE_GZ):
        if not path.is_file():
            raise ReproductionError(f"Missing reference input: {path}")

    prices = pd.read_csv(ADJCLOSE_GZ, index_col=0, low_memory=False,
                         parse_dates=True)
    universe = pd.read_csv(UNIVERSE_CSV)
    universe.columns = [c.strip().lstrip("﻿") for c in universe.columns]

    if prices.shape[1] != N_ASSETS or list(prices.columns) != list(universe["symbol"]):
        raise ReproductionError("AdjClose columns do not match the universe order.")

    etfs = list(universe[universe["class"] == "ETF"]["symbol"])
    stocks = list(universe[universe["class"] == "Stock"]["symbol"])
    if "VXX" not in etfs or "DRE" not in stocks:
        raise ReproductionError("Expected VXX among ETFs and DRE among stocks.")
    etfs.remove("VXX")
    stocks.remove("DRE")
    groups = {"etfs": etfs, "stocks": stocks, "vxx": ["VXX"], "dre": ["DRE"]}
    logger.info("Reference inputs: %d dates x %d assets, %s .. %s | "
                "groups ETF=%d Stock=%d VXX=1 DRE=1",
                prices.shape[0], prices.shape[1], prices.index.min().date(),
                prices.index.max().date(), len(etfs), len(stocks))
    return prices, groups


def build_quintile_panel(prices: pd.DataFrame) -> pd.DataFrame:
    """The notebook's `qf`: daily cross-sectional quintiles of 20-day log returns.

    Reproduces cell 4 verbatim, including its leading trim:
        qf = 1 + log(P).diff(20)[-lag*((len(P)-1)//lag - 1) - 1:]
                 .apply(lambda x: pd.qcut(x, 5, duplicates='drop', labels=False),
                        axis=1)
    `axis=1` makes the qcut CROSS-SECTIONAL on each date; `labels=False` gives
    0..4 ascending in return, and the leading `1 +` shifts to 1..5, so quintile 1
    is the worst 20-day return and quintile 5 the best. NaN returns stay NaN.
    """
    trim = -LAG * ((len(prices) - 1) // LAG - 1) - 1
    returns = np.log(prices).diff(LAG)[trim:]
    panel = 1 + returns.apply(
        lambda row: pd.qcut(row, N_QUINTILES, duplicates="drop", labels=False),
        axis=1)
    logger.info("Quintile panel: %d dates (trim %d) x %d assets, %s .. %s",
                panel.shape[0], trim, panel.shape[1],
                panel.index.min().date(), panel.index.max().date())
    return panel


def q_dists(column: pd.Series) -> np.ndarray:
    """The notebook's `q_dists`, unchanged: cumulative quintile distribution.

    NaNs are dropped for THIS asset and THIS window only. With nothing left, the
    source's uniform fallback is returned.
    """
    values = column.dropna().astype(int)
    if len(values) == 0:
        return UNIFORM_CUMULATIVE.copy()
    counts = np.array([(values == q).sum() for q in range(1, N_QUINTILES + 1)])
    return counts.cumsum() / counts.sum()


# --------------------------------------------------------------------------- #
# The three distributions
# --------------------------------------------------------------------------- #

def _window(panel: pd.DataFrame, end: int, periods: int) -> pd.DataFrame:
    """Rows [end - periods*LAG, end), clamped at the start of the panel."""
    return panel.iloc[max(0, end - periods * LAG): end]


def type_distributions(panel: pd.DataFrame, groups: dict[str, list[str]],
                       end: int) -> dict[str, np.ndarray]:
    """Per-group cumulative distribution over the trailing 50 periods.

    Each group's vector is the MEAN over its member assets of that asset's own
    cumulative distribution - the notebook's
    `qf[...][group].apply(q_dists, axis=0).mean(axis=1)`.
    """
    window = _window(panel, end, TYPE_PERIODS)
    out = {}
    for name, members in groups.items():
        per_asset = np.stack([q_dists(window[symbol]) for symbol in members])
        out[name] = per_asset.mean(axis=0)
    return out


def temporal_distribution(panel: pd.DataFrame, end: int) -> np.ndarray:
    """Weighted 5/10/400-period cumulative distribution, per asset.

    Returns an array of shape (5, n_assets) matching the panel's column order.
    """
    total = np.zeros((N_QUINTILES, panel.shape[1]))
    for periods, weight in TEMPORAL_WEIGHTS:
        window = _window(panel, end, periods)
        block = np.stack([q_dists(window[symbol]) for symbol in panel.columns],
                         axis=1)
        total += weight * block
    return total


def mixed_forecast(panel: pd.DataFrame, groups: dict[str, list[str]],
                   end: int) -> pd.DataFrame:
    """The released fixed method: 0.5 * type + 0.5 * temporal, as probabilities.

    ``end`` is the EXCLUSIVE stop of the information window, so the last usable
    observation is ``panel.index[end - 1]``. Nothing at or after ``end`` is read.
    """
    type_vectors = type_distributions(panel, groups, end)
    membership = {**{s: "etfs" for s in groups["etfs"]},
                  **{s: "stocks" for s in groups["stocks"]},
                  "VXX": "vxx"}
    type_block = np.stack(
        [UNIFORM_CUMULATIVE.copy() if symbol == "DRE"      # notebook: 0 * res[3]
         else type_vectors[membership[symbol]]
         for symbol in panel.columns], axis=1)

    cumulative = ((1 - MIXED_TEMPORAL_WEIGHT) * type_block
                  + MIXED_TEMPORAL_WEIGHT * temporal_distribution(panel, end))

    probabilities = np.diff(cumulative, axis=0, prepend=0.0)   # notebook's desum
    return pd.DataFrame(probabilities, columns=panel.columns,
                        index=[f"Rank{i}" for i in range(1, N_QUINTILES + 1)])


# --------------------------------------------------------------------------- #
# M6 evaluation on the project's official schedule
# --------------------------------------------------------------------------- #

def load_rounds() -> pd.DataFrame:
    schedule = pd.read_csv(M6_SCHEDULE, parse_dates=["origin_date",
                                                     "forecast_start_date",
                                                     "forecast_end_date"])
    if len(schedule) != N_ROUNDS or sorted(schedule["round"]) != list(range(1, 13)):
        raise ReproductionError("The official M6 schedule is not 12 rounds 1..12.")
    return schedule


def forecast_all_rounds(panel: pd.DataFrame, groups: dict[str, list[str]],
                        schedule: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """One mixed forecast per official M6 origin, with explicit causality checks."""
    blocks, audit = [], []
    for record in schedule.itertuples():
        origin = pd.Timestamp(record.origin_date)
        if origin not in panel.index:
            raise ReproductionError(f"Origin {origin.date()} is not a trading day "
                                    "in the reference price history.")
        end = int(panel.index.get_loc(origin)) + 1

        # --- leakage checks, before anything is computed ------------------- #
        last_used = panel.index[end - 1]
        if last_used != origin:
            raise ReproductionError(f"Round {record.round}: information window "
                                    f"ends {last_used.date()}, not the origin.")
        if end < len(panel) and panel.index[end] <= origin:
            raise ReproductionError("Panel index is not sorted ascending.")
        # The quintile at date d is the return over (d-20, d]; the last row used
        # is the origin itself, so no return reaching past the origin is read.

        forecast = mixed_forecast(panel, groups, end)
        probabilities = forecast.to_numpy().T                  # (assets, 5)
        if not np.isfinite(probabilities).all():
            raise ReproductionError(f"Round {record.round}: non-finite probability.")
        if probabilities.min() < -1e-12:
            raise ReproductionError(f"Round {record.round}: negative probability "
                                    f"{probabilities.min()}.")
        row_error = float(np.abs(probabilities.sum(axis=1) - 1.0).max())
        if row_error > 1e-9:
            raise ReproductionError(f"Round {record.round}: rows do not sum to 1 "
                                    f"(max error {row_error:.3g}).")

        # Vilar's own target for this origin: the quintile 20 rows later.
        target_position = end - 1 + LAG
        target_date = (panel.index[target_position]
                       if target_position < len(panel) else pd.NaT)

        block = pd.DataFrame(probabilities, columns=RANK_COLUMNS)
        block.insert(0, "symbol", [VILAR_TO_PROJECT_SYMBOL.get(s, s)
                                   for s in panel.columns])
        block.insert(0, "origin_date", origin.date().isoformat())
        block.insert(0, "round", int(record.round))
        block.insert(0, "model", "Vilar mixed (0.5 type + 0.5 temporal)")
        blocks.append(block)

        audit.append({
            "round": int(record.round),
            "origin_date": origin.date().isoformat(),
            "information_rows_used": end,
            "last_information_date": last_used.date().isoformat(),
            "vilar_target_date": (target_date.date().isoformat()
                                  if pd.notna(target_date) else ""),
            "official_forecast_end": record.forecast_end_date.date().isoformat(),
            "target_dates_agree": bool(pd.notna(target_date)
                                       and target_date == record.forecast_end_date),
            "min_probability": round(float(probabilities.min()), 8),
            "max_probability": round(float(probabilities.max()), 8),
            "max_row_sum_error": row_error,
        })
        logger.info("Round %2d %s: %d information rows, last used %s, "
                    "Vilar target %s (official end %s, agree=%s)",
                    record.round, origin.date(), end, last_used.date(),
                    audit[-1]["vilar_target_date"],
                    audit[-1]["official_forecast_end"],
                    audit[-1]["target_dates_agree"])
    return pd.concat(blocks, ignore_index=True), audit


def internal_reference_rps(panel: pd.DataFrame, groups: dict[str, list[str]],
                           schedule: pd.DataFrame) -> float | None:
    """Vilar's OWN RPS for the 12 M6 periods, using his own target definition.

    This is a like-for-like check against the released notebook (which scores
    `q_dists` of the realised quintile 20 rows later against the cumulative
    forecast). It is NOT the project's official score - that comes from the
    project's ground truth and scorer - but it shows what the released method
    scores on its own terms and its own data.
    """
    errors = []
    for record in schedule.itertuples():
        end = int(panel.index.get_loc(pd.Timestamp(record.origin_date))) + 1
        target_position = end - 1 + LAG
        if target_position >= len(panel):
            return None
        cumulative_forecast = np.cumsum(
            mixed_forecast(panel, groups, end).to_numpy(), axis=0)
        realised = panel.iloc[target_position: target_position + 1]
        realised_cumulative = np.stack(
            [q_dists(realised[symbol]) for symbol in panel.columns], axis=1)
        errors.append(np.nanmean((realised_cumulative - cumulative_forecast) ** 2))
    return float(np.mean(errors))


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def run() -> pd.DataFrame:
    prices, groups = load_reference_inputs()
    panel = build_quintile_panel(prices)
    schedule = load_rounds()

    predictions, audit = forecast_all_rounds(panel, groups, schedule)
    if len(predictions) != N_ROUNDS * N_ASSETS:
        raise ReproductionError(f"{len(predictions)} forecasts, expected 1200.")

    reference_rps = internal_reference_rps(panel, groups, schedule)
    logger.info("Internal fixed-method RPS on Vilar's own targets: %s",
                f"{reference_rps:.6f}" if reference_rps is not None else "n/a")

    # Official scoring happens only after every forecast exists.
    logger.info("Forecasts complete; loading the project's official M6 ground truth.")
    truth = build_ground_truth(load_prices(), pd.read_csv(M6_SCHEDULE))
    per_round = {}
    for round_number, truth_block in truth.groupby("round"):
        block = predictions[predictions["round"] == round_number]
        aligned = block.set_index("symbol").reindex(truth_block["symbol"])
        if aligned[RANK_COLUMNS].isna().any().any():
            missing = sorted(set(truth_block["symbol"]) - set(block["symbol"]))
            raise ReproductionError(f"Round {round_number}: no forecast for {missing}")
        per_round[int(round_number)] = float(rps_scores(
            truth_block[RANK_COLUMNS].to_numpy(dtype=float),
            aligned[RANK_COLUMNS].to_numpy(dtype=float)).mean())

    overall = float(np.mean(list(per_round.values())))
    beating = sum(1 for value in per_round.values() if value < NAIVE_RPS)
    logger.info("Official project RPS: %.6f (naive %.4f, difference %+.6f); "
                "%d/12 rounds beat naive", overall, NAIVE_RPS,
                overall - NAIVE_RPS, beating)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUT_PREDICTIONS, index=False)
    logger.info("Wrote %s (%d rows)", OUT_PREDICTIONS, len(predictions))

    audit_frame = pd.DataFrame(audit)
    record = {
        "model": "Vilar mixed (0.5 type + 0.5 temporal)",
        "source": "Vilar (2025) released reproducibility notebook, fixed parameters",
        **{f"round_{i:02d}_rps": round(per_round[i], 8) for i in range(1, 13)},
        "mean_m6_rps": round(overall, 8),
        "naive_rps": NAIVE_RPS,
        "difference_vs_naive": round(overall - NAIVE_RPS, 8),
        "rounds_beating_naive": beating,
        "internal_reference_rps_vilar_targets": (round(reference_rps, 8)
                                                 if reference_rps is not None else ""),
        "published_vilar_global_rps": PUBLISHED_VILAR_RPS,
        "difference_vs_published": round(overall - PUBLISHED_VILAR_RPS, 8),
        "published_vilar_rank": PUBLISHED_VILAR_RANK,
        "published_best_performer_rps": PUBLISHED_BEST_RPS,
        "lag": LAG,
        "type_periods": TYPE_PERIODS,
        "temporal_windows_and_weights": "5:0.2, 10:0.2, 400:0.6",
        "mixed_weight_temporal": MIXED_TEMPORAL_WEIGHT,
        "n_forecasts": len(predictions),
        "min_probability": round(float(audit_frame["min_probability"].min()), 8),
        "max_probability": round(float(audit_frame["max_probability"].max()), 8),
        "max_row_sum_error": float(audit_frame["max_row_sum_error"].max()),
        "all_target_dates_agree_with_official": bool(
            audit_frame["target_dates_agree"].all()),
        "evaluated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    results = pd.DataFrame([record])
    results.to_csv(OUT_RPS, index=False)
    logger.info("Wrote %s", OUT_RPS)
    return results


def main() -> int:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)])
    try:
        run()
    except ReproductionError as exc:
        logger.error("Reproduction failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
