"""Stage 8: held-out M6 evaluation of the finalized RF and LightGBM baselines.

The four configurations are FROZEN. Their hyperparameters are read from
``Data/metadata/feature_baseline_tuning_results.csv`` exactly as the pre-M6
tuning stage selected them, and nothing here retunes, reselects or adjusts any
modelling choice. No M6 outcome influences hyperparameters, features,
preprocessing, model settings or probability post-processing: the official
ground truth is loaded only AFTER every forecast for a configuration has been
produced, and is used solely to score.

Procedure, per configuration, per official M6 origin T (12 rounds,
2022-03-04 .. 2023-01-06):
    1. training set = every row of that branch with
       ``eligible_for_training_from <= T`` - the same causal eligibility rule
       used during tuning. The window EXPANDS as T advances (earlier M6 rounds
       become training rows for later ones, using this project's Dataset-D
       historical labels, never the official outcome).
    2. refit from scratch with the frozen hyperparameters;
    3. predict Rank1..Rank5 for that round's assets;
    4. apply the released cross-sectional adjustment,
       ``adjusted = raw + (0.2 - column_mean)``, unchanged from tuning;
    5. validate. If any adjusted probability is < 0, > 1, non-finite, or a row
       fails to sum to 1, that CONFIGURATION IS STOPPED. Probabilities are
       never clipped or renormalised, no other hyperparameter set is
       substituted, and no partial mean RPS is reported for it.
Only once all 12 rounds are forecast is the configuration scored against the
official M6 ground truth with the project's existing evaluator.

Inputs (read-only):
    Data/metadata/feature_baseline_tuning_results.csv   (frozen hyperparameters)
    Data/metadata/feature_baseline_origin_schedule.csv  (the 12 M6 origins)
    Data/metadata/m6_round_schedule.csv                 (authoritative rounds)
    Data/processed/feature_baseline/supervised_rows_{no_knn,knn}.csv
    Data/assets_m6.csv                                  (official truth, scoring only)

Outputs:
    Results/Evaluation/feature_baseline_m6_predictions.csv
    Results/Evaluation/feature_baseline_m6_rps.csv
    logs/feature_baseline_m6_evaluation.log

Usage:
    python scripts/evaluate_feature_baselines_m6.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# The project's existing official M6 machinery, imported rather than reimplemented.
from evaluate_m6_rps import (  # noqa: E402
    RANK_COLUMNS,
    build_ground_truth,
    load_prices,
    rps_scores,
)
from tune_feature_baselines import (  # noqa: E402
    FEATURE_COLUMNS,
    LGBM_BOOST_ROUNDS,
    N_QUINTILES,
    RANDOM_SEED,
    TARGET_COLUMN,
    ValidationError,
    adjust_probabilities,
    fit_predict,
    load_branch,
    probability_report,
    split_at_origin,
)

TUNING_RESULTS = PROJECT_ROOT / "Data" / "metadata" / "feature_baseline_tuning_results.csv"
ORIGIN_SCHEDULE = PROJECT_ROOT / "Data" / "metadata" / "feature_baseline_origin_schedule.csv"
M6_SCHEDULE = PROJECT_ROOT / "Data" / "metadata" / "m6_round_schedule.csv"

EVAL_DIR = PROJECT_ROOT / "Results" / "Evaluation"
OUT_PREDICTIONS = EVAL_DIR / "feature_baseline_m6_predictions.csv"
OUT_RPS = EVAL_DIR / "feature_baseline_m6_rps.csv"
LOG_PATH = PROJECT_ROOT / "logs" / "feature_baseline_m6_evaluation.log"

EXPECTED_ROUNDS = 12
EXPECTED_ASSETS = 100
NAIVE_RPS = 0.16

logger = logging.getLogger("evaluate_feature_baselines_m6")


# --------------------------------------------------------------------------- #
# Frozen configurations and the official rounds
# --------------------------------------------------------------------------- #

def load_frozen_configurations() -> list[dict]:
    """The four tuning winners, read verbatim. Nothing here may alter them."""
    if not TUNING_RESULTS.is_file():
        raise ValidationError(f"Tuning results not found: {TUNING_RESULTS}")
    frame = pd.read_csv(TUNING_RESULTS)
    expected = {("rf", "no_knn"), ("rf", "knn"), ("lgbm", "no_knn"), ("lgbm", "knn")}
    found = set(zip(frame["model"], frame["variant"]))
    if found != expected:
        raise ValidationError(f"Expected the four tuned configurations, found {found}.")
    configs = []
    for record in frame.itertuples():
        params = json.loads(record.best_hyperparameters)
        stored_features = json.loads(record.feature_columns)
        if stored_features != FEATURE_COLUMNS:
            raise ValidationError(
                f"{record.model}/{record.variant}: the tuned feature set differs "
                "from the current one."
            )
        if int(record.random_seed) != RANDOM_SEED:
            raise ValidationError("Tuned seed differs from the evaluation seed.")
        configs.append({
            "model": record.model,
            "variant": record.variant,
            "params": params,
            "tuning_rps": float(record.mean_tuning_rps),
        })
    return sorted(configs, key=lambda c: (c["model"], c["variant"]))


def load_m6_origins() -> pd.DataFrame:
    """The 12 official M6 rounds, cross-checked against both schedules."""
    baseline = pd.read_csv(ORIGIN_SCHEDULE, parse_dates=["origin_date",
                                                         "target_start_date",
                                                         "target_end_date"])
    baseline = baseline.loc[baseline["origin_role"] == "m6"].sort_values("m6_round")
    official = pd.read_csv(M6_SCHEDULE, parse_dates=["origin_date",
                                                     "forecast_start_date",
                                                     "forecast_end_date"])
    if len(baseline) != EXPECTED_ROUNDS or len(official) != EXPECTED_ROUNDS:
        raise ValidationError("Expected exactly 12 M6 rounds in both schedules.")
    if list(baseline["origin_date"]) != list(official["origin_date"]):
        raise ValidationError("M6 origins disagree between the two schedules.")
    if list(baseline["target_end_date"]) != list(official["forecast_end_date"]):
        raise ValidationError("M6 target ends disagree between the two schedules.")
    if sorted(official["round"]) != list(range(1, EXPECTED_ROUNDS + 1)):
        raise ValidationError("Official rounds are not 1..12.")
    return official


# --------------------------------------------------------------------------- #
# Forecasting
# --------------------------------------------------------------------------- #

def forecast_configuration(config: dict, rounds: pd.DataFrame,
                           ) -> tuple[pd.DataFrame | None, dict]:
    """Produce all 12 rounds of adjusted probabilities for one configuration.

    Returns (predictions, status). ``predictions`` is None when the
    configuration was stopped by an invalid adjusted probability.
    """
    model, variant = config["model"], config["variant"]
    label = f"{model}/{variant}"
    frame = load_branch(variant)
    blocks, train_sizes = [], []
    minimum, maximum, row_error = 1.0, 0.0, 0.0
    started = time.time()

    for record in rounds.itertuples():
        origin = pd.Timestamp(record.origin_date)
        train, predict = split_at_origin(frame, origin)
        if (train["target_end_date"] > origin).any():
            raise ValidationError(f"{label} round {record.round}: an unrealised "
                                  "target entered training.")
        raw = fit_predict(model, config["params"], train, predict, RANDOM_SEED)
        proba = adjust_probabilities(raw)
        report = probability_report(proba)
        if not report["valid"]:
            logger.error("%s: STOPPED at round %d (%s) - %s", label, record.round,
                         origin.date(), report["reason"])
            return None, {
                "status": "stopped_invalid_probability",
                "failed_round": int(record.round),
                "failed_origin": origin.date().isoformat(),
                "failure_reason": report["reason"],
                "rounds_completed": int(record.round) - 1,
                "runtime_minutes": round((time.time() - started) / 60, 2),
            }
        minimum = min(minimum, report["min"])
        maximum = max(maximum, report["max"])
        row_error = max(row_error, report["row_sum_error"])
        train_sizes.append(len(train))

        block = pd.DataFrame(proba, columns=RANK_COLUMNS)
        block.insert(0, "symbol", predict["symbol"].to_numpy())
        block.insert(0, "origin_date", origin.date().isoformat())
        block.insert(0, "round", int(record.round))
        block.insert(0, "variant", variant)
        block.insert(0, "model", model)
        blocks.append(block)
        logger.info("%s round %2d (%s): trained on %d rows, forecast %d assets",
                    label, record.round, origin.date(), len(train), len(predict))

    return pd.concat(blocks, ignore_index=True), {
        "status": "complete",
        "failed_round": "",
        "failed_origin": "",
        "failure_reason": "",
        "rounds_completed": EXPECTED_ROUNDS,
        "train_rows_min": int(min(train_sizes)),
        "train_rows_max": int(max(train_sizes)),
        "min_probability": round(minimum, 8),
        "max_probability": round(maximum, 8),
        "max_row_sum_error": row_error,
        "runtime_minutes": round((time.time() - started) / 60, 2),
    }


# --------------------------------------------------------------------------- #
# Scoring against the official ground truth
# --------------------------------------------------------------------------- #

def score_against_official_truth(predictions: pd.DataFrame,
                                 truth: pd.DataFrame) -> dict[int, float]:
    """Official M6 RPS per round, using the project's existing evaluator.

    The forecast is aligned to the ground truth's own symbol order per round, so
    a mis-ordered prediction cannot silently score against the wrong asset.
    """
    per_round: dict[int, float] = {}
    for round_number, truth_block in truth.groupby("round"):
        forecast_block = predictions.loc[predictions["round"] == round_number]
        if len(forecast_block) != len(truth_block):
            raise ValidationError(
                f"Round {round_number}: forecast has {len(forecast_block)} assets, "
                f"official truth has {len(truth_block)}.")
        aligned = forecast_block.set_index("symbol").reindex(truth_block["symbol"])
        if aligned[RANK_COLUMNS].isna().any().any():
            missing = sorted(set(truth_block["symbol"]) - set(forecast_block["symbol"]))
            raise ValidationError(f"Round {round_number}: no forecast for {missing}.")
        per_round[int(round_number)] = float(rps_scores(
            truth_block[RANK_COLUMNS].to_numpy(dtype=float),
            aligned[RANK_COLUMNS].to_numpy(dtype=float),
        ).mean())
    return per_round


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def run() -> pd.DataFrame:
    rounds = load_m6_origins()
    configs = load_frozen_configurations()
    logger.info("Frozen configurations: %s",
                ", ".join(f"{c['model']}/{c['variant']}" for c in configs))
    logger.info("M6 origins: %s",
                ", ".join(str(pd.Timestamp(d).date()) for d in rounds["origin_date"]))

    # Every forecast is produced BEFORE the official outcome is loaded.
    forecasts, statuses = {}, {}
    for config in configs:
        key = (config["model"], config["variant"])
        logger.info("Forecasting %s/%s with frozen params %s",
                    *key, json.dumps(config["params"], sort_keys=True))
        forecasts[key], statuses[key] = forecast_configuration(config, rounds)

    logger.info("All forecasts produced; now loading the official M6 ground truth.")
    truth = build_ground_truth(load_prices(), pd.read_csv(M6_SCHEDULE))
    if len(truth) != EXPECTED_ROUNDS * EXPECTED_ASSETS:
        raise ValidationError(f"Official truth has {len(truth)} rows, expected "
                              f"{EXPECTED_ROUNDS * EXPECTED_ASSETS}.")

    rows, prediction_blocks = [], []
    for config in configs:
        key = (config["model"], config["variant"])
        status = statuses[key]
        record = {"model": key[0], "variant": key[1],
                  "mean_tuning_rps": config["tuning_rps"],
                  "hyperparameters": json.dumps(config["params"], sort_keys=True),
                  **status}
        if forecasts[key] is None:
            record.update({f"round_{i:02d}_rps": "" for i in range(1, 13)})
            record["mean_m6_rps"] = ""
            record["vs_naive"] = ""
            logger.error("%s/%s: no M6 RPS reported (configuration stopped).", *key)
        else:
            per_round = score_against_official_truth(forecasts[key], truth)
            mean_rps = float(np.mean(list(per_round.values())))
            record.update({f"round_{i:02d}_rps": round(per_round[i], 8)
                           for i in range(1, 13)})
            record["mean_m6_rps"] = round(mean_rps, 8)
            record["vs_naive"] = round(mean_rps - NAIVE_RPS, 8)
            prediction_blocks.append(forecasts[key])
            logger.info("%s/%s: mean M6 RPS %.6f (naive %.6f, difference %+.6f)",
                        *key, mean_rps, NAIVE_RPS, mean_rps - NAIVE_RPS)
        record["sklearn_version"] = sklearn.__version__
        record["lightgbm_version"] = lgb.__version__
        record["lgbm_boost_rounds"] = LGBM_BOOST_ROUNDS if key[0] == "lgbm" else ""
        record["random_seed"] = RANDOM_SEED
        record["evaluated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        rows.append(record)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    if prediction_blocks:
        pd.concat(prediction_blocks, ignore_index=True).to_csv(OUT_PREDICTIONS,
                                                               index=False)
        logger.info("Wrote %s", OUT_PREDICTIONS)
    results = pd.DataFrame(rows)
    results.to_csv(OUT_RPS, index=False)
    logger.info("Wrote %s", OUT_RPS)
    return results


def main() -> int:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)],
    )
    try:
        run()
    except ValidationError as exc:
        logger.error("Validation failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
