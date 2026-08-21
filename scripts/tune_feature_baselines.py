"""Stage 7: Random Forest and LightGBM baselines - implementation and PRE-M6
hyperparameter tuning.

This stage ONLY tunes. It does not run the 12 official M6 evaluation rounds and
never reads the official M6 ground truth.

Four configurations are tuned independently:
    RF   / no_knn      RF   / knn      LGBM / no_knn      LGBM / knn

Methodology adapted from Samartzis (2025). The released implementation under
``Relevant Context/`` is a critically checked reference only - nothing is
imported from it and nothing in it is modified. Preserved from the source: the
model families and their fixed settings, the HyperOpt/TPE search spaces, the
evaluation budget (20 x |space| evaluations with no-progress early stopping),
the four-week training-row structure and the cross-sectional probability
post-processing. Replaced by this project's finalized decisions: the ten
predictors, the future four-week quintile target, the forecast-origin
alignment, the KNN / no-KNN branches and the causal training-eligibility rule.

Tuning protocol, per hyperparameter candidate:
    for each of the 12 pre-M6 tuning origins T (four-week Fridays,
    2021-03-05 .. 2022-01-07):
        1. training set = every historical row of that branch whose four-week
           target had already been realised by T, i.e.
           ``eligible_for_training_from <= T``. The set EXPANDS as T advances,
           so the model is refitted at every origin.
        2. fit;
        3. predict the assets at T -> Rank1..Rank5 probabilities;
        4. apply the released cross-sectional adjustment;
        5. RPS against that BRANCH'S OWN historical target.
    objective = mean RPS over the 12 origins (lower is better).

Inputs (read-only):
    Data/processed/feature_baseline/supervised_rows_{no_knn,knn}.csv
    Data/metadata/feature_baseline_origin_schedule.csv

Output (the only file written):
    Data/metadata/feature_baseline_tuning_results.csv

Usage:
    python scripts/tune_feature_baselines.py
    python scripts/tune_feature_baselines.py --config rf:no_knn --max-evals 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from hyperopt import Trials, hp, tpe
from hyperopt.early_stop import no_progress_loss
from hyperopt.fmin import fmin
from hyperopt.pyll.base import scope
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# The project's single RPS implementation and quintile column order, reused so
# tuning is scored exactly as the official evaluator scores the M6 rounds.
from evaluate_m6_rps import RANK_COLUMNS, rps_scores  # noqa: E402
from build_feature_baseline_dataset import FEATURE_COLUMNS, VARIANTS  # noqa: E402

SUPERVISED_DIR = PROJECT_ROOT / "Data" / "processed" / "feature_baseline"
ORIGIN_SCHEDULE = PROJECT_ROOT / "Data" / "metadata" / "feature_baseline_origin_schedule.csv"
OUT_RESULTS = PROJECT_ROOT / "Data" / "metadata" / "feature_baseline_tuning_results.csv"

TARGET_COLUMN = "hist_target_quintile"
TRUTH_COLUMNS = [f"hist_target_{c}" for c in RANK_COLUMNS]

RANDOM_SEED = 42
N_QUINTILES = 5
N_TUNING_ORIGINS = 12
EVALS_PER_DIMENSION = 20          # source: MAX_EVALS = 20 * len(space)
NO_PROGRESS_PATIENCE = 50         # source: no_progress_loss(50)
LGBM_BOOST_ROUNDS = 50            # source: num_boost_round=50
PROBABILITY_TOLERANCE = 1e-9

MODELS = ("rf", "lgbm")

# --------------------------------------------------------------------------- #
# Search spaces - reproduced from the released model registry
# --------------------------------------------------------------------------- #
RF_SPACE = {
    "criterion": hp.choice("criterion", ["gini", "entropy", "log_loss"]),
    "max_features": hp.choice("max_features", ["sqrt", "log2", 0.5]),
    "n_estimators": scope.int(hp.quniform("n_estimators", 50, 500, 50)),
    "max_depth": scope.int(hp.quniform("max_depth", 3, 20, 1)),
    "min_samples_split": scope.int(hp.quniform("min_samples_split", 2, 20, 2)),
    "min_samples_leaf": scope.int(hp.quniform("min_samples_leaf", 1, 10, 1)),
}

LGBM_SPACE = {
    "num_leaves": scope.int(hp.quniform("num_leaves", 16, 512, 16)),
    "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.2)),
    "min_data_in_leaf": scope.int(hp.quniform("min_data_in_leaf", 5, 55, 5)),
    "subsample": hp.uniform("subsample", 0.5, 0.9),
    "feature_fraction": hp.uniform("feature_fraction", 0.6, 0.9),
    "max_depth": scope.int(hp.quniform("max_depth", 5, 15, 1)),
}

# Fixed LightGBM settings, from the released configuration.
LGBM_FIXED = {
    "boosting_type": "gbdt",
    "objective": "multiclass",
    "metric": "multi_logloss",
    "num_class": N_QUINTILES,
    "linear_tree": True,
    "verbose": -1,
    "random_state": RANDOM_SEED,
    # The released fixed config sets subsample_freq=1, but its SEARCH SPACE
    # omits it, so a tuned `subsample` would silently do nothing. bagging_freq=1
    # is set here so that searched dimension is actually active.
    "bagging_freq": 1,
}

SPACES = {"rf": RF_SPACE, "lgbm": LGBM_SPACE}

logger = logging.getLogger("tune_feature_baselines")


class ValidationError(RuntimeError):
    """A structural, causality or probability expectation was violated."""


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def load_tuning_origins() -> list[pd.Timestamp]:
    """The 12 pre-M6 tuning origins fixed by the preprocessing stage."""
    schedule = pd.read_csv(ORIGIN_SCHEDULE, parse_dates=["origin_date"])
    origins = list(schedule.loc[schedule["origin_role"] == "tuning", "origin_date"])
    if len(origins) != N_TUNING_ORIGINS:
        raise ValidationError(f"Expected {N_TUNING_ORIGINS} tuning origins, "
                              f"found {len(origins)}.")
    if any(o.dayofweek != 4 for o in origins):
        raise ValidationError("A tuning origin is not a Friday.")
    steps = {int(d.days) for d in np.diff(origins)}
    if steps != {28}:
        raise ValidationError(f"Tuning origins are not four-week spaced: {steps}.")
    m6 = set(schedule.loc[schedule["origin_role"] == "m6", "origin_date"])
    if m6 & set(origins):
        raise ValidationError("A tuning origin coincides with an M6 origin.")
    if max(origins) >= min(m6):
        raise ValidationError("A tuning origin is not strictly before the M6 sample.")
    return origins


def load_branch(variant: str) -> pd.DataFrame:
    """One preprocessing branch's supervised rows, restricted to labelled rows."""
    if variant not in VARIANTS:
        raise ValidationError(f"Unknown variant {variant!r}.")
    frame = pd.read_csv(
        SUPERVISED_DIR / f"supervised_rows_{variant}.csv",
        parse_dates=["origin_date", "target_end_date", "eligible_for_training_from"],
    )
    missing = [c for c in FEATURE_COLUMNS + TRUTH_COLUMNS + [TARGET_COLUMN]
               if c not in frame.columns]
    if missing:
        raise ValidationError(f"{variant}: missing columns {missing}.")
    frame = frame.loc[frame["hist_target_available"]].copy()
    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].astype(int)
    if not frame[TARGET_COLUMN].between(0, N_QUINTILES - 1).all():
        raise ValidationError(f"{variant}: target quintile outside 0..4.")
    if (frame["eligible_for_training_from"] != frame["target_end_date"]).any():
        raise ValidationError(f"{variant}: eligibility is not target_end.")
    return frame


def split_at_origin(frame: pd.DataFrame, origin: pd.Timestamp
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Causal expanding training set and the prediction cross-section at ``origin``.

    A historical row may train a model at ``origin`` only once its own four-week
    outcome has been realised, i.e. target_end(t) <= origin.
    """
    train = frame.loc[frame["eligible_for_training_from"] <= origin]
    predict = frame.loc[frame["origin_date"] == origin]
    if train.empty or predict.empty:
        raise ValidationError(f"Empty train/predict split at {origin.date()}.")
    if (train["target_end_date"] > origin).any():
        raise ValidationError(f"Leakage: an unrealised target entered training "
                              f"at {origin.date()}.")
    return train, predict


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #

def _to_full_probability_matrix(proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Map a classifier's class-ordered output onto the fixed 5 quintile columns.

    Column j is quintile j, i.e. Rank{j+1}: quintile 0 = lowest realised return
    = Rank1, quintile 4 = highest = Rank5 - the same direction as the project's
    ``rank_to_quintiles`` and the official evaluator.
    """
    full = np.zeros((proba.shape[0], N_QUINTILES), dtype=float)
    for column, quintile in enumerate(classes):
        full[:, int(quintile)] = proba[:, column]
    return full


def fit_predict(model: str, params: dict, train: pd.DataFrame,
                predict: pd.DataFrame, seed: int = RANDOM_SEED) -> np.ndarray:
    """Fit on the causal training set and return raw (n_assets, 5) probabilities.

    Missing predictor values are passed through untouched: both estimators
    handle them natively (scikit-learn >= 1.4 for trees, LightGBM by default).
    No imputation, filling or scaling is applied anywhere.
    """
    x_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET_COLUMN].to_numpy()
    x_predict = predict[FEATURE_COLUMNS]

    if model == "rf":
        estimator = RandomForestClassifier(random_state=seed, n_jobs=-1, **params)
        estimator.fit(x_train, y_train)
        return _to_full_probability_matrix(estimator.predict_proba(x_predict),
                                           estimator.classes_)
    if model == "lgbm":
        settings = dict(LGBM_FIXED, **params)
        settings["random_state"] = seed
        booster = lgb.train(settings,
                            lgb.Dataset(x_train, y_train, free_raw_data=False),
                            num_boost_round=LGBM_BOOST_ROUNDS)
        proba = booster.predict(x_predict)
        return _to_full_probability_matrix(proba, np.arange(N_QUINTILES))
    raise ValidationError(f"Unknown model {model!r}.")


def adjust_probabilities(proba: np.ndarray) -> np.ndarray:
    """The released cross-sectional post-processing.

        results[ranks] += 0.2 - results[ranks].mean()

    ``mean()`` on the released DataFrame is a per-COLUMN mean, so each quintile
    column is shifted until its cross-sectional average is exactly 0.20. Because
    the five column means of a valid probability matrix sum to 1, the five
    shifts sum to 0 and every row still sums to 1.
    """
    return proba + (0.2 - proba.mean(axis=0))


def validate_probabilities(proba: np.ndarray, context: str) -> None:
    """Every row must be a valid probability vector summing to 1."""
    if proba.shape[1] != N_QUINTILES:
        raise ValidationError(f"{context}: expected {N_QUINTILES} quintile columns.")
    if not np.isfinite(proba).all():
        raise ValidationError(f"{context}: non-finite probability.")
    row_error = float(np.abs(proba.sum(axis=1) - 1.0).max())
    if row_error > PROBABILITY_TOLERANCE:
        raise ValidationError(f"{context}: probability rows do not sum to 1 "
                              f"(max error {row_error:.3g}).")
    minimum, maximum = float(proba.min()), float(proba.max())
    if minimum < -PROBABILITY_TOLERANCE or maximum > 1 + PROBABILITY_TOLERANCE:
        raise ValidationError(
            f"{context}: the released cross-sectional adjustment produced an "
            f"invalid probability (min {minimum:.6f}, max {maximum:.6f}). "
            "Stopping rather than inventing another correction."
        )


# --------------------------------------------------------------------------- #
# Objective
# --------------------------------------------------------------------------- #

def evaluate_candidate(model: str, params: dict, frame: pd.DataFrame,
                       origins: list[pd.Timestamp], seed: int = RANDOM_SEED,
                       ) -> dict[str, object]:
    """Mean RPS over the 12 tuning origins, refitting at each one."""
    per_origin, train_sizes, predict_sizes = [], [], []
    minimum_probability, row_sum_error = 1.0, 0.0
    for origin in origins:
        train, predict = split_at_origin(frame, origin)
        raw = fit_predict(model, params, train, predict, seed)
        proba = adjust_probabilities(raw)
        validate_probabilities(proba, f"{model} @ {origin.date()}")
        truth = predict[TRUTH_COLUMNS].to_numpy(dtype=float)
        per_origin.append(float(rps_scores(truth, proba).mean()))
        train_sizes.append(len(train))
        predict_sizes.append(len(predict))
        minimum_probability = min(minimum_probability, float(proba.min()))
        row_sum_error = max(row_sum_error,
                            float(np.abs(proba.sum(axis=1) - 1.0).max()))
    return {
        "mean_rps": float(np.mean(per_origin)),
        "per_origin_rps": per_origin,
        "train_rows_min": int(min(train_sizes)),
        "train_rows_max": int(max(train_sizes)),
        "predict_rows_total": int(sum(predict_sizes)),
        "min_probability": minimum_probability,
        "max_row_sum_error": row_sum_error,
    }


def _jsonable(params: dict) -> dict:
    out = {}
    for key, value in params.items():
        if isinstance(value, (np.integer,)):
            out[key] = int(value)
        elif isinstance(value, (np.floating,)):
            out[key] = float(value)
        else:
            out[key] = value
    return out


def tune(model: str, variant: str, origins: list[pd.Timestamp],
         max_evals: int | None = None) -> dict[str, object]:
    """HyperOpt/TPE search for one configuration."""
    space = SPACES[model]
    budget = max_evals if max_evals is not None else EVALS_PER_DIMENSION * len(space)
    frame = load_branch(variant)
    label = f"{model}/{variant}"
    started = time.time()

    def objective(candidate: dict) -> dict:
        params = _jsonable(candidate)
        result = evaluate_candidate(model, params, frame, origins)
        return {"loss": result["mean_rps"], "status": "ok",
                "hyperparameters": params, "diagnostics": result}

    trials = Trials()
    logger.info("Tuning %s: budget %d evaluations over %d origins",
                label, budget, len(origins))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fmin(fn=objective, space=space, algo=tpe.suggest, trials=trials,
             max_evals=budget, early_stop_fn=no_progress_loss(NO_PROGRESS_PATIENCE),
             rstate=np.random.default_rng(RANDOM_SEED), show_progressbar=False)

    # The best candidate is taken from the recorded trial results rather than
    # from fmin's return value, which reports hp.choice INDICES rather than the
    # chosen values (the released code sidesteps this the same way, via its log).
    results = [r for r in trials.results if r.get("status") == "ok"]
    best = min(results, key=lambda r: r["loss"])
    diagnostics = best["diagnostics"]
    logger.info("%s: best mean RPS %.6f after %d evaluations (%.1f min) -> %s",
                label, best["loss"], len(results), (time.time() - started) / 60,
                best["hyperparameters"])
    return {
        "model": model,
        "variant": variant,
        "mean_tuning_rps": round(best["loss"], 8),
        "best_hyperparameters": json.dumps(best["hyperparameters"], sort_keys=True),
        "evaluations_completed": len(results),
        "evaluation_budget": budget,
        "early_stopped": len(results) < budget,
        "n_tuning_origins": len(origins),
        "first_tuning_origin": origins[0].date().isoformat(),
        "last_tuning_origin": origins[-1].date().isoformat(),
        "per_origin_rps": json.dumps([round(v, 8)
                                      for v in diagnostics["per_origin_rps"]]),
        "train_rows_min": diagnostics["train_rows_min"],
        "train_rows_max": diagnostics["train_rows_max"],
        "predict_rows_total": diagnostics["predict_rows_total"],
        "min_probability": round(diagnostics["min_probability"], 8),
        "max_row_sum_error": diagnostics["max_row_sum_error"],
        "random_seed": RANDOM_SEED,
        "feature_columns": json.dumps(FEATURE_COLUMNS),
        "lgbm_boost_rounds": LGBM_BOOST_ROUNDS if model == "lgbm" else "",
        "lgbm_fixed_params": (json.dumps(LGBM_FIXED, sort_keys=True)
                              if model == "lgbm" else ""),
        "probability_postprocessing": "column means shifted to 0.20 (released rule)",
        "sklearn_version": sklearn.__version__,
        "lightgbm_version": lgb.__version__,
        "tuning_minutes": round((time.time() - started) / 60, 2),
        "tuned_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def write_results(rows: list[dict]) -> None:
    """Rewrite the single compact results file (one row per configuration)."""
    frame = pd.DataFrame(rows).sort_values(["model", "variant"])
    OUT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT_RESULTS, index=False)
    logger.info("Wrote %s (%d configurations)", OUT_RESULTS, len(frame))


def run(configs: list[tuple[str, str]], max_evals: int | None = None
        ) -> list[dict]:
    origins = load_tuning_origins()
    logger.info("Tuning origins: %s",
                ", ".join(o.date().isoformat() for o in origins))
    existing: dict[tuple[str, str], dict] = {}
    if OUT_RESULTS.is_file():
        for record in pd.read_csv(OUT_RESULTS).to_dict("records"):
            existing[(record["model"], record["variant"])] = record
    for model, variant in configs:
        existing[(model, variant)] = tune(model, variant, origins, max_evals)
        write_results(list(existing.values()))
    return list(existing.values())


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Pre-M6 tuning of the RF and "
                                                 "LightGBM feature baselines.")
    parser.add_argument("--config", action="append", default=None,
                        help="model:variant, e.g. rf:no_knn (default: all four).")
    parser.add_argument("--max-evals", type=int, default=None,
                        help="Override the 20 x |space| budget (smoke tests).")
    args = parser.parse_args()

    if args.config:
        configs = []
        for item in args.config:
            model, _, variant = item.partition(":")
            if model not in MODELS or variant not in VARIANTS:
                parser.error(f"Bad --config {item!r}; expected model:variant.")
            configs.append((model, variant))
    else:
        configs = [(m, v) for m in MODELS for v in VARIANTS]

    try:
        run(configs, args.max_evals)
    except ValidationError as exc:
        logger.error("Validation failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
