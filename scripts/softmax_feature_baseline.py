"""Stage 9: Softmax (multinomial logistic) regression baseline - pre-M6 tuning
and held-out M6 evaluation.

An M6 benchmark adapted from the feature-based classification methodology of
Samartzis (2025); not an exact reproduction. Softmax is the low-complexity,
interpretable linear member of the feature-based family already containing
Random Forest (bagged trees) and LightGBM (boosted trees). It tests whether the
signal those ensembles find needs a nonlinear tree structure at all, or is
largely representable by a linear multiclass decision boundary.

Input branch: the finalized KNN supervised dataset, used READ-ONLY. In that
branch 8 of the 10 predictors are already complete; only the two volume-derived
features (feat_dollar_volume_2m, feat_abs_ret_to_volume_3m) carry residual NaNs,
which come from the Samartzis zero-volume rule and are irreducible by KNN.
scikit-learn's LogisticRegression cannot consume NaN, so this model - and only
this model - adds two causal preprocessing steps INSIDE the modelling pipeline:

    causal training selection
      -> median imputation   (medians fitted on the training rows at T only)
      -> standardization     (StandardScaler fitted on those imputed rows only)
      -> Softmax fit
      -> prediction          (prediction rows are imputed and scaled with the
                              SAME training-fitted statistics)

Neither step ever sees a prediction row or a future observation, and neither is
written back to disk: the finalized KNN CSV is never modified and no
median-filled copy of it is created.

The Samartzis paper states that scale-sensitive models are normalized by
subtracting the training mean and dividing by the training standard deviation.
The released `logistic_model` does not itself show that scaler (the only
StandardScaler in all_models.py belongs to `mlp_model`), so for this
scale-sensitive model the project follows the paper's stated methodology rather
than the released code.

Search space, confirmed against the released registry: C ~ Uniform(0.5, 1.0),
one tuned hyperparameter, so the released MAX_EVALS = 20 x |space| gives 20
TPE evaluations over the same 12 pre-M6 tuning origins used by RF and LightGBM.

Outputs:
    Data/metadata/softmax_tuning_results.csv
    Results/Evaluation/softmax_m6_predictions.csv
    Results/Evaluation/softmax_m6_rps.csv
    logs/softmax_feature_baseline.log

Usage:
    python scripts/softmax_feature_baseline.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from hyperopt import Trials, hp, tpe
from hyperopt.early_stop import no_progress_loss
from hyperopt.fmin import fmin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_m6_rps import RANK_COLUMNS, build_ground_truth, load_prices  # noqa: E402
from evaluate_feature_baselines_m6 import (  # noqa: E402
    M6_SCHEDULE,
    load_m6_origins,
    score_against_official_truth,
)
from tune_feature_baselines import (  # noqa: E402
    FEATURE_COLUMNS,
    INFEASIBLE_LOSS,
    N_QUINTILES,
    RANDOM_SEED,
    TARGET_COLUMN,
    TRUTH_COLUMNS,
    ValidationError,
    _to_full_probability_matrix,
    adjust_probabilities,
    load_branch,
    load_tuning_origins,
    probability_report,
    rps_scores,
    split_at_origin,
)

BRANCH = "knn"
MODEL_NAME = "softmax"

# Released registry: 'C': hp.uniform('C', 0.5, 1.0) - a single tuned parameter.
C_SPACE = {"C": hp.uniform("C", 0.5, 1.0)}
EVALS_PER_DIMENSION = 20                       # source: MAX_EVALS = 20 * |space|
NO_PROGRESS_PATIENCE = 50                      # source: no_progress_loss(50)
MAX_ITER = 100                                 # scikit-learn default; converges
                                               # in ~24 lbfgs iterations once the
                                               # predictors are standardized

OUT_TUNING = PROJECT_ROOT / "Data" / "metadata" / "softmax_tuning_results.csv"
EVAL_DIR = PROJECT_ROOT / "Results" / "Evaluation"
OUT_PREDICTIONS = EVAL_DIR / "softmax_m6_predictions.csv"
OUT_RPS = EVAL_DIR / "softmax_m6_rps.csv"
LOG_PATH = PROJECT_ROOT / "logs" / "softmax_feature_baseline.log"

NAIVE_RPS = 0.16
EXPECTED_ROUNDS = 12

logger = logging.getLogger("softmax_feature_baseline")


# --------------------------------------------------------------------------- #
# Causal preprocessing (inside the pipeline; nothing is persisted)
# --------------------------------------------------------------------------- #

def causal_design(train: pd.DataFrame, predict: pd.DataFrame, origin: pd.Timestamp
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Median-impute then standardize, with every statistic fitted on ``train``.

    Returns (X_train, y_train, X_predict, diagnostics). Raises if a required
    training median cannot be formed - no substitute rule is invented.
    """
    raw_train = train[FEATURE_COLUMNS]
    raw_predict = predict[FEATURE_COLUMNS]

    medians = raw_train.median(skipna=True)
    unavailable = [c for c in FEATURE_COLUMNS if pd.isna(medians[c])]
    if unavailable:
        raise ValidationError(
            f"{origin.date()}: no training median available for {unavailable}; "
            "stopping rather than inventing another rule."
        )

    filled_train = raw_train.fillna(medians)
    filled_predict = raw_predict.fillna(medians)      # TRAINING medians, not its own
    if filled_train.isna().to_numpy().any() or filled_predict.isna().to_numpy().any():
        raise ValidationError(f"{origin.date()}: NaN survived median imputation.")

    scaler = StandardScaler().fit(filled_train)       # fitted on TRAINING rows only
    diagnostics = {
        "train_cells_imputed": int(raw_train.isna().to_numpy().sum()),
        "predict_cells_imputed": int(raw_predict.isna().to_numpy().sum()),
        "features_imputed": [c for c in FEATURE_COLUMNS
                             if raw_train[c].isna().any() or raw_predict[c].isna().any()],
    }
    return (scaler.transform(filled_train),
            train[TARGET_COLUMN].to_numpy(),
            scaler.transform(filled_predict),
            diagnostics)


def fit_predict_softmax(c_value: float, train: pd.DataFrame, predict: pd.DataFrame,
                        origin: pd.Timestamp) -> tuple[np.ndarray, dict]:
    """Fit multinomial logistic regression at one origin and return raw probabilities."""
    x_train, y_train, x_predict, diagnostics = causal_design(train, predict, origin)
    model = LogisticRegression(random_state=RANDOM_SEED, C=float(c_value),
                               max_iter=MAX_ITER).fit(x_train, y_train)
    diagnostics["lbfgs_iterations"] = int(np.max(model.n_iter_))
    diagnostics["converged"] = bool(np.max(model.n_iter_) < MAX_ITER)
    return (_to_full_probability_matrix(model.predict_proba(x_predict), model.classes_),
            diagnostics)


# --------------------------------------------------------------------------- #
# One candidate over a set of origins
# --------------------------------------------------------------------------- #

def evaluate_over_origins(c_value: float, frame: pd.DataFrame,
                          origins: list[pd.Timestamp]) -> dict:
    """Mean RPS over the given origins, refitting everything at each one."""
    per_origin, train_sizes, imputed = [], [], []
    minimum, maximum, row_error, iterations = 1.0, 0.0, 0.0, 0
    converged = True
    for origin in origins:
        train, predict = split_at_origin(frame, origin)
        raw, diagnostics = fit_predict_softmax(c_value, train, predict, origin)
        proba = adjust_probabilities(raw)
        report = probability_report(proba)
        if not report["valid"]:
            return {"feasible": False, "failed_origin": origin.date().isoformat(),
                    "failure_reason": report["reason"],
                    "min_probability": report["min"], "max_probability": report["max"],
                    "origins_completed": len(per_origin)}
        truth = predict[TRUTH_COLUMNS].to_numpy(dtype=float)
        per_origin.append(float(rps_scores(truth, proba).mean()))
        train_sizes.append(len(train))
        imputed.append(diagnostics["train_cells_imputed"]
                       + diagnostics["predict_cells_imputed"])
        minimum = min(minimum, report["min"])
        maximum = max(maximum, report["max"])
        row_error = max(row_error, report["row_sum_error"])
        iterations = max(iterations, diagnostics["lbfgs_iterations"])
        converged &= diagnostics["converged"]
    return {
        "feasible": True,
        "mean_rps": float(np.mean(per_origin)),
        "per_origin_rps": per_origin,
        "train_rows_min": int(min(train_sizes)),
        "train_rows_max": int(max(train_sizes)),
        "cells_imputed_total": int(sum(imputed)),
        "min_probability": minimum,
        "max_probability": maximum,
        "max_row_sum_error": row_error,
        "max_lbfgs_iterations": iterations,
        "all_converged": converged,
    }


# --------------------------------------------------------------------------- #
# Tuning
# --------------------------------------------------------------------------- #

def tune(frame: pd.DataFrame, origins: list[pd.Timestamp]) -> dict:
    budget = EVALS_PER_DIMENSION * len(C_SPACE)
    logger.info("Tuning softmax/%s: C ~ Uniform(0.5, 1.0), budget %d evaluations "
                "over %d pre-M6 origins", BRANCH, budget, len(origins))
    started = time.time()

    def objective(candidate: dict) -> dict:
        c_value = float(candidate["C"])
        result = evaluate_over_origins(c_value, frame, origins)
        if not result["feasible"]:
            logger.info("softmax: INFEASIBLE C=%.6f (%s at %s) -> loss %.1f",
                        c_value, result["failure_reason"], result["failed_origin"],
                        INFEASIBLE_LOSS)
            return {"loss": INFEASIBLE_LOSS, "status": "ok", "feasible": False,
                    "C": c_value, "diagnostics": result}
        return {"loss": result["mean_rps"], "status": "ok", "feasible": True,
                "C": c_value, "diagnostics": result}

    trials = Trials()
    fmin(fn=objective, space=C_SPACE, algo=tpe.suggest, trials=trials,
         max_evals=budget, early_stop_fn=no_progress_loss(NO_PROGRESS_PATIENCE),
         rstate=np.random.default_rng(RANDOM_SEED), show_progressbar=False)

    results = [r for r in trials.results if r.get("status") == "ok"]
    feasible = [r for r in results if r.get("feasible")]
    if not feasible:
        raise ValidationError("Every softmax candidate was infeasible.")
    best = min(feasible, key=lambda r: r["loss"])
    n_infeasible = len(results) - len(feasible)
    logger.info("softmax: best C=%.8f mean tuning RPS %.6f after %d evaluations "
                "(%d infeasible) in %.2f min", best["C"], best["loss"], len(results),
                n_infeasible, (time.time() - started) / 60)
    diagnostics = best["diagnostics"]
    return {
        "model": MODEL_NAME, "variant": BRANCH,
        "best_C": round(best["C"], 10),
        "mean_tuning_rps": round(best["loss"], 8),
        "evaluations_completed": len(results),
        "evaluation_budget": budget,
        "infeasible_trials": n_infeasible,
        "feasible_trials": len(feasible),
        "infeasible_pct": round(100 * n_infeasible / len(results), 2),
        "n_tuning_origins": len(origins),
        "first_tuning_origin": origins[0].date().isoformat(),
        "last_tuning_origin": origins[-1].date().isoformat(),
        "per_origin_rps": json.dumps([round(v, 8)
                                      for v in diagnostics["per_origin_rps"]]),
        "train_rows_min": diagnostics["train_rows_min"],
        "train_rows_max": diagnostics["train_rows_max"],
        "cells_imputed_total": diagnostics["cells_imputed_total"],
        "min_probability": round(diagnostics["min_probability"], 8),
        "max_probability": round(diagnostics["max_probability"], 8),
        "max_row_sum_error": diagnostics["max_row_sum_error"],
        "max_lbfgs_iterations": diagnostics["max_lbfgs_iterations"],
        "all_converged": diagnostics["all_converged"],
        "search_space": "C ~ Uniform(0.5, 1.0)",
        "imputation": "training-median, fitted per origin on eligible training rows",
        "standardization": "StandardScaler fitted per origin on imputed training rows",
        "probability_postprocessing": "column means shifted to 0.20 (released rule)",
        "feature_columns": json.dumps(FEATURE_COLUMNS),
        "random_seed": RANDOM_SEED,
        "sklearn_version": sklearn.__version__,
        "tuning_minutes": round((time.time() - started) / 60, 2),
        "tuned_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


# --------------------------------------------------------------------------- #
# Held-out M6 evaluation with the frozen C
# --------------------------------------------------------------------------- #

def evaluate_m6(c_value: float, frame: pd.DataFrame, rounds: pd.DataFrame
                ) -> tuple[pd.DataFrame | None, dict]:
    blocks, train_sizes, imputed = [], [], []
    minimum, maximum, row_error = 1.0, 0.0, 0.0
    started = time.time()
    for record in rounds.itertuples():
        origin = pd.Timestamp(record.origin_date)
        train, predict = split_at_origin(frame, origin)
        if (train["target_end_date"] > origin).any():
            raise ValidationError(f"Round {record.round}: an unrealised target "
                                  "entered training.")
        raw, diagnostics = fit_predict_softmax(c_value, train, predict, origin)
        proba = adjust_probabilities(raw)
        report = probability_report(proba)
        if not report["valid"]:
            logger.error("softmax: STOPPED at round %d (%s) - %s",
                         record.round, origin.date(), report["reason"])
            return None, {"status": "stopped_invalid_probability",
                          "failed_round": int(record.round),
                          "failed_origin": origin.date().isoformat(),
                          "failure_reason": report["reason"],
                          "rounds_completed": int(record.round) - 1,
                          "runtime_minutes": round((time.time() - started) / 60, 2)}
        minimum = min(minimum, report["min"])
        maximum = max(maximum, report["max"])
        row_error = max(row_error, report["row_sum_error"])
        train_sizes.append(len(train))
        imputed.append(diagnostics["train_cells_imputed"]
                       + diagnostics["predict_cells_imputed"])
        block = pd.DataFrame(proba, columns=RANK_COLUMNS)
        block.insert(0, "symbol", predict["symbol"].to_numpy())
        block.insert(0, "origin_date", origin.date().isoformat())
        block.insert(0, "round", int(record.round))
        block.insert(0, "variant", BRANCH)
        block.insert(0, "model", MODEL_NAME)
        blocks.append(block)
        logger.info("softmax round %2d (%s): trained on %d rows (%d cells imputed), "
                    "forecast %d assets, lbfgs %d iters",
                    record.round, origin.date(), len(train),
                    diagnostics["train_cells_imputed"] + diagnostics["predict_cells_imputed"],
                    len(predict), diagnostics["lbfgs_iterations"])
    return pd.concat(blocks, ignore_index=True), {
        "status": "complete", "failed_round": "", "failed_origin": "",
        "failure_reason": "", "rounds_completed": EXPECTED_ROUNDS,
        "train_rows_min": int(min(train_sizes)), "train_rows_max": int(max(train_sizes)),
        "cells_imputed_total": int(sum(imputed)),
        "min_probability": round(minimum, 8), "max_probability": round(maximum, 8),
        "max_row_sum_error": row_error,
        "runtime_minutes": round((time.time() - started) / 60, 2)}


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def run() -> None:
    frame = load_branch(BRANCH)
    tuning_origins = load_tuning_origins()
    rounds = load_m6_origins()

    # ---- 1. tune C on the pre-M6 origins only -----------------------------
    tuning = tune(frame, tuning_origins)
    OUT_TUNING.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([tuning]).to_csv(OUT_TUNING, index=False)
    logger.info("Wrote %s", OUT_TUNING)

    # ---- 2. freeze C, then evaluate the held-out M6 rounds -----------------
    frozen_c = float(tuning["best_C"])
    logger.info("C frozen at %.8f; beginning held-out M6 evaluation.", frozen_c)
    predictions, status = evaluate_m6(frozen_c, frame, rounds)

    logger.info("Forecasting finished; now loading the official M6 ground truth.")
    truth = build_ground_truth(load_prices(), pd.read_csv(M6_SCHEDULE))

    record = {"model": MODEL_NAME, "variant": BRANCH, "frozen_C": frozen_c,
              "mean_tuning_rps": tuning["mean_tuning_rps"], **status}
    if predictions is None:
        record.update({f"round_{i:02d}_rps": "" for i in range(1, 13)})
        record["mean_m6_rps"] = ""
        record["vs_naive"] = ""
    else:
        per_round = score_against_official_truth(predictions, truth)
        mean_rps = float(np.mean(list(per_round.values())))
        record.update({f"round_{i:02d}_rps": round(per_round[i], 8)
                       for i in range(1, 13)})
        record["mean_m6_rps"] = round(mean_rps, 8)
        record["vs_naive"] = round(mean_rps - NAIVE_RPS, 8)
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(OUT_PREDICTIONS, index=False)
        logger.info("Wrote %s", OUT_PREDICTIONS)
        logger.info("softmax: mean M6 RPS %.6f (naive %.6f, difference %+.6f)",
                    mean_rps, NAIVE_RPS, mean_rps - NAIVE_RPS)
    record["sklearn_version"] = sklearn.__version__
    record["random_seed"] = RANDOM_SEED
    record["evaluated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    pd.DataFrame([record]).to_csv(OUT_RPS, index=False)
    logger.info("Wrote %s", OUT_RPS)


def main() -> int:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)])
    try:
        run()
    except ValidationError as exc:
        logger.error("Validation failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
