"""
Outlier-rule sensitivity analysis.

The exclusion of implausible positive-change glaciers (mapping artifacts)
involved judgment calls, so the key results must be shown to be robust to
that judgment. This script re-runs the headline analyses on all three
documented rules from config.OUTLIER_RULES:

  primary       min 1980 area 0.10 km², positive change capped at +20%
  no_exclusion  the full 3,165-glacier table, artifacts included
  strict        min area 0.25 km² (removes 22% of glaciers)

For each rule: GPR (Matérn 1.5 ARD, the selected model) under the shared
5-fold CV protocol -> RMSE/MAE/R²/PICP95, plus a full-data fit for the
top-3 ARD features; and Random Forest
for reference. If the ARD top-3 and the model ranking survive all three
rules, the conclusions do not hinge on the outlier handling.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src import config
from src.model_utils import (build_design_matrix, get_kfold, rmse, mae, r2,
                             summarize_folds)
from src.models_baselines import make_rf_search
from src.models_gpr import make_gpr, make_kernels, picp
from src.utils_logging import get_logger

N_RESTARTS_CV = 2
N_RESTARTS_FULL = 3


def gpr_cv_metrics(X, y, logger) -> dict:
    """Selected-GPR metrics under the shared 5-fold CV."""
    rows = []
    for k, (tr, te) in enumerate(get_kfold().split(X)):
        scaler = StandardScaler().fit(X.iloc[tr])
        gp = make_gpr(make_kernels(X.shape[1])["matern15"], N_RESTARTS_CV)
        gp.fit(scaler.transform(X.iloc[tr]), y.iloc[tr].values)
        mu, sigma = gp.predict(scaler.transform(X.iloc[te]), return_std=True)
        rows.append({"fold": k, "rmse": rmse(y.iloc[te], mu),
                     "mae": mae(y.iloc[te], mu), "r2": r2(y.iloc[te], mu),
                     "picp95": picp(y.iloc[te].values, mu, sigma)})
        logger.info(f"    GPR fold {k}: RMSE={rows[-1]['rmse']:.2f}")
    m = summarize_folds(rows)
    m = m[m["fold"] == "mean"].iloc[0]
    return {"gpr_rmse": m["rmse"], "gpr_mae": m["mae"],
            "gpr_r2": m["r2"], "gpr_picp95": m["picp95"]}


def ard_top3(X, y, logger) -> str:
    """Top-3 ARD features from a full-data fit (relevance = 1/length-scale
    on standardized inputs)."""
    scaler = StandardScaler().fit(X)
    gp = make_gpr(make_kernels(X.shape[1])["matern15"], N_RESTARTS_FULL)
    gp.fit(scaler.transform(X), y.values)
    base = gp.kernel_.k1
    while hasattr(base, "k1") and not hasattr(base, "length_scale"):
        base = base.k2 if hasattr(base.k1, "constant_value") else base.k1
    ranking = (pd.Series(base.length_scale, index=X.columns)
               .sort_values())  # shortest length-scale = most relevant
    top3 = " > ".join(ranking.index[:3])
    logger.info(f"    ARD top-3: {top3}")
    return top3


def rf_cv_metrics(X, y, logger) -> dict:
    """Random Forest under the same outer CV (nested tuning, as Task 3)."""
    rows = []
    for k, (tr, te) in enumerate(get_kfold().split(X)):
        est = make_rf_search()
        est.fit(X.iloc[tr], y.iloc[tr])
        pred = est.predict(X.iloc[te])
        rows.append({"fold": k, "rmse": rmse(y.iloc[te], pred),
                     "r2": r2(y.iloc[te], pred)})
    m = summarize_folds(rows)
    m = m[m["fold"] == "mean"].iloc[0]
    logger.info(f"    RF: RMSE={m['rmse']:.2f}, R2={m['r2']:.3f}")
    return {"rf_rmse": m["rmse"], "rf_r2": m["r2"]}


def main():
    logger = get_logger("sensitivity")
    rows = []
    for rule in config.OUTLIER_RULES:
        logger.info(f"--- Rule '{rule}' ---")
        mt = pd.read_csv(config.PROCESSED_DIR / f"model_table_{rule}.csv",
                         index_col="GLIMS_ID")
        X, y, _ = build_design_matrix(mt)
        logger.info(f"  {len(mt)} glaciers")
        row = {"rule": rule, "n_glaciers": len(mt)}
        row |= gpr_cv_metrics(X, y, logger)
        row |= rf_cv_metrics(X, y, logger)
        row["ard_top3"] = ard_top3(X, y, logger)
        rows.append(row)

    tab = pd.DataFrame(rows)
    path = config.TABLES_DIR / "sensitivity_analysis.csv"
    tab.to_csv(path, index=False)
    logger.info("Sensitivity analysis saved:\n" + tab.round(3).to_string(index=False))
    logger.info("SENSITIVITY ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
