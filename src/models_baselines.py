"""
Baseline models: linear regression and Random Forest

These are the comparison methods: A simple linear baseline and a strong
nonparametric ensemble. Both run under the SAME 5-fold CV protocol (same
fold indices, verified by fingerprint) that the GPR models will use.

Random Forest hyperparameters are tuned by NESTED cross-validation:
an inner 3-fold GridSearchCV runs inside each outer training fold, so the
reported outer-fold performance is never contaminated by tuning choices.
Linear regression has no hyperparameters and runs directly.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.figures import new_fig, save_fig, GREEN
from src.model_utils import (build_design_matrix, get_kfold,
                             fold_fingerprint, rmse, mae, r2,
                             summarize_folds, integrity_check)
from src.utils_logging import get_logger

# Modest, documented RF search grid (9 combinations). n_estimators is fixed
# at 300: more trees only stabilize, they do not overfit, so tuning it
# wastes compute. Depth and leaf size are the capacity-controlling knobs.
RF_GRID = {
    "max_depth": [None, 10, 20],
    "min_samples_leaf": [1, 5, 10],
}


def make_linear() -> Pipeline:
    """Standardize-then-fit linear regression. Scaling is fitted inside
    each training fold only (Pipeline), so no information leaks from the
    test fold into the scaler."""
    return Pipeline([("scaler", StandardScaler()),
                     ("model", LinearRegression())])


def make_rf_search() -> GridSearchCV:
    """Random Forest wrapped in an inner 3-fold grid search (nested CV).
    Trees are scale-invariant, so no scaler is needed."""
    rf = RandomForestRegressor(n_estimators=300,
                               random_state=config.RANDOM_SEED, n_jobs=-1)
    return GridSearchCV(rf, RF_GRID, cv=3,
                        scoring="neg_root_mean_squared_error", n_jobs=-1)


def run_cv(name: str, make_estimator, X, y, logger) -> pd.DataFrame:
    """Run one model through the shared outer 5-fold CV, collecting
    held-out RMSE/MAE/R^2 per fold plus train R^2 for the overfit check."""
    rows = []
    for k, (tr, te) in enumerate(get_kfold().split(X)):
        est = make_estimator()
        est.fit(X.iloc[tr], y.iloc[tr])
        pred_te = est.predict(X.iloc[te])
        pred_tr = est.predict(X.iloc[tr])
        row = {"model": name, "fold": k,
               "rmse": rmse(y.iloc[te], pred_te),
               "mae": mae(y.iloc[te], pred_te),
               "r2": r2(y.iloc[te], pred_te),
               "train_r2": r2(y.iloc[tr], pred_tr)}
        if isinstance(est, GridSearchCV):
            row["best_params"] = str(est.best_params_)
            logger.info(f"  fold {k}: inner-CV best params {est.best_params_}")
        rows.append(row)
    out = summarize_folds(rows)
    m = out[out["fold"] == "mean"].iloc[0]
    logger.info(f"[{name}] 5-fold CV: RMSE={m['rmse']:.2f}, MAE={m['mae']:.2f}, "
                f"R2={m['r2']:.3f}")
    integrity_check(name, m["r2"], m["train_r2"], logger)
    return out


def rf_feature_importances(X, y, logger) -> pd.DataFrame:
    """Fit a tuned RF on the full dataset and extract impurity-based
    feature importances — DESCRIPTIVE only (no performance is claimed from
    this fit); used later for qualitative comparison with ARD rankings."""
    search = make_rf_search()
    search.fit(X, y)
    logger.info(f"Full-data RF for importances — best params: "
                f"{search.best_params_}")
    imp = (pd.DataFrame({"feature": X.columns,
                         "importance": search.best_estimator_.feature_importances_})
           .sort_values("importance", ascending=False))
    path = config.TABLES_DIR / "rf_feature_importances.csv"
    imp.to_csv(path, index=False)
    logger.info(f"RF importances saved: {path}\n" + imp.to_string(index=False))

    fig, ax = new_fig(height=2.6)
    top = imp.iloc[::-1]  # horizontal bars read bottom-up
    ax.barh(top["feature"].str.replace("_", " "), top["importance"],
            color=GREEN, height=0.6)
    ax.set_xlabel("Random Forest impurity importance")
    save_fig(fig, "fig5_rf_importances", logger)
    return imp


def main():
    logger = get_logger("baselines")
    mt = pd.read_csv(config.PROCESSED_DIR / "model_table_primary.csv",
                     index_col="GLIMS_ID")
    X, y, _ = build_design_matrix(mt)
    logger.info(f"Design matrix: {X.shape[0]} glaciers x {X.shape[1]} features "
                f"({list(X.columns)})")
    logger.info(f"CV fold fingerprint (must match all model runs): "
                f"{fold_fingerprint(X)}")

    results = pd.concat([
        run_cv("linear", make_linear, X, y, logger),
        run_cv("random_forest", make_rf_search, X, y, logger),
    ], ignore_index=True)

    path = config.TABLES_DIR / "baseline_results.csv"
    results.to_csv(path, index=False)
    logger.info(f"Baseline results saved: {path}")

    rf_feature_importances(X, y, logger)
    logger.info("BASELINES COMPLETE")


if __name__ == "__main__":
    main()
