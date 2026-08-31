"""
Uncertainty evaluation and spatial validation.

Deep-dive on the final model (Matérn nu=1.5 ARD, hyperparameters by
marginal-likelihood maximization):

  1. OUT-OF-FOLD PREDICTIONS: the shared 5-fold CV is run once more and
     the held-out (mu, sigma) for every glacier is collected, so every
     calibration statistic below is computed on predictions the model
     made for glaciers it never saw.
  2. CALIBRATION CURVE (Fig 8): empirical coverage of central prediction
     intervals vs nominal level, 50%..99%. A calibrated model tracks the
     diagonal (Kuleshov et al. 2018 framing).
  3. PREDICTED VS ACTUAL with 95% intervals.
  4. LEAVE-ONE-BASIN-OUT (LOBO): train on three river basins, predict the
     fourth — the honest test of spatial generalization to unmapped
     regions, far harder than random CV because basins differ
     systematically.
  5. HYPOTHESIS SUPPORT (Fig 10): model-implied partial dependence of
     predicted change on Elv_max (other features at their medians) with
     the 95% band — the model's answer to "is loss concentrated at lower
     elevations?", complementing the empirical Fig 2.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

from src import config
from src.figures import (new_fig, save_fig, BLUE, GRAY, ORANGE, VERMILLION,
                         SINGLE_COL)
from src.model_utils import (build_design_matrix, get_kfold,
                             fold_fingerprint, rmse, mae, r2)
from src.models_gpr import make_gpr, make_kernels, nlpd, picp
from src.utils_logging import get_logger

N_RESTARTS = 2          # per-fold restarts (as in Task 4's CV protocol)
SELECTED_KERNEL = "matern15"


def fit_predict(Xtr, ytr, Xte) -> tuple[np.ndarray, np.ndarray]:
    """Standardize on the training rows only, fit the selected GP, return
    held-out predictive mean and std (std includes observation noise via
    the WhiteKernel — intervals cover observations, not just latent f)."""
    scaler = StandardScaler().fit(Xtr)
    gp = make_gpr(make_kernels(Xtr.shape[1])[SELECTED_KERNEL], N_RESTARTS)
    gp.fit(scaler.transform(Xtr), ytr)
    return gp.predict(scaler.transform(Xte), return_std=True)


def collect_oof(X, y, logger) -> pd.DataFrame:
    """Out-of-fold (mu, sigma) for every glacier under the shared CV."""
    oof = pd.DataFrame(index=X.index,
                       columns=["y", "mu", "sigma", "fold"], dtype=float)
    for k, (tr, te) in enumerate(get_kfold().split(X)):
        mu, sigma = fit_predict(X.iloc[tr], y.iloc[tr].values, X.iloc[te])
        oof.iloc[te, 0] = y.iloc[te].values
        oof.iloc[te, 1] = mu
        oof.iloc[te, 2] = sigma
        oof.iloc[te, 3] = k
        logger.info(f"  fold {k}: RMSE={rmse(y.iloc[te], mu):.2f}, "
                    f"PICP95={picp(y.iloc[te].values, mu, sigma):.3f}")
    path = config.TABLES_DIR / "predictions_oof.csv"
    oof.to_csv(path)
    logger.info(f"Out-of-fold predictions saved: {path}")
    return oof


def uncertainty_metrics(oof: pd.DataFrame, logger) -> None:
    """Headline uncertainty table + standardized-residual diagnostics."""
    y, mu, s = oof["y"].values, oof["mu"].values, oof["sigma"].values
    z = (y - mu) / s  # standardized residuals: calibrated -> mean 0, std 1
    tab = pd.DataFrame([{
        "rmse": rmse(y, mu), "mae": mae(y, mu), "r2": r2(y, mu),
        "nlpd": nlpd(y, mu, s),
        "picp_90": picp(y, mu, s, 0.90),
        "picp_95": picp(y, mu, s, 0.95),
        "picp_99": picp(y, mu, s, 0.99),
        "mean_95_interval_width": float(np.mean(2 * 1.96 * s)),
        "z_mean": float(z.mean()), "z_std": float(z.std()),
    }])
    path = config.TABLES_DIR / "uncertainty_metrics.csv"
    tab.to_csv(path, index=False)
    logger.info(f"Uncertainty metrics saved: {path}\n"
                + tab.round(3).to_string(index=False))


def fig_calibration(oof: pd.DataFrame, logger) -> None:
    """Figure 8: reliability diagram — empirical vs nominal coverage."""
    y, mu, s = oof["y"].values, oof["mu"].values, oof["sigma"].values
    nominal = np.linspace(0.50, 0.99, 25)
    empirical = [picp(y, mu, s, lv) for lv in nominal]
    pd.DataFrame({"nominal": nominal, "empirical": empirical}).to_csv(
        config.TABLES_DIR / "calibration_curve.csv", index=False)

    fig, ax = new_fig(width=SINGLE_COL, height=2.8)
    ax.plot([0.5, 1.0], [0.5, 1.0], color=GRAY, linewidth=1,
            linestyle="--", label="perfect calibration")
    ax.plot(nominal, empirical, color=BLUE, linewidth=1.5,
            label="GPR (out-of-fold)")
    ax.set_xlabel("Nominal coverage level")
    ax.set_ylabel("Empirical coverage")
    ax.legend(frameon=False, loc="upper left")
    ax.set_aspect("equal")
    save_fig(fig, "fig8_calibration", logger)


def fig_pred_vs_actual(oof: pd.DataFrame, logger) -> None:
    """Figure 9: predicted vs actual with 95% intervals. All glaciers as
    faint marks; a seeded random subsample of 60 carries visible error
    bars (3,050 overlapping bars would be unreadable)."""
    rng = np.random.default_rng(config.RANDOM_SEED)
    sub = oof.iloc[rng.choice(len(oof), 60, replace=False)]

    fig, ax = new_fig(width=SINGLE_COL, height=3.0)
    lims = (-100, 40)
    ax.plot(lims, lims, color=GRAY, linewidth=1, linestyle="--")
    ax.scatter(oof["y"], oof["mu"], s=2, alpha=0.15, color=BLUE,
               edgecolors="none", rasterized=True)
    ax.errorbar(sub["y"], sub["mu"], yerr=1.96 * sub["sigma"], fmt="o",
                markersize=2.5, color=VERMILLION, ecolor=VERMILLION,
                elinewidth=0.6, capsize=0, alpha=0.85,
                label="95% interval (n=60 sample)")
    ax.set_xlabel("Actual area change (%)")
    ax.set_ylabel("Predicted area change (%)")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.legend(frameon=False, loc="upper left")
    save_fig(fig, "fig9_pred_vs_actual", logger)


def run_lobo(X, y, groups, logger) -> pd.DataFrame:
    """Leave-one-basin-out: the spatial-generalization stress test."""
    rows = []
    for basin in sorted(groups.unique()):
        te = groups == basin
        tr = ~te
        mu, sigma = fit_predict(X[tr], y[tr].values, X[te])
        yte = y[te].values
        rows.append({"basin": basin, "n_test": int(te.sum()),
                     "rmse": rmse(yte, mu), "mae": mae(yte, mu),
                     "r2": r2(yte, mu), "nlpd": nlpd(yte, mu, sigma),
                     "picp95": picp(yte, mu, sigma)})
        logger.info(f"  LOBO {basin}: n={te.sum()}, RMSE={rows[-1]['rmse']:.2f}, "
                    f"R2={rows[-1]['r2']:.3f}, PICP95={rows[-1]['picp95']:.3f}")
    tab = pd.DataFrame(rows)
    # Pooled row: every glacier predicted from a model that never saw its
    # basin (concatenation of the four held-out sets).
    w = tab["n_test"]
    tab.loc[len(tab)] = ["pooled", int(w.sum()),
                         float(np.sqrt((w * tab["rmse"] ** 2).sum() / w.sum())),
                         float((w * tab["mae"]).sum() / w.sum()),
                         np.nan,  # pooled R2 not weightable this way
                         float((w * tab["nlpd"]).sum() / w.sum()),
                         float((w * tab["picp95"]).sum() / w.sum())]
    path = config.TABLES_DIR / "leave_one_basin_out.csv"
    tab.to_csv(path, index=False)
    logger.info(f"LOBO table saved: {path}\n" + tab.round(3).to_string(index=False))

    # NOTE on basin dummies under LOBO: the held-out basin's dummy column
    # is all-zero in training, so the model treats it as the reference
    # category — a conservative, honest handling logged here for the record.
    logger.info("LOBO note: held-out basin's one-hot column is constant "
                "zero during training (falls back to reference category).")
    return tab


def fig_partial_dependence(X, y, logger) -> None:
    """Figure 10: model-implied effect of Elv_max with 95% band.

    One GP fitted on all data (as in Task 4); prediction along an Elv_max
    sweep with every other feature frozen at its median — the model's
    ceteris-paribus answer to the elevation hypothesis. Interpretation is
    correlational, and the collinearity of the elevation features means
    the medians choice matters; both caveats go in the caption."""
    scaler = StandardScaler().fit(X)
    gp = make_gpr(make_kernels(X.shape[1])[SELECTED_KERNEL], N_RESTARTS)
    gp.fit(scaler.transform(X), y.values)

    # Sweep lower bound: Elv_max below the median Elv_mean would describe a
    # physically impossible glacier (max below mean elevation) since the
    # other features are frozen at their medians — starting there kept the
    # first version of this figure extrapolating to <-100% change.
    lo = max(X["Elv_max"].quantile(0.02), float(X["Elv_mean"].median()))
    sweep = np.linspace(lo, X["Elv_max"].quantile(0.98), 60)
    grid = pd.DataFrame(np.tile(X.median().values, (60, 1)),
                        columns=X.columns)
    grid["Elv_max"] = sweep
    mu, sigma = gp.predict(scaler.transform(grid), return_std=True)

    fig, ax = new_fig()
    ax.fill_between(sweep, mu - 1.96 * sigma, mu + 1.96 * sigma,
                    color=BLUE, alpha=0.18, linewidth=0,
                    label="95% interval")
    ax.plot(sweep, mu, color=BLUE, linewidth=1.5, label="GP mean")
    ax.set_xlabel("Maximum elevation (m a.s.l.), other features at median")
    ax.set_ylabel("Predicted area change (%)")
    ax.legend(frameon=False, loc="lower right")
    save_fig(fig, "fig10_partial_dependence", logger)


def main(pd_only: bool = False):
    """Full run; pass --pd-only to regenerate only Figure 10
    (partial dependence) without repeating the CV and LOBO fits."""
    logger = get_logger("evaluate")
    mt = pd.read_csv(config.PROCESSED_DIR / "model_table_primary.csv",
                     index_col="GLIMS_ID")
    X, y, groups = build_design_matrix(mt)
    logger.info(f"Model under evaluation: GPR {SELECTED_KERNEL} ARD "
                f"(MLL-optimized). Fold fingerprint: {fold_fingerprint(X)}")

    if not pd_only:
        logger.info("--- Out-of-fold predictions (shared 5-fold CV) ---")
        oof = collect_oof(X, y, logger)
        uncertainty_metrics(oof, logger)
        fig_calibration(oof, logger)
        fig_pred_vs_actual(oof, logger)

        logger.info("--- Leave-one-basin-out ---")
        run_lobo(X, y, groups, logger)

    logger.info("--- Partial dependence (elevation hypothesis) ---")
    fig_partial_dependence(X, y, logger)
    logger.info("UNCERTAINTY EVALUATION COMPLETE"
                + (" (partial-dependence figure only)" if pd_only else ""))


if __name__ == "__main__":
    import sys
    main(pd_only="--pd-only" in sys.argv)
