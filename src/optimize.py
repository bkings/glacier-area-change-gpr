"""
Hyperparameter-optimization strategy comparison

Compare three strategies for setting the GP hyperparameters
of the SELECTED kernel family:

  (a) mll_ard   log-marginal-likelihood maximization (sklearn's internal
                L-BFGS with restarts) on the FULL ARD kernel — 13
                hyperparameters (11 length-scales + signal + noise).
  (a') mll_iso  the same MLL machinery on an ISOTROPIC kernel — 3
                hyperparameters — included so grid/BO have a like-for-like
                MLL opponent (see search-space note below).
  (b) grid_iso  exhaustive grid search over the isotropic space, scored
                by inner 3-fold CV RMSE on the training split.
  (c) bo_iso    Bayesian optimization over the SAME
                isotropic bounds and SAME inner-CV objective as the grid —
                so (b) vs (c) is a pure optimizer-vs-optimizer comparison.

One fixed 80/20 train/test split (seed 42). Each strategy tunes
on the training split only (MLL maximizes training LML; grid/BO minimize
inner 3-fold CV RMSE within the training split), then the tuned model is
refitted on the full training split and evaluated ONCE on the untouched
test split. Kernel quality under full 5-fold CV was already established
in Task 4; this task isolates the OPTIMIZERS, where wall-clock cost is a
headline metric and must be measured on identical work.
"""

import time

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from skopt import gp_minimize
from skopt.space import Real

from src import config
from src.figures import new_fig, save_fig, BLUE, GRAY
from src.model_utils import build_design_matrix, rmse, mae, r2
from src.models_gpr import make_kernels, nlpd, picp
from src.utils_logging import get_logger

# Isotropic search space shared by mll_iso, grid_iso, and bo_iso.
# Bounds in standardized-feature / standardized-target units.
C_BOUNDS = (0.1, 10.0)          # signal variance
LS_BOUNDS_ISO = (0.1, 100.0)    # shared length-scale
NOISE_BOUNDS_ISO = (0.01, 2.0)  # observation-noise variance

# Grid: 3 x 6 x 5 = 90 combinations, log-spaced (scale hyperparameters
# live on multiplicative scales, so log spacing is the correct coverage).
GRID_C = np.geomspace(*C_BOUNDS, 3)
GRID_LS = np.geomspace(*LS_BOUNDS_ISO, 6)
GRID_NOISE = np.geomspace(*NOISE_BOUNDS_ISO, 5)

N_BO_CALLS = 40           # BO budget: under half the grid's 90 points
N_INNER_FOLDS = 3         # inner CV for the grid/BO objective
TEST_FRACTION = 0.2


def iso_kernel(c: float, ls: float, noise: float):
    """Isotropic Matérn(1.5) kernel with FIXED hyperparameters (used with
    optimizer=None so the GP fit is a single Cholesky, no tuning)."""
    return (ConstantKernel(c, "fixed")
            * Matern(length_scale=ls, length_scale_bounds="fixed", nu=1.5)
            + WhiteKernel(noise, "fixed"))


def fit_fixed(Xtr, ytr, c, ls, noise) -> GaussianProcessRegressor:
    """Fit a GP with fixed hyperparameters (no internal optimization)."""
    gp = GaussianProcessRegressor(kernel=iso_kernel(c, ls, noise),
                                  optimizer=None, normalize_y=True,
                                  random_state=config.RANDOM_SEED)
    gp.fit(Xtr, ytr)
    return gp


def inner_cv_rmse(Xtr, ytr, c, ls, noise) -> float:
    """The tuning objective for grid search and BO: mean held-out RMSE
    over an inner 3-fold CV of the TRAINING split (test split untouched)."""
    kf = KFold(N_INNER_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
    scores = []
    for tr, va in kf.split(Xtr):
        gp = fit_fixed(Xtr[tr], ytr[tr], c, ls, noise)
        scores.append(rmse(ytr[va], gp.predict(Xtr[va])))
    return float(np.mean(scores))


def evaluate(gp, Xte, yte) -> dict:
    """Single evaluation on the untouched test split (point + interval)."""
    mu, sigma = gp.predict(Xte, return_std=True)
    return {"test_rmse": rmse(yte, mu), "test_mae": mae(yte, mu),
            "test_r2": r2(yte, mu), "test_nlpd": nlpd(yte, mu, sigma),
            "test_picp95": picp(yte, mu, sigma)}


def strategy_mll(Xtr, ytr, ard: bool, logger):
    """(a)/(a'): maximize training log marginal likelihood via sklearn's
    internal L-BFGS with 5 restarts. Tuning and fitting are one step."""
    if ard:
        kernel = make_kernels(Xtr.shape[1])["matern15"]  # full ARD form
        n_hyper = Xtr.shape[1] + 2
    else:
        kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                  * Matern(1.0, (1e-2, 1e3), nu=1.5)
                  + WhiteKernel(0.5, (1e-3, 1e1)))
        n_hyper = 3
    t0 = time.perf_counter()
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                  n_restarts_optimizer=5,
                                  random_state=config.RANDOM_SEED)
    gp.fit(Xtr, ytr)
    seconds = time.perf_counter() - t0
    logger.info(f"  optimized kernel: {gp.kernel_}")
    row = {"strategy": "mll_ard" if ard else "mll_iso",
           "n_hyperparameters": n_hyper,
           "n_fits_or_evals": 6,  # 1 + 5 restarted L-BFGS runs
           "tuning_seconds": seconds,
           "inner_cv_best_rmse": np.nan,  # MLL does not use the CV objective
           "best_params": str(gp.kernel_)}
    return row, gp


def strategy_grid(Xtr, ytr, logger):
    """(b): exhaustive log-spaced grid, inner-CV RMSE objective."""
    t0 = time.perf_counter()
    best, best_score = None, np.inf
    n_evals = 0
    for c in GRID_C:
        for ls in GRID_LS:
            for noise in GRID_NOISE:
                score = inner_cv_rmse(Xtr, ytr, c, ls, noise)
                n_evals += 1
                if score < best_score:
                    best, best_score = (c, ls, noise), score
    gp = fit_fixed(Xtr, ytr, *best)  # refit best on full training split
    seconds = time.perf_counter() - t0
    logger.info(f"  grid best: C={best[0]:.3g}, ls={best[1]:.3g}, "
                f"noise={best[2]:.3g} (inner-CV RMSE {best_score:.2f}, "
                f"{n_evals} points x {N_INNER_FOLDS} inner fits)")
    row = {"strategy": "grid_iso", "n_hyperparameters": 3,
           "n_fits_or_evals": n_evals * N_INNER_FOLDS + 1,
           "tuning_seconds": seconds,
           "inner_cv_best_rmse": best_score,
           "best_params": f"C={best[0]:.3g}, ls={best[1]:.3g}, "
                          f"noise={best[2]:.3g}"}
    return row, gp


def strategy_bo(Xtr, ytr, logger):
    """(c): Bayesian optimization over the same space and objective as the
    grid. gp_minimize builds a surrogate GP over the objective surface and
    picks each next evaluation by expected improvement."""
    space = [Real(*C_BOUNDS, prior="log-uniform", name="c"),
             Real(*LS_BOUNDS_ISO, prior="log-uniform", name="ls"),
             Real(*NOISE_BOUNDS_ISO, prior="log-uniform", name="noise")]
    t0 = time.perf_counter()
    result = gp_minimize(lambda p: inner_cv_rmse(Xtr, ytr, *p), space,
                         n_calls=N_BO_CALLS, n_initial_points=10,
                         random_state=config.RANDOM_SEED)
    best = result.x
    gp = fit_fixed(Xtr, ytr, *best)
    seconds = time.perf_counter() - t0
    logger.info(f"  BO best: C={best[0]:.3g}, ls={best[1]:.3g}, "
                f"noise={best[2]:.3g} (inner-CV RMSE {result.fun:.2f}, "
                f"{N_BO_CALLS} calls)")
    row = {"strategy": "bo_iso", "n_hyperparameters": 3,
           "n_fits_or_evals": N_BO_CALLS * N_INNER_FOLDS + 1,
           "tuning_seconds": seconds,
           "inner_cv_best_rmse": float(result.fun),
           "best_params": f"C={best[0]:.3g}, ls={best[1]:.3g}, "
                          f"noise={best[2]:.3g}"}
    return row, gp, result


def fig_bo_convergence(result, grid_best_rmse: float, logger):
    """Figure 7: BO running best vs evaluation number, with the grid's
    final best as a reference line — shows how quickly BO reaches (or
    beats) the exhaustive answer with a fraction of the evaluations."""
    running_best = np.minimum.accumulate(result.func_vals)
    pd.DataFrame({"call": np.arange(1, len(running_best) + 1),
                  "objective_rmse": result.func_vals,
                  "running_best": running_best}).to_csv(
        config.TABLES_DIR / "bo_convergence.csv", index=False)

    fig, ax = new_fig()
    ax.plot(np.arange(1, len(running_best) + 1), running_best,
            color=BLUE, linewidth=1.5, drawstyle="steps-post",
            label="BO running best")
    ax.axhline(grid_best_rmse, color=GRAY, linewidth=1, linestyle="--",
               label="grid best (90 points)")
    ax.set_xlabel("BO objective evaluations")
    ax.set_ylabel("Inner-CV RMSE (%)")
    ax.legend(frameon=False)
    save_fig(fig, "fig7_bo_convergence", logger)


def main():
    logger = get_logger("optimize")
    mt = pd.read_csv(config.PROCESSED_DIR / "model_table_primary.csv",
                     index_col="GLIMS_ID")
    X, y, _ = build_design_matrix(mt)

    # One fixed split; scaler fitted on the training split only.
    Xtr_raw, Xte_raw, ytr, yte = train_test_split(
        X, y, test_size=TEST_FRACTION, random_state=config.RANDOM_SEED)
    scaler = StandardScaler().fit(Xtr_raw)
    Xtr, Xte = scaler.transform(Xtr_raw), scaler.transform(Xte_raw)
    ytr, yte = ytr.values, yte.values
    logger.info(f"Split: train {Xtr.shape[0]} / test {Xte.shape[0]} "
                f"(seed {config.RANDOM_SEED})")

    rows = []
    logger.info("--- (a) MLL, full ARD kernel ---")
    row, gp = strategy_mll(Xtr, ytr, ard=True, logger=logger)
    rows.append(row | evaluate(gp, Xte, yte))

    logger.info("--- (a') MLL, isotropic kernel ---")
    row, gp = strategy_mll(Xtr, ytr, ard=False, logger=logger)
    rows.append(row | evaluate(gp, Xte, yte))

    logger.info("--- (b) Grid search, isotropic ---")
    grid_row, gp = strategy_grid(Xtr, ytr, logger)
    rows.append(grid_row | evaluate(gp, Xte, yte))

    logger.info("--- (c) Bayesian optimization, isotropic ---")
    row, gp, bo_result = strategy_bo(Xtr, ytr, logger)
    rows.append(row | evaluate(gp, Xte, yte))

    comp = pd.DataFrame(rows)
    path = config.TABLES_DIR / "optimizer_comparison.csv"
    comp.to_csv(path, index=False)
    logger.info("Optimizer comparison saved:\n"
                + comp.drop(columns="best_params").to_string(index=False))

    fig_bo_convergence(bo_result, grid_row["inner_cv_best_rmse"], logger)
    logger.info("OPTIMIZER COMPARISON COMPLETE")


if __name__ == "__main__":
    main()
