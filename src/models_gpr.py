"""
Gaussian Process Regression with ARD kernel comparison

Four kernels are compared, all in ARD form.

  rbf         C * RBF_ARD                + WhiteKernel
  matern15    C * Matern_ARD(nu=1.5)     + WhiteKernel
  matern25    C * Matern_ARD(nu=2.5)     + WhiteKernel
  rbf_linear  C * RBF_ARD + C * DotProduct + WhiteKernel   (composite:
              smooth nonlinear component + global linear trend)

Every kernel includes a ConstantKernel signal-variance scale and a
WhiteKernel observation-noise term, whose variance is learned jointly
with the other hyperparameters by maximizing the log marginal likelihood
(LML). Because WhiteKernel is part of the kernel, predict(return_std=True)
returns the std of OBSERVATIONS (latent f + noise), which is what
prediction intervals for real glaciers must cover.

Evaluation per kernel: RMSE, MAE, R^2, NLPD, 95% PICP on held-out folds;
LML, fit time, and ARD length-scales on the full-data fit.
"""

import time

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, ConstantKernel, DotProduct,
                                              Matern, WhiteKernel)
from sklearn.preprocessing import StandardScaler

from src import config
from src.figures import new_fig, save_fig, BLUE
from src.model_utils import (build_design_matrix, get_kfold,
                             fold_fingerprint, rmse, mae, r2,
                             summarize_folds, integrity_check)
from src.utils_logging import get_logger

# Restart counts for the LML optimizer (each restart re-runs L-BFGS from a
# random point in hyperparameter space to escape local optima). The
# full-data fit is the one whose length-scales we interpret, so it gets
# more restarts; CV folds use fewer to keep total runtime tractable.
N_RESTARTS_FULL = 5
N_RESTARTS_CV = 2

# Length-scales are bounded well away from degenerate solutions:
# lower bound 1e-2 (memorizing single points) / upper 1e3 (= feature
# effectively ignored; hitting this bound is itself an ARD statement).
LS_BOUNDS = (1e-2, 1e3)
NOISE_BOUNDS = (1e-3, 1e1)  # variance of standardized y


def make_kernels(n_features: int) -> dict:
    """The four ARD kernels compared. Fresh objects on every call because
    sklearn mutates kernel hyperparameters in-place during fitting."""
    ls0 = np.ones(n_features)  # initial length-scales (features standardized)

    def scaled(base):  # ConstantKernel learns the signal variance
        return ConstantKernel(1.0, (1e-3, 1e3)) * base

    noise = WhiteKernel(noise_level=0.5, noise_level_bounds=NOISE_BOUNDS)
    return {
        "rbf": scaled(RBF(ls0, LS_BOUNDS)) + noise,
        "matern15": scaled(Matern(ls0, LS_BOUNDS, nu=1.5)) + noise,
        "matern25": scaled(Matern(ls0, LS_BOUNDS, nu=2.5)) + noise,
        "rbf_linear": (scaled(RBF(ls0, LS_BOUNDS))
                       + ConstantKernel(0.1, (1e-3, 1e3))
                       * DotProduct(sigma_0=1.0)
                       + noise),
    }


def make_gpr(kernel, n_restarts: int) -> GaussianProcessRegressor:
    """GP with y-normalization (zero mean/unit variance internally;
    predictions are returned back on the original % scale)."""
    return GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                    n_restarts_optimizer=n_restarts,
                                    random_state=config.RANDOM_SEED)


def nlpd(y_true, mu, sigma) -> float:
    """Negative log predictive density (lower = better): how much
    probability mass the Gaussian predictive distribution puts on the
    actually observed value. THE metric that scores mean AND variance."""
    return float(-np.mean(stats.norm.logpdf(y_true, loc=mu, scale=sigma)))


def picp(y_true, mu, sigma, level: float = 0.95) -> float:
    """Prediction Interval Coverage Probability: fraction of held-out
    observations inside the central `level` interval. Calibrated -> ~level."""
    z = stats.norm.ppf(0.5 + level / 2)
    inside = (y_true >= mu - z * sigma) & (y_true <= mu + z * sigma)
    return float(np.mean(inside))


def run_kernel_cv(name: str, X, y, logger) -> pd.DataFrame:
    """Evaluate one kernel on the shared 5-fold CV. The feature scaler is
    fitted inside each training fold (no test-fold leakage)."""
    rows = []
    for k, (tr, te) in enumerate(get_kfold().split(X)):
        scaler = StandardScaler().fit(X.iloc[tr])
        Xtr, Xte = scaler.transform(X.iloc[tr]), scaler.transform(X.iloc[te])
        ytr, yte = y.iloc[tr].values, y.iloc[te].values

        t0 = time.perf_counter()
        gp = make_gpr(make_kernels(X.shape[1])[name], N_RESTARTS_CV)
        gp.fit(Xtr, ytr)
        fit_s = time.perf_counter() - t0

        mu, sigma = gp.predict(Xte, return_std=True)
        mu_tr = gp.predict(Xtr)
        rows.append({"kernel": name, "fold": k,
                     "rmse": rmse(yte, mu), "mae": mae(yte, mu),
                     "r2": r2(yte, mu), "train_r2": r2(ytr, mu_tr),
                     "nlpd": nlpd(yte, mu, sigma),
                     "picp95": picp(yte, mu, sigma),
                     "fit_seconds": fit_s})
        logger.info(f"  [{name}] fold {k}: RMSE={rows[-1]['rmse']:.2f}, "
                    f"NLPD={rows[-1]['nlpd']:.3f}, "
                    f"PICP95={rows[-1]['picp95']:.3f}, fit {fit_s:.0f}s")
    out = summarize_folds(rows)
    m = out[out["fold"] == "mean"].iloc[0]
    logger.info(f"[{name}] 5-fold CV: RMSE={m['rmse']:.2f}, MAE={m['mae']:.2f},"
                f" R2={m['r2']:.3f}, NLPD={m['nlpd']:.3f}, "
                f"PICP95={m['picp95']:.3f}")
    integrity_check(f"gpr_{name}", m["r2"], m["train_r2"], logger)
    return out


def fit_full(name: str, X, y, logger):
    """Full-data fit: source of the LML used for kernel selection and of
    the ARD length-scales used for feature-relevance ranking."""
    scaler = StandardScaler().fit(X)
    t0 = time.perf_counter()
    gp = make_gpr(make_kernels(X.shape[1])[name], N_RESTARTS_FULL)
    gp.fit(scaler.transform(X), y.values)
    fit_s = time.perf_counter() - t0
    lml = gp.log_marginal_likelihood_value_
    logger.info(f"[{name}] full-data fit {fit_s:.0f}s, LML={lml:.1f}")
    logger.info(f"  optimized kernel: {gp.kernel_}")
    return gp, lml, fit_s


def extract_ard(gp, feature_names, logger) -> pd.DataFrame:
    """ARD relevance from the fitted kernel: relevance = 1/length_scale on
    standardized inputs (short length-scale = the function varies quickly
    along that feature = the feature matters)."""
    # The ARD component is the first summand's base kernel (C * base).
    base = gp.kernel_.k1  # strip "+ WhiteKernel" (and "+ C*DotProduct")
    while hasattr(base, "k1") and not hasattr(base, "length_scale"):
        # descend product/sum structure until the ARD kernel is reached
        base = base.k2 if hasattr(base.k1, "constant_value") else base.k1
    ls = np.asarray(base.length_scale)
    tab = (pd.DataFrame({"feature": feature_names, "length_scale": ls,
                         "relevance_1_over_ls": 1.0 / ls})
           .sort_values("relevance_1_over_ls", ascending=False))
    at_upper = tab["length_scale"] >= LS_BOUNDS[1] * 0.99
    if at_upper.any():
        logger.info("  Length-scales AT UPPER BOUND (feature ~ignored): "
                    + ", ".join(tab.loc[at_upper, "feature"]))
    path = config.TABLES_DIR / "ard_lengthscales.csv"
    tab.to_csv(path, index=False)
    logger.info(f"ARD length-scales saved: {path}\n" + tab.to_string(index=False))
    return tab


def fig_ard(tab: pd.DataFrame, logger):
    fig, ax = new_fig(height=2.6)
    top = tab.iloc[::-1]
    ax.barh(top["feature"].str.replace("_", " "), top["relevance_1_over_ls"],
            color=BLUE, height=0.6)
    ax.set_xlabel("ARD relevance (1 / length-scale)")
    save_fig(fig, "fig6_ard_relevance", logger)


def main():
    logger = get_logger("gpr_kernels")
    mt = pd.read_csv(config.PROCESSED_DIR / "model_table_primary.csv",
                     index_col="GLIMS_ID")
    X, y, _ = build_design_matrix(mt)
    logger.info(f"Design matrix: {X.shape[0]} x {X.shape[1]}")
    logger.info(f"CV fold fingerprint (must equal baselines run): "
                f"{fold_fingerprint(X)}")

    kernel_names = list(make_kernels(X.shape[1]).keys())
    fold_results, comparison = [], []
    for name in kernel_names:
        logger.info(f"--- Kernel: {name} ---")
        cv_res = run_kernel_cv(name, X, y, logger)
        fold_results.append(cv_res)
        gp, lml, fit_s = fit_full(name, X, y, logger)
        m = cv_res[cv_res["fold"] == "mean"].iloc[0]
        comparison.append({"kernel": name, "log_marginal_likelihood": lml,
                           "cv_rmse": m["rmse"], "cv_mae": m["mae"],
                           "cv_r2": m["r2"], "cv_nlpd": m["nlpd"],
                           "cv_picp95": m["picp95"],
                           "full_fit_seconds": fit_s})
        # Keep the full-data model of every kernel for later phases.
        if name == kernel_names[0]:
            best_gp, best_name, best_lml = gp, name, lml
        elif lml > best_lml:
            best_gp, best_name, best_lml = gp, name, lml

    pd.concat(fold_results, ignore_index=True).to_csv(
        config.TABLES_DIR / "gpr_fold_results.csv", index=False)
    comp = pd.DataFrame(comparison)
    comp.to_csv(config.TABLES_DIR / "kernel_comparison.csv", index=False)
    logger.info("Kernel comparison saved:\n" + comp.to_string(index=False))

    logger.info(f"Selected kernel by LML: {best_name} (LML={best_lml:.1f}) — "
                f"cross-check against cv_rmse/cv_nlpd in the table above")
    ard_tab = extract_ard(best_gp, list(X.columns), logger)
    fig_ard(ard_tab, logger)
    logger.info("GPR KERNEL COMPARISON COMPLETE")


if __name__ == "__main__":
    main()
