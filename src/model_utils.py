"""
Shared modeling utilities: design matrix, CV protocol, metrics.

Every model in the project (linear, Random Forest, all GPR variants) gets
its data and cross-validation folds from THIS module, so comparisons are
like-for-like: identical features, identical fold indices, identical
metric definitions.
"""

import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src import config


def build_design_matrix(mt: pd.DataFrame):
    """Build (X, y, groups) from a modeling table.

    X = the 8 config.FEATURES + one-hot river basin (drop_first=True so
        linear regression keeps full rank; Mahakali becomes the reference
        category). All columns numeric.
    y = pct_change_1980_2010 (percent, negative = shrinkage).
    groups = raw Basin labels, used only for leave-one-basin-out CV.
    """
    basin_dummies = pd.get_dummies(mt[config.BASIN_COL], prefix="basin",
                                   drop_first=True, dtype=float)
    X = pd.concat([mt[config.FEATURES].astype(float), basin_dummies], axis=1)
    y = mt[config.TARGET].astype(float)
    groups = mt[config.BASIN_COL]
    return X, y, groups


def get_kfold() -> KFold:
    """The single shared CV splitter: 5-fold, shuffled, fixed global seed."""
    return KFold(n_splits=config.N_CV_FOLDS, shuffle=True,
                 random_state=config.RANDOM_SEED)


def fold_fingerprint(X: pd.DataFrame) -> str:
    """Hash of the exact fold assignment, logged by every model run so the
    appendix can PROVE all models saw identical folds (criterion 2.4)."""
    idx = np.zeros(len(X), dtype=int)
    for k, (_, test) in enumerate(get_kfold().split(X)):
        idx[test] = k
    return hashlib.md5(idx.tobytes()).hexdigest()[:12]


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    ss_res = np.sum((y_true - np.asarray(y_pred)) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot)


def summarize_folds(rows: list[dict]) -> pd.DataFrame:
    """Per-fold metric rows -> DataFrame with a mean +/- std summary row."""
    df = pd.DataFrame(rows)
    num = df.select_dtypes("number")
    summary = num.mean().to_dict() | {"fold": "mean"}
    summary_sd = num.std().to_dict() | {"fold": "std"}
    return pd.concat([df, pd.DataFrame([summary, summary_sd])],
                     ignore_index=True)


def integrity_check(model_name: str, test_r2: float, train_r2: float, logger):
    """Evaluation-integrity tripwire (professor's guidance, PROJECT_PLAN.md).

    Held-out R^2 above the suspicion threshold on this noisy environmental
    dataset almost certainly means leakage or fold contamination — halt
    and audit before reporting. Also logs the train-CV gap (overfit check).
    """
    logger.info(f"  [{model_name}] overfit check: train R2={train_r2:.3f}, "
                f"test R2={test_r2:.3f}, gap={train_r2 - test_r2:.3f}")
    if test_r2 > config.R2_SUSPICION_THRESHOLD:
        logger.info(f"  *** WARNING [{model_name}]: held-out R2 {test_r2:.3f} "
                    f"exceeds suspicion threshold "
                    f"{config.R2_SUSPICION_THRESHOLD} — AUDIT FOR LEAKAGE "
                    f"BEFORE REPORTING ***")
