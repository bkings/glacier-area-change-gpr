"""
Single end-to-end entry point: runs every pipeline stage in dependency order, from the
raw ICIMOD shapefile to every table, figure, and appendix artifact used
in the report.

    python run_all.py

The Gaussian-process stages dominate (exact GP inference is O(n^3)). Each
stage writes its own timestamped log to outputs/logs/ and can also be run
individually, e.g. `python -m src.models_gpr`.

Stage order and report-appendix mapping:
    A.1 data_prep              raw shapefile -> modeling tables
    A.2 eda                    descriptive figures + basin summary
    A.3 models_baselines       linear + Random Forest (shared 5-fold CV)
    A.4 models_gpr             GPR ARD kernel comparison
    A.5 optimize               MLL vs grid vs Bayesian optimization
    A.6 evaluate               calibration, LOBO, partial dependence
    A.7 sensitivity            outlier-rule robustness
    A.8 results_consolidation  paper tables (LaTeX) + report numbers
    A.9 make_appendix          code listing + evidence index
"""

import time

from src import (data_prep, eda, models_baselines, models_gpr, optimize,
                 evaluate, sensitivity, results_consolidation, make_appendix)

STAGES = [
    ("data_prep", data_prep.main),
    ("eda", eda.main),
    ("models_baselines", models_baselines.main),
    ("models_gpr", models_gpr.main),
    ("optimize", optimize.main),
    ("evaluate", evaluate.main),
    ("sensitivity", sensitivity.main),
    ("results_consolidation", results_consolidation.main),
    ("make_appendix", make_appendix.main),
]


def main():
    t0 = time.perf_counter()
    for name, stage_main in STAGES:
        print(f"\n{'#' * 70}\n# RUN_ALL stage: {name}\n{'#' * 70}")
        stage_main()
    print(f"\nRUN_ALL COMPLETE in {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
