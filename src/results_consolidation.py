"""
Results consolidation and report-support outputs.

Pure post-processing: reads every table the experimental phases produced
(never recomputes anything), and emits:

  1. outputs/tables/model_comparison.csv — the paper's headline Table:
     linear vs Random Forest vs selected GPR, mean ± std over the shared
     5-fold CV.
  2. outputs/tables/latex/*.tex — booktabs LaTeX fragments for every
     paper table, so the report can \\input{} real pipeline output and no
     number is ever hand-typed.
  3. outputs/report_numbers.md — every statistic the report text will
     cite, keyed by report section, each traceable to its source CSV.

"""

import pandas as pd

from src import config
from src.utils_logging import get_logger

LATEX_DIR = config.TABLES_DIR / "latex"


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(config.TABLES_DIR / f"{name}.csv")


def mean_std(df: pd.DataFrame, cols: list[str]) -> dict:
    """'mean ± std' strings computed from the per-fold rows of a
    summarize_folds()-style table. The stored summary rows are NOT used:
    they carry NaN in the model/kernel column, so they vanish when the
    caller filters by model name — recomputing from fold rows is robust."""
    folds = df[df["fold"].astype(str).str.isdigit()]
    return {c: f"{folds[c].mean():.2f} ± {folds[c].std():.2f}" for c in cols}


def build_model_comparison(logger) -> pd.DataFrame:
    """Headline model comparison from the stored per-fold results."""
    base = read("baseline_results")
    gpr = read("gpr_fold_results")
    rows = []
    for model in ["linear", "random_forest"]:
        sub = base[base["model"] == model]
        rows.append({"model": model, **mean_std(sub, ["rmse", "mae", "r2"]),
                     "uncertainty": "—"})
    sub = gpr[gpr["kernel"] == "matern15"]
    ms = mean_std(sub, ["rmse", "mae", "r2", "nlpd", "picp95"])
    rows.append({"model": "GPR Matérn(1.5) ARD",
                 "rmse": ms["rmse"], "mae": ms["mae"], "r2": ms["r2"],
                 "uncertainty": f"NLPD {ms['nlpd']}, PICP95 {ms['picp95']}"})
    tab = pd.DataFrame(rows)
    tab.to_csv(config.TABLES_DIR / "model_comparison.csv", index=False)
    logger.info("Model comparison saved:\n" + tab.to_string(index=False))
    return tab


def _esc_cell(value):
    """Escape one data cell for LaTeX. Headers and captions are authored
    as trusted LaTeX; only CSV-derived cell CONTENT passes through here."""
    if not isinstance(value, str):
        return value
    for raw, esc in [("&", "\\&"), ("%", "\\%"), ("_", "\\_"),
                     ("#", "\\#"), ("±", "$\\pm$"), ("—", "--"),
                     ("≥", "$\\geq$"), ("<", "$<$"), (">", "$>$"),
                     ("Matérn", "Mat\\'ern")]:
        value = value.replace(raw, esc)
    return value


def to_latex(df: pd.DataFrame, name: str, caption: str, label: str,
             logger, cols: dict | None = None, wide: bool = False):
    """Write one report-ready booktabs LaTeX fragment.

    cols: ordered {csv_column: short_header} — selects AND renames, so
          fragments fit the IEEE column width with readable headers
          (units/meaning live in the caption, not the header). Headers
          may contain LaTeX (e.g. $R^2$); data cells are escaped by
          _esc_cell so pdflatex compiles without extra packages.
    wide: use the table* environment (spans both columns).
    """
    LATEX_DIR.mkdir(parents=True, exist_ok=True)
    if cols:
        df = df[list(cols)].rename(columns=cols)
    df = df.map(_esc_cell)
    tex = df.to_latex(index=False, escape=False, caption=caption,
                      label=f"tab:{label}", position="t",
                      float_format=lambda x: f"{x:.6g}")
    if wide:
        tex = (tex.replace("\\begin{table}", "\\begin{table*}")
                  .replace("\\end{table}", "\\end{table*}"))
    # Center and shrink to footnotesize — IEEE tables are typically small.
    tex = tex.replace("\\begin{tabular}",
                      "\\centering\n\\footnotesize\n\\begin{tabular}")
    path = LATEX_DIR / f"{name}.tex"
    path.write_text(tex)
    logger.info(f"LaTeX fragment: {path}")


def export_latex_tables(model_comp: pd.DataFrame, logger):
    """Every paper table as a LaTeX fragment, numbers straight from CSVs.
    Headers are short ASCII; full meanings and units go in the captions."""
    basin = read("basin_summary")
    basin = basin.rename(columns={basin.columns[0]: "Basin"})
    to_latex(basin, "tab1_basin_summary",
             "Study data summary by river basin (primary-rule table). "
             "Areas in km$^2$; change in \\% of 1980 area.",
             "basin-summary", logger,
             cols={"Basin": "Basin", "n_glaciers": "N",
                   "total_area_1980_km2": "Area 1980",
                   "total_area_2010_km2": "Area 2010",
                   "mean_pct_change": "Mean chg.",
                   "median_pct_change": "Med. chg.",
                   "mean_elv_m": "Mean elev."})
    to_latex(read("preprocessing_row_counts"), "tab2_row_counts",
             "Data preparation row-count trail.", "row-counts", logger,
             cols={"step": "Step", "rows": "Rows"})
    to_latex(read("outlier_quantification"), "tab3_outliers",
             "Glaciers with positive (physically implausible) area change "
             "by 1980 size class (km$^2$).", "outliers", logger,
             cols={"size_class_km2_1980": "Size class", "n_glaciers": "N",
                   "n_positive": "Positive", "n_above_20pct": "$>$+20\\%",
                   "pct_positive": "\\% positive"})
    to_latex(model_comp, "tab4_model_comparison",
             "Predictive performance under the shared 5-fold CV "
             "(mean $\\pm$ std across folds). Only GPR provides "
             "predictive uncertainty.", "model-comparison", logger,
             cols={"model": "Model", "rmse": "RMSE", "mae": "MAE",
                   "r2": "$R^2$", "uncertainty": "Uncertainty"})
    to_latex(read("kernel_comparison").round(3), "tab5_kernel_comparison",
             "GPR kernel comparison: full-data log marginal likelihood "
             "and shared 5-fold CV metrics (RMSE/MAE in \\% points).",
             "kernel-comparison", logger,
             cols={"kernel": "Kernel", "log_marginal_likelihood": "LML",
                   "cv_rmse": "RMSE", "cv_mae": "MAE", "cv_r2": "$R^2$",
                   "cv_nlpd": "NLPD", "cv_picp95": "PICP95"})
    to_latex(read("optimizer_comparison").round(3),
             "tab6_optimizer_comparison",
             "Hyperparameter-optimization strategies on the fixed 80/20 "
             "split: dimensionality, cost, accuracy, and uncertainty "
             "quality. Dims = number of tuned hyperparameters; Evals = "
             "model fits or objective evaluations.",
             "optimizer-comparison", logger, wide=True,
             cols={"strategy": "Strategy", "n_hyperparameters": "Dims",
                   "n_fits_or_evals": "Evals",
                   "tuning_seconds": "Time (s)", "test_rmse": "RMSE",
                   "test_mae": "MAE", "test_r2": "$R^2$",
                   "test_nlpd": "NLPD", "test_picp95": "PICP95"})
    # Single-row metrics table reads better transposed (metric, value).
    unc = read("uncertainty_metrics").round(3).T.reset_index()
    unc.columns = ["Metric", "Value"]
    to_latex(unc, "tab7_uncertainty",
             "Out-of-fold uncertainty quality of the selected GPR "
             "(Mat\\'ern $\\nu{=}1.5$ ARD), all 3{,}050 glaciers.",
             "uncertainty", logger)
    to_latex(read("leave_one_basin_out").round(3), "tab8_lobo",
             "Leave-one-basin-out spatial validation: train on three "
             "basins, predict the fourth.", "lobo", logger,
             cols={"basin": "Basin", "n_test": "N", "rmse": "RMSE",
                   "mae": "MAE", "r2": "$R^2$", "nlpd": "NLPD",
                   "picp95": "PICP95"})
    to_latex(read("sensitivity_analysis").round(3), "tab9_sensitivity",
             "Key results under alternative outlier rules: the model "
             "ranking and ARD top-3 are rule-invariant.",
             "sensitivity", logger, wide=True,
             cols={"rule": "Rule", "n_glaciers": "N",
                   "gpr_rmse": "GPR RMSE", "gpr_r2": "GPR $R^2$",
                   "gpr_picp95": "PICP95", "rf_rmse": "RF RMSE",
                   "rf_r2": "RF $R^2$", "ard_top3": "ARD top-3"})
    to_latex(read("ard_lengthscales").round(3), "tab10_ard",
             "ARD length-scales of the selected kernel (full-data fit, "
             "standardized inputs); relevance $=1/\\ell$.", "ard", logger,
             cols={"feature": "Feature", "length_scale": "$\\ell$",
                   "relevance_1_over_ls": "Relevance"})


def write_report_numbers(logger):
    """report_numbers.md: the citable statistics, keyed by report section.
    Everything is read from the CSVs so the file is regenerable and every
    number has a file-level provenance."""
    unc = read("uncertainty_metrics").iloc[0]
    lobo = read("leave_one_basin_out").set_index("basin")
    kern = read("kernel_comparison").set_index("kernel")
    opt = read("optimizer_comparison").set_index("strategy")
    sens = read("sensitivity_analysis").set_index("rule")
    rc = read("preprocessing_row_counts").set_index("step")["rows"]
    basin = read("basin_summary")
    basin = basin.set_index(basin.columns[0])  # first column = basin name

    lines = [
        "# report_numbers.md — statistics for the report text",
        "",
        "Auto-generated by src/results_consolidation.py from outputs/tables/.",
        "Each block names its source CSV. Regenerate after any pipeline rerun.",
        "",
        "## §1 Abstract / headline numbers",
        f"- Glaciers modeled (primary rule): "
        f"{int(rc["after outlier rule 'primary'"])} "
        "(source: preprocessing_row_counts.csv)",
        f"- Best model: GPR Matérn(1.5) ARD — CV RMSE "
        f"{kern.loc['matern15', 'cv_rmse']:.2f}, R² "
        f"{kern.loc['matern15', 'cv_r2']:.3f} (kernel_comparison.csv)",
        f"- Calibration: PICP95 {unc['picp_95']:.3f}, NLPD "
        f"{unc['nlpd']:.3f} (uncertainty_metrics.csv)",
        "- Optimizer finding: CV-RMSE-tuned grid/BO collapse calibration "
        f"(PICP95 {opt.loc['grid_iso', 'test_picp95']:.3f} / "
        f"{opt.loc['bo_iso', 'test_picp95']:.3f}) vs MLL "
        f"{opt.loc['mll_ard', 'test_picp95']:.3f} (optimizer_comparison.csv)",
        "",
        "## §3 Problem & Dataset",
        f"- Raw rows {int(rc['raw rows (glacier-decade)'])}; unique "
        f"glaciers {int(rc['unique glaciers (GLIMS_ID)'])}; with both "
        f"1980+2010 areas {int(rc['modeling table pre-exclusion (1980 & 2010 areas)'])}; "
        f"after primary rule {int(rc["after outlier rule 'primary'"])}",
        "- Nepal-wide mean change (primary table): "
        f"{basin.loc['All Nepal', 'mean_pct_change']}% "
        f"(area {basin.loc['All Nepal', 'total_area_1980_km2']} -> "
        f"{basin.loc['All Nepal', 'total_area_2010_km2']} km²) "
        "(basin_summary.csv)",
        "",
        "## §6 Results — model comparison (model_comparison.csv)",
        "- See tab4; GPR beats RF and linear on RMSE/MAE/R².",
        "",
        "## §6 Results — kernels (kernel_comparison.csv)",
    ]
    for k in kern.index:
        lines.append(f"- {k}: LML {kern.loc[k, 'log_marginal_likelihood']:.1f}, "
                     f"CV RMSE {kern.loc[k, 'cv_rmse']:.2f}, "
                     f"NLPD {kern.loc[k, 'cv_nlpd']:.3f}")
    lines += [
        "",
        "## §6 Results — optimizers (optimizer_comparison.csv)",
    ]
    for s in opt.index:
        lines.append(f"- {s}: test RMSE {opt.loc[s, 'test_rmse']:.2f}, "
                     f"R² {opt.loc[s, 'test_r2']:.3f}, "
                     f"NLPD {opt.loc[s, 'test_nlpd']:.2f}, "
                     f"PICP95 {opt.loc[s, 'test_picp95']:.3f}, "
                     f"tuning {opt.loc[s, 'tuning_seconds']:.1f}s, "
                     f"{int(opt.loc[s, 'n_fits_or_evals'])} fits/evals")
    lines += [
        "",
        "## §6 Results — uncertainty (uncertainty_metrics.csv)",
        f"- OOF RMSE {unc['rmse']:.2f}, MAE {unc['mae']:.2f}, R² {unc['r2']:.3f}",
        f"- PICP 90/95/99: {unc['picp_90']:.3f} / {unc['picp_95']:.3f} / "
        f"{unc['picp_99']:.3f}; mean 95% width "
        f"{unc['mean_95_interval_width']:.1f}; z mean {unc['z_mean']:.2f}, "
        f"z std {unc['z_std']:.3f}",
        "",
        "## §6 Results — spatial validation (leave_one_basin_out.csv)",
    ]
    for b in lobo.index:
        lines.append(f"- {b}: n={int(lobo.loc[b, 'n_test'])}, RMSE "
                     f"{lobo.loc[b, 'rmse']:.2f}, R² {lobo.loc[b, 'r2']:.3f}, "
                     f"PICP95 {lobo.loc[b, 'picp95']:.3f}")
    lines += [
        "",
        "## §6/§8 — sensitivity (sensitivity_analysis.csv)",
    ]
    for r in sens.index:
        lines.append(f"- {r} (n={int(sens.loc[r, 'n_glaciers'])}): GPR RMSE "
                     f"{sens.loc[r, 'gpr_rmse']:.2f}, R² "
                     f"{sens.loc[r, 'gpr_r2']:.3f}, PICP95 "
                     f"{sens.loc[r, 'gpr_picp95']:.3f}; RF RMSE "
                     f"{sens.loc[r, 'rf_rmse']:.2f}; ARD top-3: "
                     f"{sens.loc[r, 'ard_top3']}")

    path = config.OUTPUTS_DIR / "report_numbers.md"
    path.write_text("\n".join(lines) + "\n")
    logger.info(f"Report numbers written: {path}")


def copy_report_assets(logger):
    """Copy table fragments and figure PDFs into report/ so the LaTeX
    project is self-contained (single-folder upload to Overleaf). The
    report always consumes pipeline outputs — never hand-edited copies."""
    import shutil
    report_tables = config.PROJECT_ROOT / "report" / "tables"
    report_figures = config.PROJECT_ROOT / "report" / "figures"
    report_tables.mkdir(parents=True, exist_ok=True)
    report_figures.mkdir(parents=True, exist_ok=True)
    n_t = n_f = 0
    for tex in sorted(LATEX_DIR.glob("*.tex")):
        shutil.copy2(tex, report_tables / tex.name)
        n_t += 1
    for pdf in sorted(config.FIGURES_DIR.glob("*.pdf")):
        shutil.copy2(pdf, report_figures / pdf.name)
        n_f += 1
    logger.info(f"Report assets copied: {n_t} table fragments -> "
                f"report/tables/, {n_f} figure PDFs -> report/figures/")


def main():
    logger = get_logger("consolidation")
    model_comp = build_model_comparison(logger)
    export_latex_tables(model_comp, logger)
    write_report_numbers(logger)
    copy_report_assets(logger)
    logger.info("RESULTS CONSOLIDATION COMPLETE")


if __name__ == "__main__":
    main()
