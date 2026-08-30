"""
Exploratory data analysis

Consumes the primary modeling table produced by src/data_prep.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import config
from src.figures import (new_fig, save_fig, BLUE, ORANGE, GRAY, VERMILLION,
                         SINGLE_COL, FULL_WIDTH)
from src.utils_logging import get_logger


def load_model_table(rule: str = "primary") -> pd.DataFrame:
    """Load a processed modeling table written by data_prep."""
    return pd.read_csv(config.PROCESSED_DIR / f"model_table_{rule}.csv",
                       index_col="GLIMS_ID")


def fig_target_distribution(mt: pd.DataFrame, logger):
    """Figure 1: histogram of the target variable.

    A single-series magnitude view -> histogram in the primary blue, with
    the mean marked. Shows the strong overall retreat signal and the
    spread that the models must explain.
    """
    fig, ax = new_fig()
    ax.hist(mt[config.TARGET], bins=40, color=BLUE, edgecolor="white",
            linewidth=0.3)
    mean = mt[config.TARGET].mean()
    ax.axvline(mean, color=VERMILLION, linewidth=1)
    ax.annotate(f"mean {mean:.1f}%", xy=(mean, ax.get_ylim()[1] * 0.95),
                xytext=(4, 0), textcoords="offset points",
                color=VERMILLION, fontsize=7, va="top")
    ax.set_xlabel("Glacier area change 1980–2010 (%)")
    ax.set_ylabel("Number of glaciers")
    save_fig(fig, "fig1_target_distribution", logger)


def fig_change_vs_elevation(mt: pd.DataFrame, logger):
    """Figure 2: the hypothesis figure — change vs mean elevation.

    Scatter of all glaciers (small translucent marks so density reads
    without overplotting) plus a binned-median line, which is the honest
    trend summary for 3k noisy points. If the literature expectation
    holds, the median line should rise (less loss) with elevation.
    """
    fig, ax = new_fig(width=SINGLE_COL, height=2.6)
    ax.scatter(mt["Elv_mean"], mt[config.TARGET], s=3, alpha=0.25,
               color=BLUE, edgecolors="none", rasterized=True)

    # Binned medians: 12 equal-count elevation bins -> robust trend line.
    bins = pd.qcut(mt["Elv_mean"], 12)
    med = mt.groupby(bins, observed=True).agg(
        x=("Elv_mean", "median"), y=(config.TARGET, "median"))
    ax.plot(med["x"], med["y"], color=VERMILLION, linewidth=1.5,
            marker="o", markersize=2.5, label="binned median")

    ax.set_xlabel("Mean elevation (m a.s.l.)")
    ax.set_ylabel("Area change 1980–2010 (%)")
    ax.legend(frameon=False, loc="lower right")
    save_fig(fig, "fig2_change_vs_elevation", logger)


def fig_change_by_basin(mt: pd.DataFrame, logger):
    """Figure 3: distribution of change per river basin (box plot).

    Identity across four basins -> one box per basin, ordered west->east
    (Mahakali, Karnali, Gandaki, Koshi), single hue since the basins are
    positions, not competing series."""
    order = ["Mahakali", "Karnali", "Gandaki", "Koshi"]  # west -> east
    data = [mt.loc[mt[config.BASIN_COL] == b, config.TARGET] for b in order]
    fig, ax = new_fig()
    bp = ax.boxplot(data, tick_labels=[f"{b}\n(n={len(d)})"
                                       for b, d in zip(order, data)],
                    showfliers=False, patch_artist=True, widths=0.55,
                    medianprops={"color": VERMILLION, "linewidth": 1.2})
    for box in bp["boxes"]:
        box.set(facecolor=BLUE, alpha=0.35, edgecolor=BLUE, linewidth=0.8)
    ax.set_ylabel("Area change 1980–2010 (%)")
    ax.set_xlabel("River basin (west → east)")
    save_fig(fig, "fig3_change_by_basin", logger)


def fig_correlation_matrix(mt: pd.DataFrame, logger):
    """Figure 4: Pearson correlations among features and target.

    Polarity data (positive/negative) -> diverging colormap (RdBu) with a
    neutral midpoint at 0; coefficients printed in each cell so the
    figure is readable in grayscale print too."""
    cols = config.FEATURES + [config.TARGET]
    corr = mt[cols].corr()

    fig, ax = new_fig(width=SINGLE_COL, height=3.2)
    ax.grid(False)
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    labels = [c.replace("_", " ") for c in cols]
    ax.set_xticks(range(len(cols)), labels, rotation=60, ha="right")
    ax.set_yticks(range(len(cols)), labels)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5,
                    color="white" if abs(v) > 0.6 else "#333333")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    save_fig(fig, "fig4_correlation_matrix", logger)
    return corr


def table_basin_summary(mt: pd.DataFrame, logger) -> pd.DataFrame:
    """Table 1: per-basin study summary for the dataset section."""
    g = mt.groupby(config.BASIN_COL)
    tab = pd.DataFrame({
        "n_glaciers": g.size(),
        "total_area_1980_km2": g["area_1980"].sum().round(1),
        "total_area_2010_km2": g["area_2010"].sum().round(1),
        "mean_pct_change": g[config.TARGET].mean().round(1),
        "median_pct_change": g[config.TARGET].median().round(1),
        "mean_elv_m": g["Elv_mean"].mean().round(0),
        "n_valley_glaciers": g["is_valley_glacier"].sum(),
    }).sort_values("mean_elv_m")
    # Nepal-wide totals row for the paper.
    tab.loc["All Nepal"] = [len(mt), mt["area_1980"].sum().round(1),
                            mt["area_2010"].sum().round(1),
                            mt[config.TARGET].mean().round(1),
                            mt[config.TARGET].median().round(1),
                            mt["Elv_mean"].mean().round(0),
                            mt["is_valley_glacier"].sum()]
    path = config.TABLES_DIR / "basin_summary.csv"
    tab.to_csv(path)
    logger.info(f"Basin summary saved: {path}\n" + tab.to_string())
    return tab


def main():
    logger = get_logger("eda")
    mt = load_model_table("primary")
    logger.info(f"Loaded model_table_primary: {mt.shape[0]} glaciers, "
                f"{mt.shape[1]} columns")

    fig_target_distribution(mt, logger)
    fig_change_vs_elevation(mt, logger)
    fig_change_by_basin(mt, logger)
    corr = fig_correlation_matrix(mt, logger)
    table_basin_summary(mt, logger)

    # Log the correlations most relevant to the hypothesis for the record.
    logger.info("Correlation of target with elevation features: "
                + ", ".join(f"{c}={corr.loc[c, config.TARGET]:.3f}"
                            for c in ["Elv_min", "Elv_mean", "Elv_max",
                                      "area_1980"]))
    logger.info("EDA COMPLETE")


if __name__ == "__main__":
    main()
