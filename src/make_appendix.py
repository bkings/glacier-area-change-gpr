"""
Appendix evidence pack generator
"""

from datetime import datetime
from pathlib import Path

from src import config
from src.utils_logging import get_logger

APPENDIX_DIR = config.OUTPUTS_DIR / "appendix"

# (appendix label, title, file) in report order. A.0 groups the shared
# infrastructure modules that every task imports.
SECTIONS = [
    ("A.0", "Shared infrastructure (configuration, logging, figure style, "
            "CV protocol and metrics)",
     ["src/config.py", "src/utils_logging.py", "src/figures.py",
      "src/model_utils.py"]),
    ("A.1", "Task 1 — Data preparation pipeline", ["src/data_prep.py"]),
    ("A.2", "Task 2 — Exploratory data analysis", ["src/eda.py"]),
    ("A.3", "Task 3 — Baseline models (linear, Random Forest)",
     ["src/models_baselines.py"]),
    ("A.4", "Task 4 — GPR with ARD kernel comparison", ["src/models_gpr.py"]),
    ("A.5", "Task 5 — Hyperparameter-optimization comparison",
     ["src/optimize.py"]),
    ("A.6", "Task 6 — Uncertainty evaluation and spatial validation",
     ["src/evaluate.py"]),
    ("A.7", "Task 7 — Outlier-rule sensitivity analysis",
     ["src/sensitivity.py"]),
    ("A.8", "Task 8 — Results consolidation and report numbers",
     ["src/results_consolidation.py"]),
    ("A.9", "Task 9 — Reproduction entry point and appendix generator",
     ["run_all.py", "src/make_appendix.py"]),
]


def write_code_listing(logger):
    lines = [
        "# Appendix A — Code Listing",
        "",
        "Complete, original source code of the analysis pipeline "
        "(Python; no screenshots). Sections A.1-A.9 mirror the task "
        "structure of the main text, which cross-references them as "
        "'see Appendix A.x'. Shared infrastructure is in A.0.",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} by src/make_appendix.py.",
        "",
    ]
    for label, title, files in SECTIONS:
        lines += [f"## {label} {title}", ""]
        for f in files:
            path = config.PROJECT_ROOT / f
            lines += [f"### `{f}`", "", "```python",
                      path.read_text().rstrip(), "```", ""]
    out = APPENDIX_DIR / "code_listing.md"
    out.write_text("\n".join(lines))
    logger.info(f"Code listing written: {out} "
                f"({out.stat().st_size / 1024:.0f} KB)")


def _index_dir(root: Path, pattern: str) -> list[str]:
    rows = []
    for p in sorted(root.glob(pattern)):
        ts = datetime.fromtimestamp(p.stat().st_mtime)
        rows.append(f"- `{p.relative_to(config.PROJECT_ROOT)}` "
                    f"({p.stat().st_size / 1024:.0f} KB, {ts:%Y-%m-%d %H:%M})")
    return rows


def write_evidence_index(logger):
    lines = [
        "# Appendix B — Evidence Index",
        "",
        "Index of the experimental evidence trail. Every run log is "
        "timestamped and records the global random seed (42), Python and "
        "package versions, row counts before/after each preprocessing "
        "step, per-fold metrics, and optimized kernels. All tables and "
        "figures are generated programmatically by the pipeline.",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} by src/make_appendix.py.",
        "",
        "## Run logs (outputs/logs/)", "",
        *_index_dir(config.LOGS_DIR, "*.log"),
        "",
        "## Result tables (outputs/tables/)", "",
        *_index_dir(config.TABLES_DIR, "*.csv"),
        "",
        "## LaTeX table fragments (outputs/tables/latex/)", "",
        *_index_dir(config.TABLES_DIR / "latex", "*.tex"),
        "",
        "## Figures (outputs/figures/)", "",
        *_index_dir(config.FIGURES_DIR, "*.*"),
        "",
        "## Report-support numbers", "",
        *_index_dir(config.OUTPUTS_DIR, "report_numbers.md"),
    ]
    out = APPENDIX_DIR / "evidence_index.md"
    out.write_text("\n".join(lines))
    logger.info(f"Evidence index written: {out}")


def main():
    logger = get_logger("make_appendix")
    APPENDIX_DIR.mkdir(parents=True, exist_ok=True)
    write_code_listing(logger)
    write_evidence_index(logger)
    logger.info("APPENDIX EVIDENCE PACK COMPLETE")


if __name__ == "__main__":
    main()
