"""
Evidence-trail logging utilities.

Timestamped logs, seeds, and row counts
before/after every preprocessing step. Every pipeline stage obtains its
logger from `get_logger(stage_name)`, which writes simultaneously to the
console and to a timestamped file in outputs/logs/, and records the global
random seed and package versions at the top of every log.
"""

import logging
import platform
import sys
import warnings
from datetime import datetime

from src import config

# scikit-learn 1.9 emits a harmless internal UserWarning from every parallel
# worker dispatch ("`sklearn.utils.parallel.delayed` should be used with
# ..."). With hundreds of CV fits this floods the terminal and buries the
# evidence log, so this one specific message is suppressed project-wide.
# It has no effect on results; all substantive warnings still surface.
warnings.filterwarnings(
    "ignore",
    message=".*sklearn.utils.parallel.delayed.*",
    category=UserWarning,
)


def get_logger(stage: str) -> logging.Logger:
    """Create a logger writing to console AND outputs/logs/<stage>_<ts>.log.

    The log header records the timestamp, global seed, Python version, and
    key package versions so every run is a self-contained evidence record.
    """
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = config.LOGS_DIR / f"{stage}_{timestamp}.log"

    logger = logging.getLogger(f"glacier.{stage}.{timestamp}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in (logging.FileHandler(log_path, encoding="utf-8"),
                    logging.StreamHandler(sys.stdout)):
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    # Environment header — reproducibility evidence for the appendix.
    import numpy, pandas, sklearn  # noqa: E401 (versions logged, not used)
    logger.info("=" * 70)
    logger.info(f"STAGE: {stage}")
    logger.info(f"GLOBAL RANDOM SEED: {config.RANDOM_SEED}")
    logger.info(f"Python {platform.python_version()} on {platform.platform()}")
    logger.info(f"numpy {numpy.__version__} | pandas {pandas.__version__} | "
                f"scikit-learn {sklearn.__version__}")
    logger.info(f"Log file: {log_path}")
    logger.info("=" * 70)
    return logger


def log_rows(logger: logging.Logger, label: str, n_before: int, n_after: int):
    """Log a row-count change through a preprocessing step (criterion 2.4)."""
    dropped = n_before - n_after
    logger.info(f"{label}: {n_before} rows -> {n_after} rows "
                f"({dropped} removed, {100 * dropped / max(n_before, 1):.2f}%)")
