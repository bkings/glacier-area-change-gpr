"""
Project configuration — single source of truth.

All paths, the global random seed, feature definitions, and outlier-rule
specifications live here so that every pipeline stage (data preparation,
modeling, evaluation, figures) reads identical settings. This is a core
reproducibility requirement of the coursework (marking criterion 5.1).

Project: Modeling Decadal Glacier Area Change in the Nepal Himalaya Using
Optimized Gaussian Process Regression (MSc Advanced ML, Task 1).
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
# One global seed used for CV splitting, Random Forest, GP optimizer restarts,
# and Bayesian optimization. Confirmed with the student (Decisions Log).
RANDOM_SEED = 42

# Number of cross-validation folds shared by ALL models (GPR, RF, linear),
# with identical fold indices, so accuracy comparisons are like-for-like.
N_CV_FOLDS = 5

# ---------------------------------------------------------------------------
# Paths (project root = parent of src/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Raw ICIMOD shapefile bundle — read-only, never modified by the pipeline.
# License: CC BY 4.0, attribution to ICIMOD required in the paper.
# Source: https://rds.icimod.org/metadata/a3e3c0d6-73b2-460a-9836-90e261469b68
DATASET_DIR = PROJECT_ROOT / "dataset" / "Glacier Area Change in Nepal 1980 - 2010"
SHAPEFILE_PATH = DATASET_DIR / "data" / "Glacier_1980_1990_2000_2010.shp"
METADATA_XML = DATASET_DIR / "data" / "Glacier_1980_1990_2000_2010.shp.xml"
METADATA_PDF = DATASET_DIR / "metadata" / "nepal_glacier_1980_to_2010.pdf"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
LOGS_DIR = OUTPUTS_DIR / "logs"

# ---------------------------------------------------------------------------
# Ground-truth dataset facts (verified by the student's inspection runs).
# The Phase 0 sanity check asserts these; a mismatch halts the pipeline.
# ---------------------------------------------------------------------------
EXPECTED_RAW_SHAPE = (14659, 22)
EXPECTED_ROWS_PER_YEAR = {1980: 3430, 1990: 3656, 2000: 3765, 2010: 3808}
EXPECTED_UNIQUE_GLIMS_IDS = 4130
EXPECTED_MODEL_TABLE_ROWS = 3165  # glaciers with both 1980 and 2010 areas

# ---------------------------------------------------------------------------
# Features and target
# ---------------------------------------------------------------------------
# Target: percentage area change over the full 1980-2010 span.
TARGET = "pct_change_1980_2010"

# Numeric model features. Notes:
#  - `area_1980` (initial area, from the pivot) is used instead of 2010 area
#    to avoid target leakage (the target is computed from the 2010 area).
#  - `Thickness` and `Reserve` are EXCLUDED: they are area-derived empirical
#    scaling estimates, not measurements, and would leak area information.
#  - Aspect is circular (0-360 deg) and therefore enters as sin/cos.
NUMERIC_FEATURES = [
    "Elv_min",
    "Elv_mean",
    "Elv_max",
    "Slope_mean",
    "aspect_sin",
    "aspect_cos",
    "area_1980",
]

# Decoded from digit 1 of the 5-digit GLIMS/WGMS morphological Class code
# (Rau et al. 2005, GLIMS Glacier Classification Manual): 5 = valley glacier,
# 6 = mountain glacier, 7 = glacieret/snowfield. NOTE: debris coverage is
# digit 9 of the extended GLIMS scheme and is NOT present in this dataset's
# 5-digit codes, so the clean-ice vs debris-covered distinction is not
# recoverable from the attribute table (documented limitation).
CLASS_FEATURES = ["is_valley_glacier"]

# Full feature list used by every model (Basin is one-hot encoded on top of
# this at modeling time; it is kept as a raw column for grouped validation).
FEATURES = NUMERIC_FEATURES + CLASS_FEATURES

# Grouping column for leave-one-basin-out spatial validation.
BASIN_COL = "Basin"

# Explicitly banned features (leakage risk) — data_prep asserts none of these
# ever reach the modeling feature list.
LEAKAGE_BANNED = ["Thickness", "Reserve", "Area_SqKm", "area_2010",
                  "Shape_area", "Shape_len"]

# ---------------------------------------------------------------------------
# Outlier-handling rules (Decisions Log #4)
# ---------------------------------------------------------------------------
# Each rule maps a name -> dict of thresholds. `min_area_1980` is the minimum
# initial (1980) area in km^2; `max_pct_change` caps positive change (values
# above it are treated as mapping artifacts and excluded). None = no bound.
# "primary" is the headline analysis; the others feed the sensitivity study.
OUTLIER_RULES = {
    "primary": {"min_area_1980": 0.10, "max_pct_change": 20.0},
    "no_exclusion": {"min_area_1980": None, "max_pct_change": None},
    "strict": {"min_area_1980": 0.25, "max_pct_change": 20.0},
}

# ---------------------------------------------------------------------------
# Evaluation-integrity tripwire (professor's guidance — see PROJECT_PLAN.md)
# ---------------------------------------------------------------------------
# If any model reaches held-out R^2 above this, the pipeline flags a loud
# warning and the result must be audited for leakage before being reported.
R2_SUSPICION_THRESHOLD = 0.90
