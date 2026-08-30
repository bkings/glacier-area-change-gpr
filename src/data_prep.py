"""
Data preparation pipeline

Builds the per-glacier modeling table from the raw ICIMOD shapefile:
  1. Load the long-format inventory (one row per glacier per decade)
  2. Pivot Area_SqKm by Year per GLIMS_ID (aggfunc="sum") to a wide table.
  3. Construct the target: pct_change_1980_2010.
  4. Attach static topographic features from the 2010 rows; use the 1980
     area (not 2010) as the size feature to avoid target leakage.
  5. Encode circular Aspect as sin/cos.
  6. Decode the GLIMS/WGMS morphological Class code (digit 1: glacier
     primary type). Debris cover is NOT encoded in the 5-digit codes —
     documented limitation, see PROJECT_PLAN.md.
  7. Quantify positive-change outliers, then apply the documented
     exclusion rules (primary + sensitivity variants from config).

"""

import numpy as np
import pandas as pd
import geopandas as gpd

from src import config
from src.utils_logging import get_logger, log_rows


def load_raw(logger) -> pd.DataFrame:
    """Load the raw shapefile attribute table (geometry skipped for speed).

    All shapefile sibling files (.shp/.dbf/.shx/...) must sit in one
    directory; a misplaced .shx raises pyogrio.errors.DataSourceError.
    """
    logger.info(f"Loading shapefile: {config.SHAPEFILE_PATH}")
    df = gpd.read_file(config.SHAPEFILE_PATH, ignore_geometry=True)
    logger.info(f"Loaded raw attribute table: {df.shape[0]} rows x "
                f"{df.shape[1]} columns")
    return df


def sanity_check(df: pd.DataFrame, logger) -> None:
    """Phase 0 gate: assert the data matches the verified ground truth.

    Expected values come from the student's own inspection runs and are
    hard-coded in config; any mismatch means the input data changed and
    every downstream number would be untrustworthy, so we halt.
    """
    n_rows = len(df)
    # +1 column allowance: geometry was skipped on load (22 incl. geometry).
    assert n_rows == config.EXPECTED_RAW_SHAPE[0], \
        f"Row count {n_rows} != expected {config.EXPECTED_RAW_SHAPE[0]}"

    per_year = df["Year"].value_counts().to_dict()
    per_year = {int(k): int(v) for k, v in per_year.items()}
    assert per_year == config.EXPECTED_ROWS_PER_YEAR, \
        f"Rows per year {per_year} != expected {config.EXPECTED_ROWS_PER_YEAR}"

    n_ids = df["GLIMS_ID"].nunique()
    assert n_ids == config.EXPECTED_UNIQUE_GLIMS_IDS, \
        f"Unique GLIMS_IDs {n_ids} != expected {config.EXPECTED_UNIQUE_GLIMS_IDS}"

    # GLIMS_ID must be unique within each year for the 2010 feature join
    # to be one-to-one (verified during inspection; asserted here).
    dup = df.groupby("Year")["GLIMS_ID"].apply(lambda s: s.duplicated().sum())
    assert (dup == 0).all(), f"Duplicate GLIMS_IDs within a year: {dup.to_dict()}"

    logger.info("SANITY CHECK PASSED: raw data matches verified ground truth")
    logger.info(f"  rows={n_rows}, per-year={per_year}, unique GLIMS_ID={n_ids}")


def decode_class(class_codes: pd.Series, logger) -> pd.DataFrame:
    """Decode digit 1 of the GLIMS/WGMS morphological classification code.

    Per the GLIMS Glacier Classification Manual (Rau et al. 2005, Table 3),
    the 5-digit code covers: [1] primary classification, [2] form,
    [3] frontal characteristic, [4] longitudinal profile, [5] nourishment.
    Digit-1 values present in this inventory: 5 = valley glacier,
    6 = mountain glacier, 7 = glacieret/snowfield.

    Debris coverage of the tongue is digit 9 of the *extended* GLIMS scheme
    and is absent from these 5-digit codes, so clean-ice vs debris-covered
    CANNOT be recovered here — reported honestly as a limitation.
    """
    codes = class_codes.astype(str)
    lengths = codes.str.len().value_counts().to_dict()
    logger.info(f"Class code lengths observed: {lengths}")
    if set(lengths) - {5}:
        odd = codes[codes.str.len() != 5]
        logger.info(f"  Non-5-digit codes ({len(odd)} rows): "
                    f"{odd.value_counts().to_dict()} — digit 1 still decodable")

    primary = codes.str[0]
    primary_names = {"5": "valley glacier", "6": "mountain glacier",
                     "7": "glacieret/snowfield"}
    counts = primary.value_counts().to_dict()
    logger.info("Primary classification (digit 1) counts: "
                + ", ".join(f"{primary_names.get(k, 'code ' + k)}={v}"
                            for k, v in counts.items()))
    logger.info("NOTE: debris cover is digit 9 of the extended GLIMS scheme; "
                "NOT present in 5-digit codes -> clean vs debris-covered "
                "distinction unavailable (documented limitation).")
    return pd.DataFrame({
        "glacier_type": primary.map(primary_names).fillna("other"),
        "is_valley_glacier": (primary == "5").astype(int),
    }, index=class_codes.index)


def build_model_table(df: pd.DataFrame, logger) -> pd.DataFrame:
    """
    Returns the pre-exclusion modeling table (one row per glacier that has
    both a 1980 and a 2010 area; expected 3,165 rows).
    """
    # -- Pivot areas to wide format (sum aggregates any fragments sharing
    #    a parent GLIMS_ID; with unique IDs per year this is a plain pivot).
    wide = df.pivot_table(index="GLIMS_ID", columns="Year",
                          values="Area_SqKm", aggfunc="sum")
    wide.columns = [f"area_{int(c)}" for c in wide.columns]
    logger.info(f"Pivoted wide area table: {wide.shape[0]} glaciers")

    # -- Target: percentage area change over the full 1980-2010 span.
    wide[config.TARGET] = (100.0 * (wide["area_2010"] - wide["area_1980"])
                           / wide["area_1980"])

    # -- Static features from the 2010 rows (topography changes little at
    #    this precision; 2010 has the most complete, best-mapped coverage).
    feat_cols = ["Elv_min", "Elv_mean", "Elv_max", "Slope_mean",
                 "Aspect", "Class", "Basin", "Sub_Basin",
                 "Longitude", "Latitude"]
    feats = df.loc[df["Year"] == 2010].set_index("GLIMS_ID")[feat_cols]

    # -- Circular aspect (degrees) -> sin/cos so 359 deg and 1 deg are close.
    rad = np.deg2rad(feats["Aspect"].astype(float))
    feats["aspect_sin"] = np.sin(rad)
    feats["aspect_cos"] = np.cos(rad)

    # -- Decode the GLIMS morphological class (digit 1 -> glacier type).
    feats = feats.join(decode_class(feats["Class"], logger))

    # -- Join and keep only glaciers observed in BOTH 1980 and 2010.
    n_before = wide.shape[0]
    mt = feats.join(wide[["area_1980", "area_2010", config.TARGET]],
                    how="inner").dropna(
        subset=["area_1980", "area_2010", config.TARGET])
    log_rows(logger, "Require both 1980 and 2010 areas (drop others)",
             n_before, len(mt))

    assert len(mt) == config.EXPECTED_MODEL_TABLE_ROWS, \
        (f"Pre-exclusion modeling table has {len(mt)} rows; expected "
         f"{config.EXPECTED_MODEL_TABLE_ROWS} from verified inspection")

    # -- Leakage guard: banned columns must not be in the feature list.
    banned_present = set(config.LEAKAGE_BANNED) & set(config.FEATURES)
    assert not banned_present, f"Leakage-banned features present: {banned_present}"
    logger.info(f"Leakage guard OK — features used by models: {config.FEATURES}"
                f" + one-hot({config.BASIN_COL})")

    desc = mt[config.TARGET].describe()
    logger.info("Target pct_change_1980_2010 descriptives (pre-exclusion): "
                + ", ".join(f"{k}={v:.2f}" for k, v in desc.items()))
    return mt


def quantify_outliers(mt: pd.DataFrame, logger) -> pd.DataFrame:
    """Quantify physically implausible positive-change glaciers.

    Area growth 1980->2010 contradicts the region-wide retreat signal and
    is attributed to mapping artifacts (snow cover in 1980 MSS imagery,
    digitization differences), expected mostly on very small glaciers.
    """
    pos = mt[mt[config.TARGET] > 0]
    logger.info(f"Glaciers with POSITIVE area change: {len(pos)} of {len(mt)} "
                f"({100 * len(pos) / len(mt):.1f}%)")

    bins = [0, 0.05, 0.1, 0.25, 0.5, 1.0, np.inf]
    labels = ["<0.05", "0.05-0.1", "0.1-0.25", "0.25-0.5", "0.5-1.0", ">1.0"]
    size_class = pd.cut(mt["area_1980"], bins=bins, labels=labels)
    tab = (pd.DataFrame({"size_class_km2_1980": size_class,
                         "positive_change": mt[config.TARGET] > 0,
                         "extreme_positive": mt[config.TARGET] > 20})
           .groupby("size_class_km2_1980", observed=True)
           .agg(n_glaciers=("positive_change", "size"),
                n_positive=("positive_change", "sum"),
                n_above_20pct=("extreme_positive", "sum"))
           .reset_index())
    tab["pct_positive"] = (100 * tab["n_positive"] / tab["n_glaciers"]).round(1)

    out = config.TABLES_DIR / "outlier_quantification.csv"
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out, index=False)
    logger.info(f"Outlier quantification by 1980 size class saved: {out}")
    logger.info("\n" + tab.to_string(index=False))
    return tab


def apply_outlier_rule(mt: pd.DataFrame, rule_name: str, logger) -> pd.DataFrame:
    """Apply one named exclusion rule from config.OUTLIER_RULES, logging
    the row counts each threshold removes (criterion 2.4 evidence)."""
    rule = config.OUTLIER_RULES[rule_name]
    out = mt
    logger.info(f"--- Applying outlier rule '{rule_name}': {rule}")
    if rule["min_area_1980"] is not None:
        n = len(out)
        out = out[out["area_1980"] >= rule["min_area_1980"]]
        log_rows(logger, f"  min 1980 area >= {rule['min_area_1980']} km2",
                 n, len(out))
    if rule["max_pct_change"] is not None:
        n = len(out)
        out = out[out[config.TARGET] <= rule["max_pct_change"]]
        log_rows(logger, f"  positive change <= +{rule['max_pct_change']}%",
                 n, len(out))
    logger.info(f"  Rule '{rule_name}' final: {len(out)} glaciers")
    return out


def main():
    logger = get_logger("data_prep")
    df = load_raw(logger)
    sanity_check(df, logger)
    mt = build_model_table(df, logger)
    quantify_outliers(mt, logger)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    row_counts = [{"step": "raw rows (glacier-decade)", "rows": len(df)},
                  {"step": "unique glaciers (GLIMS_ID)",
                   "rows": df["GLIMS_ID"].nunique()},
                  {"step": "modeling table pre-exclusion (1980 & 2010 areas)",
                   "rows": len(mt)}]

    for rule_name in config.OUTLIER_RULES:
        sub = apply_outlier_rule(mt, rule_name, logger)
        path = config.PROCESSED_DIR / f"model_table_{rule_name}.csv"
        sub.to_csv(path)
        logger.info(f"Saved {path} ({len(sub)} rows)")
        row_counts.append({"step": f"after outlier rule '{rule_name}'",
                           "rows": len(sub)})

    trail = pd.DataFrame(row_counts)
    trail_path = config.TABLES_DIR / "preprocessing_row_counts.csv"
    trail.to_csv(trail_path, index=False)
    logger.info(f"Row-count trail saved: {trail_path}\n" + trail.to_string(index=False))
    logger.info("DATA PREPARATION COMPLETE")


if __name__ == "__main__":
    main()
