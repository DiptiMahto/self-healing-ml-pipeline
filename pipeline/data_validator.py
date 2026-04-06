"""
pipeline/data_validator.py
--------------------------
MODULE : Data Validator
PURPOSE: Checks the incoming DataFrame for common data quality problems
         BEFORE it is used for predictions or drift analysis.

CHECKS PERFORMED:
  1. Empty DataFrame guard
  2. Missing-value fraction per column
  3. Infinite / NaN value detection
  4. Negative values in columns that should be non-negative (all BC features)

CONNECTS TO PIPELINE:
    main.py calls DataValidator().validate(incoming_df)
    The returned validation_report dict is consumed by the DecisionEngine.
"""

import logging
import numpy  as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates an incoming DataFrame against basic data-quality rules."""

    def validate(self, df: pd.DataFrame) -> dict:
        """
        Run all validation checks.

        Returns
        -------
        dict with keys:
            is_valid  : bool   — True only if ALL checks pass
            issues    : list   — human-readable list of problems found
            stats     : dict   — summary statistics for the dashboard
        """
        issues = []
        stats  = {}

        # ── Check 1: empty DataFrame ─────────────────────────────────────────
        if df.empty:
            issues.append("CRITICAL: Incoming DataFrame is empty.")
            return {"is_valid": False, "issues": issues, "stats": stats}

        stats["n_rows"]    = int(df.shape[0])
        stats["n_columns"] = int(df.shape[1])

        # ── Check 2: missing values ───────────────────────────────────────────
        missing_fractions = df.isnull().mean()
        bad_cols = missing_fractions[
            missing_fractions > config.MAX_MISSING_FRACTION
        ].index.tolist()
        if bad_cols:
            issues.append(
                f"High missing-value fraction (>{config.MAX_MISSING_FRACTION*100:.0f}%) "
                f"in columns: {bad_cols}"
            )
        stats["total_missing"] = int(df.isnull().sum().sum())
        stats["missing_pct"]   = round(float(df.isnull().mean().mean()) * 100, 2)

        # ── Check 3: infinite values ──────────────────────────────────────────
        numeric_df   = df.select_dtypes(include=[np.number])
        n_inf        = int(np.isinf(numeric_df.values).sum())
        stats["n_inf"] = n_inf
        if n_inf > 0:
            issues.append(f"Infinite values found: {n_inf} cells affected.")

        # ── Check 4: negative values (BC features are all non-negative) ───────
        n_negative = int((numeric_df.values < 0).sum())
        stats["n_negative"] = n_negative
        if n_negative > 0:
            issues.append(
                f"Negative values found in {n_negative} cells "
                f"(all Breast Cancer features should be ≥ 0)."
            )

        is_valid = len(issues) == 0

        if is_valid:
            logger.info("DataValidator: All checks passed ✓")
        else:
            for issue in issues:
                logger.warning(f"DataValidator: {issue}")

        return {
            "is_valid": is_valid,
            "issues":   issues,
            "stats":    stats,
        }
