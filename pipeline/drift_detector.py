"""
pipeline/drift_detector.py
--------------------------
MODULE : Drift Detector
PURPOSE: Compares the statistical distribution of each feature between
         the reference (training) dataset and the current (incoming) dataset
         using the Kolmogorov-Smirnov (KS) two-sample test.

HOW DRIFT IS SCORED:
    - For each numeric feature, run a KS test between training and incoming.
    - If p-value < KS_P_VALUE_THRESHOLD  → feature is considered "drifted".
    - drift_score = drifted_feature_count / total_feature_count
    - drift_score ∈ [0.0, 1.0]; higher = more drift.

CONNECTS TO PIPELINE:
    main.py calls DriftDetector().detect(training_df, incoming_df)
    The returned drift_score is consumed by the DecisionEngine.
"""

import os
import csv
import logging
from datetime import datetime

import numpy  as np
import pandas as pd
from scipy    import stats as scipy_stats

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

logger = logging.getLogger(__name__)


class DriftDetector:
    """Statistical drift detection using the KS test per feature."""

    def detect(self, reference_df: pd.DataFrame,
                      current_df:   pd.DataFrame) -> dict:
        """
        Compare feature distributions between reference and current data.

        Parameters
        ----------
        reference_df : training data (with or without target column)
        current_df   : incoming batch (no target column expected)

        Returns
        -------
        dict with keys:
            drift_score       : float  — fraction of drifted features
            drifted_features  : list   — names of features that drifted
            feature_results   : dict   — per-feature {ks_stat, p_value, drifted}
            timestamp         : str
        """
        # Align columns: use only numeric features present in both DataFrames
        ref = reference_df.select_dtypes(include=[np.number]).copy()
        cur = current_df.select_dtypes(include=[np.number]).copy()

        # Remove target from reference if present
        for col in [config.TARGET_COLUMN, "target"]:
            if col in ref.columns:
                ref = ref.drop(columns=[col])

        common_cols = [c for c in ref.columns if c in cur.columns]
        ref = ref[common_cols]
        cur = cur[common_cols]

        feature_results  = {}
        drifted_features = []

        for col in common_cols:
            ref_vals = ref[col].dropna().values
            cur_vals = cur[col].dropna().values

            if len(ref_vals) < 5 or len(cur_vals) < 5:
                # Not enough data for a meaningful test
                feature_results[col] = {
                    "ks_stat": None, "p_value": None, "drifted": False
                }
                continue

            ks_stat, p_value = scipy_stats.ks_2samp(ref_vals, cur_vals)
            drifted          = bool(p_value < config.KS_P_VALUE_THRESHOLD)

            feature_results[col] = {
                "ks_stat": round(float(ks_stat), 4),
                "p_value": round(float(p_value), 4),
                "drifted": drifted,
            }

            if drifted:
                drifted_features.append(col)

        n_total      = len(common_cols)
        n_drifted    = len(drifted_features)
        drift_score  = round(n_drifted / n_total, 4) if n_total > 0 else 0.0

        logger.info(f"DriftDetector: {n_drifted}/{n_total} features drifted  "
                    f"(score={drift_score:.4f}, threshold={config.DRIFT_THRESHOLD})")

        return {
            "drift_score":      drift_score,
            "drifted_features": drifted_features,
            "feature_results":  feature_results,
            "n_features_total": n_total,
            "n_features_drifted": n_drifted,
            "timestamp":        datetime.now().isoformat(),
        }

    def save_drift_log(self, drift_result: dict):
        """Append this run's drift result to the drift CSV log."""
        os.makedirs(config.LOGS_DIR, exist_ok=True)
        file_exists = os.path.exists(config.DRIFT_LOG)

        with open(config.DRIFT_LOG, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "timestamp", "drift_score",
                    "n_drifted", "n_total", "drifted_features"
                ])
            writer.writerow([
                drift_result["timestamp"],
                drift_result["drift_score"],
                drift_result["n_features_drifted"],
                drift_result["n_features_total"],
                "|".join(drift_result["drifted_features"]),
            ])
        logger.debug(f"Drift log updated: {config.DRIFT_LOG}")
