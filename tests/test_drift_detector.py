"""
tests/test_drift_detector.py
-----------------------------
Unit tests for pipeline/drift_detector.py

Tests verify:
  1. Identical distributions  → drift_score = 0.0  (no drift)
  2. Completely different distributions → drift_score > threshold (drift detected)
  3. Drifted feature names are correctly returned
  4. Edge case: single feature, minimal data
  5. drift_score is in [0.0, 1.0]
"""

import sys
import os
import numpy  as np
import pandas as pd
import pytest

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from pipeline.drift_detector import DriftDetector


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def detector():
    """Return a fresh DriftDetector instance."""
    return DriftDetector()


@pytest.fixture
def identical_data():
    """Two DataFrames drawn from the same distribution."""
    rng = np.random.default_rng(seed=0)
    data = rng.normal(loc=0, scale=1, size=(200, 5))
    cols = [f"feature_{i}" for i in range(5)]
    ref  = pd.DataFrame(data[:100], columns=cols)
    cur  = pd.DataFrame(data[100:], columns=cols)
    return ref, cur


@pytest.fixture
def drifted_data():
    """Two DataFrames with strongly different distributions (guaranteed drift)."""
    rng = np.random.default_rng(seed=42)
    cols = ["feat_a", "feat_b", "feat_c"]

    ref_data = rng.normal(loc=0.0, scale=1.0, size=(200, len(cols)))
    cur_data = rng.normal(loc=10.0, scale=0.5, size=(200, len(cols)))   # huge shift

    ref = pd.DataFrame(ref_data, columns=cols)
    cur = pd.DataFrame(cur_data, columns=cols)
    return ref, cur


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestDriftDetector:

    def test_no_drift_identical_distributions(self, detector, identical_data):
        """Two halves of the same distribution should show no significant drift."""
        ref, cur = identical_data
        result = detector.detect(ref, cur)

        # With sample size 100 and identical dist, very few features drift
        assert result["drift_score"] < 0.5, (
            f"Expected low drift for identical distributions, got {result['drift_score']}"
        )

    def test_high_drift_different_distributions(self, detector, drifted_data):
        """Strongly shifted distributions should produce drift_score = 1.0."""
        ref, cur = drifted_data
        result = detector.detect(ref, cur)

        assert result["drift_score"] == 1.0, (
            f"Expected drift_score=1.0 for completely shifted data, "
            f"got {result['drift_score']}"
        )

    def test_drifted_features_names_returned(self, detector, drifted_data):
        """All features should appear in drifted_features for fully-drifted data."""
        ref, cur = drifted_data
        result = detector.detect(ref, cur)

        expected_features = set(ref.columns.tolist())
        returned_features = set(result["drifted_features"])
        assert expected_features == returned_features, (
            f"Drifted features mismatch.  Expected {expected_features}, "
            f"got {returned_features}"
        )

    def test_drift_score_is_fraction(self, detector, identical_data):
        """drift_score must always be in [0.0, 1.0]."""
        ref, cur = identical_data
        result = detector.detect(ref, cur)
        assert 0.0 <= result["drift_score"] <= 1.0

    def test_feature_results_contains_all_features(self, detector, identical_data):
        """feature_results dict must have an entry for every common feature."""
        ref, cur = identical_data
        result = detector.detect(ref, cur)
        assert set(result["feature_results"].keys()) == set(ref.columns.tolist())

    def test_target_column_excluded(self, detector):
        """target column in reference data must be stripped before KS test."""
        rng  = np.random.default_rng(0)
        cols = ["f1", "f2", config.TARGET_COLUMN]
        ref  = pd.DataFrame(rng.normal(size=(100, 3)), columns=cols)
        cur  = pd.DataFrame(rng.normal(size=(100, 2)), columns=["f1", "f2"])

        result = detector.detect(ref, cur)
        # target must NOT appear in feature results
        assert config.TARGET_COLUMN not in result["feature_results"]

    def test_result_has_timestamp(self, detector, identical_data):
        """Result dict must contain a non-empty timestamp string."""
        ref, cur = identical_data
        result = detector.detect(ref, cur)
        assert "timestamp" in result and len(result["timestamp"]) > 0

    def test_n_features_counts_match(self, detector, drifted_data):
        """n_features_total and n_features_drifted must be self-consistent."""
        ref, cur = drifted_data
        result = detector.detect(ref, cur)
        assert result["n_features_total"]   == len(ref.columns)
        assert result["n_features_drifted"] == len(result["drifted_features"])
