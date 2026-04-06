"""
tests/test_data_validator.py
-----------------------------
Unit tests for pipeline/data_validator.py

Tests verify:
  1. Valid DataFrame   → is_valid=True, issues=[]
  2. Empty DataFrame   → is_valid=False, critical issue reported
  3. High missing fraction per column → is_valid=False
  4. Infinite values   → is_valid=False
  5. Negative values   → is_valid=False
  6. stats dict always present with expected keys
  7. Multiple issues accumulated correctly
"""

import sys
import os
import numpy  as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from pipeline.data_validator import DataValidator


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def validator():
    """Return a fresh DataValidator instance."""
    return DataValidator()


@pytest.fixture
def clean_df():
    """A perfectly clean DataFrame — all checks should pass."""
    rng  = np.random.default_rng(seed=0)
    data = rng.uniform(low=0.1, high=10.0, size=(100, 5))
    return pd.DataFrame(data, columns=[f"col_{i}" for i in range(5)])


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestDataValidator:

    def test_valid_dataframe_passes(self, validator, clean_df):
        """A clean DataFrame must pass all checks with no issues."""
        report = validator.validate(clean_df)
        assert report["is_valid"] is True
        assert report["issues"] == []

    def test_empty_dataframe_fails(self, validator):
        """An empty DataFrame must immediately fail with a CRITICAL issue."""
        empty_df = pd.DataFrame()
        report   = validator.validate(empty_df)
        assert report["is_valid"] is False
        assert len(report["issues"]) >= 1
        assert any("empty" in issue.lower() or "critical" in issue.lower()
                   for issue in report["issues"])

    def test_high_missing_values_fails(self, validator, clean_df):
        """Column with > MAX_MISSING_FRACTION missing values must trigger issue."""
        df = clean_df.copy()
        # Set 90% of col_0 to NaN — well above threshold
        n_missing = int(0.90 * len(df))
        df.iloc[:n_missing, 0] = np.nan

        report = validator.validate(df)
        assert report["is_valid"] is False
        assert any("missing" in issue.lower() for issue in report["issues"])

    def test_exactly_below_missing_threshold_passes(self, validator, clean_df):
        """Column with < MAX_MISSING_FRACTION missing values must still pass."""
        df = clean_df.copy()
        # Set exactly 10% to NaN — below the 20% threshold
        n_missing = int(0.10 * len(df))
        df.iloc[:n_missing, 0] = np.nan

        report = validator.validate(df)
        assert report["is_valid"] is True

    def test_infinite_values_fails(self, validator, clean_df):
        """DataFrame containing np.inf must be flagged as invalid."""
        df = clean_df.copy()
        df.iloc[0, 0] = np.inf

        report = validator.validate(df)
        assert report["is_valid"] is False
        assert any("inf" in issue.lower() or "infinite" in issue.lower()
                   for issue in report["issues"])

    def test_negative_values_fails(self, validator, clean_df):
        """DataFrame containing negative values must be flagged."""
        df = clean_df.copy()
        df.iloc[0, 1] = -999.0

        report = validator.validate(df)
        assert report["is_valid"] is False
        assert any("negative" in issue.lower() for issue in report["issues"])

    def test_stats_always_present(self, validator, clean_df):
        """stats dict must always be returned with core keys."""
        report   = validator.validate(clean_df)
        required = {"n_rows", "n_columns", "total_missing", "missing_pct"}
        for key in required:
            assert key in report["stats"], f"Missing stats key: {key}"

    def test_multiple_issues_accumulated(self, validator, clean_df):
        """Both inf and negative issues should be collected together."""
        df = clean_df.copy()
        df.iloc[0, 0] = np.inf             # triggers infinite-values check
        df.iloc[5:55, 1] = -1.0            # 50 negative values — guaranteed trigger

        report = validator.validate(df)
        assert report["is_valid"] is False
        # Should report at least 2 distinct issues
        assert len(report["issues"]) >= 2, (
            f"Expected >= 2 issues but got {len(report['issues'])}: {report['issues']}"
        )

    def test_report_structure_always_returned(self, validator, clean_df):
        """report must always have is_valid, issues, and stats keys."""
        report = validator.validate(clean_df)
        assert "is_valid" in report
        assert "issues"   in report
        assert "stats"    in report
        assert isinstance(report["is_valid"], bool)
        assert isinstance(report["issues"],   list)
        assert isinstance(report["stats"],    dict)
