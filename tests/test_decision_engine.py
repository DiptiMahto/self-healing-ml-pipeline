"""
tests/test_decision_engine.py
------------------------------
Unit tests for pipeline/decision_engine.py

Tests verify all 5 decision rules fire correctly:
  Rule 1 — data_invalid AND drift_high → retrain
  Rule 2 — drift_high only             → retrain
  Rule 3 — data_invalid only           → retrain
  Rule 4 — low avg confidence          → retrain
  Rule 5 — all clear                   → continue

Also tests:
  - cooldown suppresses a retrain decision
  - returned dict has the correct structure
  - avg_confidence echo in result
"""

import sys
import os
import json
import shutil
import tempfile
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from pipeline.decision_engine import DecisionEngine


# ── Helpers ───────────────────────────────────────────────────────────────────

def _valid_report() -> dict:
    return {"is_valid": True, "issues": [], "stats": {}}


def _invalid_report() -> dict:
    return {"is_valid": False, "issues": ["High missing values"], "stats": {}}


ABOVE_THRESHOLD = config.DRIFT_THRESHOLD + 0.10   # e.g. 0.40
BELOW_THRESHOLD = config.DRIFT_THRESHOLD - 0.10   # e.g. 0.20

HIGH_CONFIDENCE = getattr(config, "CONFIDENCE_THRESHOLD", 0.50) + 0.20   # 0.70
LOW_CONFIDENCE  = getattr(config, "CONFIDENCE_THRESHOLD", 0.50) - 0.05   # 0.45


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def engine():
    """Return a DecisionEngine with cooldown disabled."""
    e = DecisionEngine()
    # Patch cooldown so it never blocks during tests
    e._is_in_cooldown = lambda: False
    e._record_retrain_timestamp = lambda: None
    return e


@pytest.fixture
def engine_with_cooldown(tmp_path):
    """Return DecisionEngine whose cooldown is pre-set to 'just now'."""
    cooldown_file = tmp_path / "retrain_cooldown.json"
    cooldown_file.write_text(
        json.dumps({"last_retrain": datetime.now().isoformat()})
    )
    e = DecisionEngine()
    # Override cooldown path to use the temp file
    orig_file = getattr(config, "RETRAIN_COOLDOWN_FILE", None)
    config.RETRAIN_COOLDOWN_FILE = str(cooldown_file)
    yield e
    # Restore original path
    if orig_file is not None:
        config.RETRAIN_COOLDOWN_FILE = orig_file


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestDecisionEngine:

    # ── Rule 1 ─────────────────────────────────────────────────────────────

    def test_rule1_invalid_data_and_high_drift_retrains(self, engine):
        """Rule 1: Both validation fail AND drift high → RETRAIN."""
        decision = engine.decide(
            drift_score       = ABOVE_THRESHOLD,
            validation_report = _invalid_report(),
            avg_confidence    = HIGH_CONFIDENCE,
        )
        assert decision["action"]  == "retrain"
        assert decision["rule_id"] == 1

    # ── Rule 2 ─────────────────────────────────────────────────────────────

    def test_rule2_high_drift_alone_retrains(self, engine):
        """Rule 2: High drift only (validation OK) → RETRAIN."""
        decision = engine.decide(
            drift_score       = ABOVE_THRESHOLD,
            validation_report = _valid_report(),
            avg_confidence    = HIGH_CONFIDENCE,
        )
        assert decision["action"]  == "retrain"
        assert decision["rule_id"] == 2

    # ── Rule 3 ─────────────────────────────────────────────────────────────

    def test_rule3_invalid_data_low_drift_retrains(self, engine):
        """Rule 3: Validation fails, drift OK → RETRAIN."""
        decision = engine.decide(
            drift_score       = BELOW_THRESHOLD,
            validation_report = _invalid_report(),
            avg_confidence    = HIGH_CONFIDENCE,
        )
        assert decision["action"]  == "retrain"
        assert decision["rule_id"] == 3

    # ── Rule 4 ─────────────────────────────────────────────────────────────

    def test_rule4_low_confidence_retrains(self, engine):
        """Rule 4: Low avg model confidence → RETRAIN."""
        decision = engine.decide(
            drift_score       = BELOW_THRESHOLD,
            validation_report = _valid_report(),
            avg_confidence    = LOW_CONFIDENCE,
        )
        assert decision["action"]  == "retrain"
        assert decision["rule_id"] == 4

    # ── Rule 5 ─────────────────────────────────────────────────────────────

    def test_rule5_all_clear_continues(self, engine):
        """Rule 5: Everything fine → CONTINUE."""
        decision = engine.decide(
            drift_score       = BELOW_THRESHOLD,
            validation_report = _valid_report(),
            avg_confidence    = HIGH_CONFIDENCE,
        )
        assert decision["action"]  == "continue"
        assert decision["rule_id"] == 5

    def test_none_confidence_skips_rule4(self, engine):
        """When avg_confidence=None the confidence rule must NOT fire."""
        decision = engine.decide(
            drift_score       = BELOW_THRESHOLD,
            validation_report = _valid_report(),
            avg_confidence    = None,
        )
        # Should reach rule 5 (continue), not rule 4
        assert decision["action"]  == "continue"
        assert decision["rule_id"] == 5

    # ── Cooldown ───────────────────────────────────────────────────────────

    def test_cooldown_suppresses_retrain(self, engine_with_cooldown):
        """When cooldown is active a retrain decision is converted to continue."""
        decision = engine_with_cooldown.decide(
            drift_score       = ABOVE_THRESHOLD,
            validation_report = _valid_report(),
            avg_confidence    = HIGH_CONFIDENCE,
        )
        assert decision["action"]          == "continue"
        assert decision["cooldown_active"] is True

    # ── Return structure ───────────────────────────────────────────────────

    def test_result_has_required_keys(self, engine):
        """The returned dict must always contain the mandatory keys."""
        decision = engine.decide(
            drift_score       = BELOW_THRESHOLD,
            validation_report = _valid_report(),
        )
        for key in ["action", "reason", "rule_id", "cooldown_active", "avg_confidence"]:
            assert key in decision, f"Missing key in decision: {key}"

    def test_avg_confidence_echoed_in_result(self, engine):
        """avg_confidence input must be echoed back in the result dict."""
        decision = engine.decide(
            drift_score       = BELOW_THRESHOLD,
            validation_report = _valid_report(),
            avg_confidence    = 0.88,
        )
        assert decision["avg_confidence"] == pytest.approx(0.88)

    def test_action_is_valid_string(self, engine):
        """action must always be 'retrain' or 'continue'."""
        for drift in [BELOW_THRESHOLD, ABOVE_THRESHOLD]:
            decision = engine.decide(
                drift_score       = drift,
                validation_report = _valid_report(),
                avg_confidence    = HIGH_CONFIDENCE,
            )
            assert decision["action"] in ("retrain", "continue")
