"""
pipeline/decision_engine.py
---------------------------
MODULE : Decision Engine
PURPOSE: The "brain" of the self-healing pipeline.
         Examines drift score, validation report, and model confidence,
         then applies RULE-BASED logic to decide whether to:
             "continue"  — current model is good enough
             "retrain"   — retraining is required

RULE TABLE (evaluated in priority order):
  Rule 1 | Data invalid AND drift high     → RETRAIN  (critical failure)
  Rule 2 | Drift score > threshold         → RETRAIN  (distribution shift)
  Rule 3 | Data invalid (drift OK)         → RETRAIN  (data quality concern)
  Rule 4 | Low avg model confidence        → RETRAIN  (model is uncertain)
  Rule 5 | All checks pass                 → CONTINUE

ADVANCED FEATURES:
  • Cooldown guard    — suppresses retraining if last retrain was < RETRAIN_COOLDOWN_MINUTES ago
  • Confidence rule   — triggers retrain when avg prediction confidence is low

CONNECTS TO PIPELINE:
    main.py → engine.decide(drift_score, validation_report, predictions, avg_confidence)
    If decision == "retrain", main.py triggers ModelTrainer().train()
"""

import json
import logging
import os
from datetime import datetime, timedelta

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

logger = logging.getLogger(__name__)


class DecisionEngine:
    """
    Rule-based engine that decides whether retraining is required.

    Advanced behaviour added over base version:
    1. Confidence-based trigger (Rule 4)
    2. Cooldown guard — prevents retraining more than once per N minutes
    """

    # ── Cooldown helpers ──────────────────────────────────────────────────────

    def _is_in_cooldown(self) -> bool:
        """
        Return True if a retrain happened within the cooldown window.
        Reads the timestamp stored in RETRAIN_COOLDOWN_FILE (JSON).
        """
        cooldown_file = getattr(
            config, "RETRAIN_COOLDOWN_FILE",
            os.path.join(config.LOGS_DIR, "retrain_cooldown.json")
        )
        if not os.path.exists(cooldown_file):
            return False  # No record → no cooldown active

        try:
            with open(cooldown_file, "r") as f:
                data = json.load(f)
            last_retrain   = datetime.fromisoformat(data["last_retrain"])
            cooldown_mins  = getattr(config, "RETRAIN_COOLDOWN_MINUTES", 10)
            elapsed        = datetime.now() - last_retrain
            if elapsed < timedelta(minutes=cooldown_mins):
                remaining = int((timedelta(minutes=cooldown_mins) - elapsed).seconds / 60)
                logger.info(
                    f"DecisionEngine: 🔒 Cooldown active — last retrain at "
                    f"{last_retrain.strftime('%H:%M:%S')} "
                    f"(cooldown={cooldown_mins} min, ~{remaining} min remaining)."
                )
                return True
        except Exception as e:
            logger.warning(f"DecisionEngine: Could not read cooldown file: {e}")

        return False

    def _record_retrain_timestamp(self):
        """Save the current time to the cooldown file after a retrain decision."""
        cooldown_file = getattr(
            config, "RETRAIN_COOLDOWN_FILE",
            os.path.join(config.LOGS_DIR, "retrain_cooldown.json")
        )
        os.makedirs(config.LOGS_DIR, exist_ok=True)
        with open(cooldown_file, "w") as f:
            json.dump({"last_retrain": datetime.now().isoformat()}, f, indent=2)
        logger.debug(f"DecisionEngine: Cooldown timestamp saved → {cooldown_file}")

    # ── Rule builder ──────────────────────────────────────────────────────────

    def _build_rules(
        self,
        drift_score       : float,
        validation_report : dict,
        avg_confidence    : float | None,
    ) -> list:
        """
        Construct the ordered list of (condition, action, reason) tuples.
        Evaluated top-to-bottom; first True condition wins.
        """
        data_invalid   = not validation_report.get("is_valid", True)
        drift_high     = drift_score > config.DRIFT_THRESHOLD
        conf_threshold = getattr(config, "CONFIDENCE_THRESHOLD", 0.50)

        # Guard: only apply confidence rule when we have actual probability values
        low_confidence = (
            avg_confidence is not None and avg_confidence < conf_threshold
        )

        rules = [
            # ── Rule 1 — Both drift AND validation failed (worst case) ──────
            (
                data_invalid and drift_high,
                "retrain",
                (
                    f"CRITICAL: Data validation failed AND drift score {drift_score:.4f} "
                    f"exceeds threshold {config.DRIFT_THRESHOLD}. "
                    "Full retraining required immediately."
                ),
            ),
            # ── Rule 2 — Drift alone exceeds threshold ────────────────────
            (
                drift_high,
                "retrain",
                (
                    f"Drift score {drift_score:.4f} exceeds threshold "
                    f"{config.DRIFT_THRESHOLD}. "
                    "Retraining to adapt model to new feature distribution."
                ),
            ),
            # ── Rule 3 — Data invalid but drift is acceptable ─────────────
            (
                data_invalid,
                "retrain",
                (
                    f"Data validation issues detected: "
                    f"{validation_report.get('issues', [])}. "
                    "Retraining on clean data is advised."
                ),
            ),
            # ── Rule 4 — Low model confidence (self-healing intelligence) ──
            (
                low_confidence,
                "retrain",
                (
                    f"Average model confidence {avg_confidence:.3f} is below "
                    f"threshold {conf_threshold:.2f}. "
                    "Model predictions are uncertain — retraining advised."
                ) if avg_confidence is not None else (
                    "Model confidence data unavailable — retraining advised."
                ),
            ),
            # ── Rule 5 — Everything looks fine (catch-all) ───────────────
            (
                True,
                "continue",
                (
                    f"Drift score {drift_score:.4f} is within threshold "
                    f"{config.DRIFT_THRESHOLD}. Data quality is acceptable. "
                    "Predictions accepted — no retraining needed."
                ),
            ),
        ]
        return rules

    # ── Public interface ──────────────────────────────────────────────────────

    def decide(
        self,
        drift_score       : float,
        validation_report : dict,
        predictions       = None,
        avg_confidence    : float | None = None,
    ) -> dict:
        """
        Evaluate rules and return a structured decision dictionary.

        Parameters
        ----------
        drift_score        : float — fraction of drifted features (0–1)
        validation_report  : dict  — output of DataValidator.validate()
        predictions        : np.ndarray | None — predicted class labels
        avg_confidence     : float | None — mean of max(P(class)) per sample

        Returns
        -------
        dict with keys:
            action          : "continue" | "retrain"
            reason          : human-readable explanation
            rule_id         : 1-based index of the rule that fired
            cooldown_active : bool — True if retrain was suppressed by cooldown
            avg_confidence  : float | None — echo of input for logging
        """
        rules = self._build_rules(drift_score, validation_report, avg_confidence)

        for rule_id, (condition, action, reason) in enumerate(rules, start=1):
            if not condition:
                continue  # This rule does not apply

            cooldown_active = False

            # Cooldown guard — suppress retrain if we just retrained recently
            if action == "retrain" and self._is_in_cooldown():
                action          = "continue"
                reason          = f"[COOLDOWN ACTIVE] Retrain suppressed. Original reason: {reason}"
                cooldown_active = True

            # Record the timestamp if we are actually going ahead with retraining
            if action == "retrain":
                self._record_retrain_timestamp()

            emoji = "🔄" if action == "retrain" else "✅"
            logger.info(
                f"DecisionEngine: Rule #{rule_id} fired → {action.upper()} {emoji}"
            )
            logger.info(f"DecisionEngine: {reason}")
            if cooldown_active:
                logger.warning("DecisionEngine: ⏳ Retrain was suppressed by cooldown.")

            return {
                "action":          action,
                "reason":          reason,
                "rule_id":         rule_id,
                "cooldown_active": cooldown_active,
                "avg_confidence":  avg_confidence,
            }

        # Should never reach here given the catch-all rule, but just in case:
        logger.error("DecisionEngine: No rule matched — this should not happen!")
        return {
            "action":          "continue",
            "reason":          "No rule matched — defaulting to continue.",
            "rule_id":         -1,
            "cooldown_active": False,
            "avg_confidence":  avg_confidence,
        }
