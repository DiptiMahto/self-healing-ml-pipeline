"""
main.py
-------
Pipeline Orchestrator for the Self-Healing ML Pipeline.

This is the ENTRY POINT. Run with:
    python main.py

Flow:
  1. Load incoming data
  2. Validate incoming data
  3. Detect drift vs training data
  4. Load trained model
  5. Make predictions
  6. Pass results to Decision Engine
  7. Decision Engine decides: continue OR retrain
  8. Log everything
"""

import os
import sys
import logging
from datetime import datetime

# ── ensure project root is on the path ──────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from pipeline.data_loader    import DataLoader
from pipeline.data_validator import DataValidator
from pipeline.drift_detector import DriftDetector
from pipeline.model_predictor import ModelPredictor
from pipeline.decision_engine import DecisionEngine
from training.train_model     import ModelTrainer

# ── logging setup ────────────────────────────────────────────────────────────
os.makedirs(config.LOGS_DIR,   exist_ok=True)
os.makedirs(config.MODELS_DIR, exist_ok=True)
os.makedirs(config.DATA_DIR,   exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(config.PIPELINE_LOG, encoding="utf-8"),
        logging.StreamHandler(open(sys.stdout.fileno(), mode='w', encoding='utf-8', closefd=False)),
    ],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline():
    """Execute the full self-healing ML pipeline."""

    logger.info("=" * 60)
    logger.info(f"  {config.PIPELINE_NAME}  |  {config.PIPELINE_VERSION}")
    logger.info(f"  Run started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # ── STEP 0 : Ensure model exists (train if not) ──────────────────────────
    if not os.path.exists(config.MODEL_PATH):
        logger.warning("No trained model found. Running initial training …")
        trainer = ModelTrainer()
        trainer.train()

    # ── STEP 1 : Load incoming data ──────────────────────────────────────────
    logger.info("[STEP 1] Loading incoming data …")
    loader = DataLoader()
    incoming_df = loader.load_incoming()
    training_df = loader.load_training()
    logger.info(f"         Incoming shape : {incoming_df.shape}")
    logger.info(f"         Training shape : {training_df.shape}")

    # ── STEP 2 : Validate incoming data ──────────────────────────────────────
    logger.info("[STEP 2] Validating incoming data …")
    validator = DataValidator()
    validation_report = validator.validate(incoming_df)
    if not validation_report["is_valid"]:
        logger.error("Data validation FAILED. Check validation_report in logs.")
        logger.error(f"Issues: {validation_report['issues']}")
        # We still continue — real pipelines often soft-fail here
    else:
        logger.info("         Validation PASSED ✓")

    # ── STEP 3 : Detect drift ─────────────────────────────────────────────────
    logger.info("[STEP 3] Running drift detection …")
    detector  = DriftDetector()
    drift_result = detector.detect(
        reference_df=training_df,
        current_df=incoming_df,
    )
    drift_score   = drift_result["drift_score"]
    drifted_feats = drift_result["drifted_features"]
    logger.info(f"         Drift score     : {drift_score:.4f}")
    logger.info(f"         Drifted features: {drifted_feats}")
    detector.save_drift_log(drift_result)

    # ── STEP 4 : Load model & predict ────────────────────────────────────────
    logger.info("[STEP 4] Loading model and generating predictions …")
    predictor = ModelPredictor()
    predictor.load_model()
    predictions, probabilities = predictor.predict(incoming_df)
    avg_confidence = predictor.get_avg_confidence(probabilities)   # NEW
    predictor.save_predictions(predictions, probabilities)
    logger.info(f"         Predictions generated for {len(predictions)} samples.")
    logger.info(f"         Avg model confidence : {avg_confidence:.4f}")

    # ── STEP 5 : Decision Engine ──────────────────────────────────────────────
    logger.info("[STEP 5] Running Decision Engine …")
    engine   = DecisionEngine()
    decision = engine.decide(
        drift_score       = drift_score,
        validation_report = validation_report,
        predictions       = predictions,
        avg_confidence    = avg_confidence,   # NEW
    )
    logger.info(f"         Decision : {decision['action'].upper()}")
    logger.info(f"         Reason   : {decision['reason']}")

    # ── STEP 6 : Act on decision ──────────────────────────────────────────────
    if decision["action"] == "retrain":
        logger.info("[STEP 6] Retraining triggered …")
        trainer = ModelTrainer()
        trainer.train()
        logger.info("         Model retrained and saved ✓")
    else:
        logger.info("[STEP 6] No retraining needed. Predictions accepted ✓")

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("-" * 60)
    logger.info("  PIPELINE SUMMARY")
    logger.info(f"  Validation  : {'PASS' if validation_report['is_valid'] else 'FAIL'}")
    logger.info(f"  Drift Score : {drift_score:.4f}  (threshold={config.DRIFT_THRESHOLD})")
    logger.info(f"  Decision    : {decision['action'].upper()}")
    logger.info(f"  Samples     : {len(predictions)}")
    logger.info("-" * 60)
    logger.info("  Pipeline run COMPLETE.")
    logger.info("=" * 60)

    return {
        "validation": validation_report,
        "drift":      drift_result,
        "decision":   decision,
        "n_samples":  len(predictions),
    }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_pipeline()
