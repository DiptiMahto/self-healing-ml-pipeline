"""
api/app.py
----------
MODULE : REST API Layer
PURPOSE: Expose the Self-Healing ML Pipeline as a FastAPI web service.

ENDPOINTS:
  GET  /health          — Liveness check
  GET  /metrics         — Latest drift score, validation status, decision
  POST /predict         — Predict on a JSON payload of feature values
  POST /run-pipeline    — Trigger the full pipeline run (async background task)

RUN WITH:
    uvicorn api.app:app --reload --port 8000

THEN VISIT:
    http://localhost:8000/docs    (Swagger UI)
    http://localhost:8000/redoc   (ReDoc)
"""

import os
import sys
import json
import logging
import subprocess
from datetime import datetime
from typing   import Any

import numpy  as np
import pandas as pd
from fastapi           import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic          import BaseModel, Field

# ── Ensure project root is on the path ──────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from pipeline.data_loader     import DataLoader
from pipeline.data_validator  import DataValidator
from pipeline.drift_detector  import DriftDetector
from pipeline.model_predictor import ModelPredictor
from pipeline.decision_engine import DecisionEngine
from training.train_model      import ModelTrainer

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("api")

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Self-Healing ML Pipeline API",
    description = (
        "REST interface for the Self-Healing ML Pipeline.\n\n"
        "Supports real-time predictions, drift metrics, and pipeline triggers."
    ),
    version     = config.PIPELINE_VERSION,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# ── Lazy model store ──────────────────────────────────────────────────────────
_predictor: ModelPredictor | None = None


def get_predictor() -> ModelPredictor:
    """Load (and cache) the model on first call."""
    global _predictor
    if _predictor is None or not _predictor.model_loaded:
        _predictor = ModelPredictor()
        _predictor.load_model()
        logger.info("Model loaded into API predictor cache.")
    return _predictor


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """
    Input payload for the /predict endpoint.
    Send feature values as a dict where keys are feature names and
    values are floats. You may send one or more samples.

    Example (single sample):
        {"features": {"mean radius": 14.0, "mean texture": 19.4, ...}}

    Example (batch — use the /predict-batch endpoint instead):
        {"features": [{"mean radius": 14.0, ...}, {"mean radius": 12.5, ...}]}
    """
    features: dict[str, float] = Field(
        ...,
        example={
            "mean radius": 14.0,
            "mean texture": 19.4,
            "mean perimeter": 92.0,
            "mean area": 625.0,
        }
    )


class PredictResponse(BaseModel):
    prediction    : int
    label         : str
    probability_benign    : float
    probability_malignant : float
    avg_confidence: float
    timestamp     : str


class MetricsResponse(BaseModel):
    drift_score        : float
    n_features_drifted : int
    n_features_total   : int
    drifted_features   : list[str]
    validation_valid   : bool
    validation_issues  : list[str]
    decision_action    : str
    decision_reason    : str
    decision_rule_id   : int
    cooldown_active    : bool
    avg_confidence     : float | None
    timestamp          : str


class PipelineRunResponse(BaseModel):
    status       : str
    message      : str
    triggered_at : str


class HealthResponse(BaseModel):
    status       : str
    pipeline     : str
    version      : str
    model_exists : bool
    timestamp    : str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model = HealthResponse,
    summary        = "Health check",
    tags           = ["System"],
)
def health_check():
    """
    Liveness probe.  Returns HTTP 200 if the service is running.
    Also reports whether the trained model file exists on disk.
    """
    return HealthResponse(
        status       = "ok",
        pipeline     = config.PIPELINE_NAME,
        version      = config.PIPELINE_VERSION,
        model_exists = os.path.exists(config.MODEL_PATH),
        timestamp    = datetime.now().isoformat(),
    )


@app.get(
    "/metrics",
    response_model = MetricsResponse,
    summary        = "Latest pipeline metrics",
    tags           = ["Monitoring"],
)
def get_metrics():
    """
    Run a fresh analysis (load data → validate → detect drift → decide)
    and return the summary metrics.  Does NOT save predictions to disk.
    """
    try:
        loader      = DataLoader()
        incoming_df = loader.load_incoming()
        training_df = loader.load_training()

        validator  = DataValidator()
        validation = validator.validate(incoming_df)

        detector = DriftDetector()
        drift    = detector.detect(training_df, incoming_df)

        avg_confidence = None
        try:
            pred = get_predictor()
            _, probs = pred.predict(incoming_df)
            avg_confidence = pred.get_avg_confidence(probs)
        except Exception:
            pass

        engine   = DecisionEngine()
        decision = engine.decide(
            drift_score       = drift["drift_score"],
            validation_report = validation,
            avg_confidence    = avg_confidence,
        )

        return MetricsResponse(
            drift_score        = drift["drift_score"],
            n_features_drifted = drift["n_features_drifted"],
            n_features_total   = drift["n_features_total"],
            drifted_features   = drift["drifted_features"],
            validation_valid   = validation["is_valid"],
            validation_issues  = validation["issues"],
            decision_action    = decision["action"],
            decision_reason    = decision["reason"],
            decision_rule_id   = decision["rule_id"],
            cooldown_active    = decision.get("cooldown_active", False),
            avg_confidence     = avg_confidence,
            timestamp          = datetime.now().isoformat(),
        )

    except Exception as exc:
        logger.exception("Error in /metrics")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post(
    "/predict",
    response_model = PredictResponse,
    summary        = "Predict a single sample",
    tags           = ["Prediction"],
)
def predict(request: PredictRequest):
    """
    Make a prediction on a single feature vector.

    Pass all feature names that the model was trained on.
    Missing features are filled with 0 (with a warning logged).
    """
    try:
        pred = get_predictor()
    except FileNotFoundError:
        raise HTTPException(
            status_code = 503,
            detail      = "Model not found. Please train the model first.",
        )

    try:
        # Build a one-row DataFrame from the request features
        df = pd.DataFrame([request.features])

        # Align features to what the model expects
        expected = list(pred.model.feature_names_in_) \
            if hasattr(pred.model, "feature_names_in_") else list(df.columns)

        missing = [c for c in expected if c not in df.columns]
        if missing:
            logger.warning(f"/predict: filling missing features with 0: {missing}")
            for c in missing:
                df[c] = 0.0

        df = df[expected]

        predictions, probabilities = pred.predict(df)
        avg_conf   = pred.get_avg_confidence(probabilities)
        label      = "malignant" if int(predictions[0]) == 1 else "benign"

        return PredictResponse(
            prediction            = int(predictions[0]),
            label                 = label,
            probability_benign    = round(float(probabilities[0][0]), 4),
            probability_malignant = round(float(probabilities[0][1]), 4),
            avg_confidence        = avg_conf,
            timestamp             = datetime.now().isoformat(),
        )

    except Exception as exc:
        logger.exception("Error in /predict")
        raise HTTPException(status_code=500, detail=str(exc))


def _run_pipeline_task():
    """Background task: invoke main.py as a subprocess."""
    main_path = os.path.join(os.path.dirname(__file__), "..", "main.py")
    result = subprocess.run(
        [sys.executable, main_path],
        capture_output=True, text=True, timeout=180,
    )
    if result.returncode == 0:
        logger.info("/run-pipeline: Pipeline completed successfully.")
    else:
        logger.error(f"/run-pipeline: Pipeline failed:\n{result.stderr}")


@app.post(
    "/run-pipeline",
    response_model = PipelineRunResponse,
    summary        = "Trigger a full pipeline run",
    tags           = ["Pipeline"],
)
def trigger_pipeline(background_tasks: BackgroundTasks):
    """
    Kick off the full self-healing pipeline (data load → validate → drift
    detect → predict → decide → retrain if needed) in a background task.

    Returns immediately with a 202-style response while the pipeline runs.
    Check `/metrics` after a few seconds to see updated results.
    """
    background_tasks.add_task(_run_pipeline_task)
    return PipelineRunResponse(
        status       = "accepted",
        message      = "Pipeline run triggered in background. "
                       "Check /metrics in a few seconds for updated results.",
        triggered_at = datetime.now().isoformat(),
    )


@app.post(
    "/retrain",
    summary = "Force model retraining",
    tags    = ["Pipeline"],
)
def force_retrain():
    """
    Immediately retrain the model on the current training data.
    This endpoint is synchronous — it waits until training is complete.
    """
    try:
        trainer = ModelTrainer()
        meta    = trainer.train()
        # Invalidate the cached predictor so next /predict loads the new model
        global _predictor
        _predictor = None
        return {
            "status"    : "ok",
            "message"   : "Model retrained successfully.",
            "accuracy"  : meta["accuracy"],
            "f1_weighted": meta["f1_weighted"],
            "timestamp" : datetime.now().isoformat(),
        }
    except Exception as exc:
        logger.exception("Error in /retrain")
        raise HTTPException(status_code=500, detail=str(exc))
