"""
pipeline/model_predictor.py
---------------------------
MODULE : Model Predictor
PURPOSE: Loads the saved model from disk and generates predictions
         (class labels + probability scores) for the incoming batch.

CONNECTS TO PIPELINE:
    main.py calls:
        predictor = ModelPredictor()
        predictor.load_model()
        predictions, probabilities = predictor.predict(incoming_df)
"""

import os
import csv
import logging
from datetime import datetime

import numpy  as np
import pandas as pd
import joblib

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

logger = logging.getLogger(__name__)


class ModelPredictor:
    """Wraps the saved sklearn model for inference."""

    def __init__(self):
        self.model        = None
        self.model_loaded = False

    # ── public interface ──────────────────────────────────────────────────────

    def load_model(self):
        """Load model from disk. Raises FileNotFoundError if absent."""
        if not os.path.exists(config.MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {config.MODEL_PATH}. "
                "Run training/train_model.py first, or let main.py auto-train."
            )
        self.model        = joblib.load(config.MODEL_PATH)
        self.model_loaded = True
        logger.info(f"Model loaded from {config.MODEL_PATH}  "
                    f"(type={type(self.model).__name__})")

    def predict(self, df: pd.DataFrame):
        """
        Run inference on df.

        Parameters
        ----------
        df : incoming DataFrame (no target column)

        Returns
        -------
        predictions  : np.ndarray of class labels  (0 or 1)
        probabilities: np.ndarray of shape (n, 2)  — P(class=0), P(class=1)
        """
        if not self.model_loaded:
            raise RuntimeError("Call load_model() before predict().")

        # Keep only numeric columns the model was trained on
        X = self._align_features(df)

        predictions   = self.model.predict(X)
        probabilities = self.model.predict_proba(X)

        logger.info(f"Predictions: {len(predictions)} samples  |  "
                    f"class-1 rate={predictions.mean():.2%}")
        return predictions, probabilities

    def save_predictions(self, predictions: np.ndarray,
                                probabilities: np.ndarray):
        """Persist predictions to a CSV log for dashboard retrieval."""
        os.makedirs(config.LOGS_DIR, exist_ok=True)
        file_exists = os.path.exists(config.PREDICTIONS_LOG)

        ts = datetime.now().isoformat()
        with open(config.PREDICTIONS_LOG, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "timestamp", "sample_index",
                    "prediction", "prob_class0", "prob_class1"
                ])
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                writer.writerow([
                    ts, i, int(pred),
                    round(float(prob[0]), 4),
                    round(float(prob[1]), 4),
                ])
        logger.debug(f"Predictions saved to {config.PREDICTIONS_LOG}")

    def get_avg_confidence(self, probabilities: np.ndarray) -> float:
        """
        Compute average model confidence across all predictions.

        Confidence for each sample = max(P(class=0), P(class=1)).
        Overall confidence = mean of per-sample confidences.

        A value close to 1.0 means the model is very certain.
        A value close to 0.5 means the model is barely better than a coin flip,
        which can indicate the data has drifted far from the training distribution.

        Parameters
        ----------
        probabilities : np.ndarray of shape (n_samples, 2)

        Returns
        -------
        float — average confidence score in [0.5, 1.0]
        """
        if probabilities is None or len(probabilities) == 0:
            return 1.0   # Default: assume confident when no data available

        avg_conf = float(np.max(probabilities, axis=1).mean())
        logger.debug(f"ModelPredictor: Avg confidence = {avg_conf:.4f}")
        return round(avg_conf, 4)



    # ── private helpers ───────────────────────────────────────────────────────

    def _align_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure the DataFrame columns match what the model was trained on.
        Uses the feature_names_in_ attribute added by sklearn ≥ 1.0.
        """
        if hasattr(self.model, "feature_names_in_"):
            expected = list(self.model.feature_names_in_)
            missing  = [c for c in expected if c not in df.columns]
            if missing:
                raise ValueError(
                    f"Incoming data is missing model features: {missing}"
                )
            return df[expected]

        # Fallback: drop non-numeric and target column
        X = df.select_dtypes(include=[np.number]).copy()
        for col in [config.TARGET_COLUMN, "target"]:
            if col in X.columns:
                X = X.drop(columns=[col])
        return X
