"""
training/train_model.py
-----------------------
MODULE : Model Trainer
PURPOSE: Trains a Random Forest classifier on training_data.csv
         (Breast Cancer Wisconsin dataset) and saves it as model_v1.pkl.

METRICS PRINTED:
  - Accuracy
  - Precision (weighted)
  - Recall    (weighted)
  - F1-score  (weighted)
  - Confusion matrix
  - Full classification report

SAVES:
  - models/model_v1.pkl         — trained model
  - logs/training_meta.json     — run metadata (accuracy, metrics, timestamp)

Can be run standalone:
    python training/train_model.py

OR called programmatically from main.py when retraining is triggered.
"""

import json
import os
import sys
import logging
from datetime import datetime

import pandas as pd
import numpy  as np
import joblib
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics         import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from pipeline.data_loader import DataLoader

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trains and saves the RandomForest ML model with full metric logging."""

    def train(self) -> dict:
        """
        Full training flow:
          1. Load training_data.csv
          2. Split into train / test  (80 / 20, stratified)
          3. Fit RandomForestClassifier using config params
          4. Evaluate on test set — print Accuracy, Precision, Recall, F1
          5. Save model  → models/model_v1.pkl
          6. Save run metadata → logs/training_meta.json

        Returns
        -------
        dict with accuracy, precision, recall, f1, confusion_matrix,
             classification_report, n_features, sample counts
        """
        logger.info("=" * 60)
        logger.info("  ModelTrainer: Starting training run …")
        logger.info(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)

        # ── Load data ─────────────────────────────────────────────────────────
        loader   = DataLoader()
        df       = loader.load_training()

        if config.TARGET_COLUMN not in df.columns:
            raise ValueError(
                f"Target column '{config.TARGET_COLUMN}' not found in training data. "
                f"Available columns: {list(df.columns)}"
            )

        X = df.drop(columns=[config.TARGET_COLUMN])
        y = df[config.TARGET_COLUMN]

        # Keep only numeric features (safety guard)
        X = X.select_dtypes(include=[np.number])

        logger.info(f"  Features : {X.shape[1]}   |   Samples : {X.shape[0]}")

        # ── Train / Test split ────────────────────────────────────────────────
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size    = config.TEST_SIZE,
            random_state = config.RANDOM_STATE,
            stratify     = y,          # maintain class distribution
        )
        logger.info(
            f"  Train samples : {len(X_train)}  |  Test samples : {len(X_test)}"
        )

        # ── Train model ───────────────────────────────────────────────────────
        clf = RandomForestClassifier(
            n_estimators = config.N_ESTIMATORS,
            random_state = config.RANDOM_STATE,
            n_jobs       = -1,          # use all CPU cores
        )
        clf.fit(X_train, y_train)
        logger.info(f"  RandomForest trained with {config.N_ESTIMATORS} trees ✓")

        # ── Evaluate ──────────────────────────────────────────────────────────
        y_pred = clf.predict(X_test)

        accuracy  = round(float(accuracy_score(y_test, y_pred)), 4)
        precision = round(float(precision_score(y_test, y_pred, average="weighted")), 4)
        recall    = round(float(recall_score(y_test, y_pred,    average="weighted")), 4)
        f1        = round(float(f1_score(y_test, y_pred,        average="weighted")), 4)
        cm        = confusion_matrix(y_test, y_pred).tolist()
        report    = classification_report(y_test, y_pred, output_dict=True)

        # Print formatted results — useful during viva / demo
        logger.info("─" * 60)
        logger.info(f"  ✅  Accuracy  : {accuracy:.4f}  ({accuracy*100:.2f}%)")
        logger.info(f"  ✅  Precision : {precision:.4f}  (weighted avg)")
        logger.info(f"  ✅  Recall    : {recall:.4f}  (weighted avg)")
        logger.info(f"  ✅  F1-Score  : {f1:.4f}  (weighted avg)")
        logger.info("─" * 60)
        logger.info(f"\n  Classification Report:\n{classification_report(y_test, y_pred)}")
        logger.info(f"  Confusion Matrix:\n  {cm}")

        # ── Save model ────────────────────────────────────────────────────────
        os.makedirs(config.MODELS_DIR, exist_ok=True)
        joblib.dump(clf, config.MODEL_PATH)
        logger.info(f"  Model saved → {config.MODEL_PATH} ✓")

        # ── Save training metadata (for dashboard & audit trail) ─────────────
        meta = {
            "timestamp":              datetime.now().isoformat(),
            "model_version":          config.MODEL_VERSION,
            "model_path":             config.MODEL_PATH,
            "n_features":             X.shape[1],
            "n_train_samples":        len(X_train),
            "n_test_samples":         len(X_test),
            "accuracy":               accuracy,
            "precision_weighted":     precision,
            "recall_weighted":        recall,
            "f1_weighted":            f1,
            "n_estimators":           config.N_ESTIMATORS,
            "random_state":           config.RANDOM_STATE,
            "test_size":              config.TEST_SIZE,
            "confusion_matrix":       cm,
            "classification_report":  report,
        }
        self._save_training_meta(meta)

        logger.info("=" * 60)
        logger.info("  Training run COMPLETE.")
        logger.info("=" * 60)

        return meta

    # ── Private helpers ───────────────────────────────────────────────────────

    def _save_training_meta(self, meta: dict):
        """
        Append the latest training run to logs/training_meta.json.
        The file stores a list — each element is one training run record.
        The dashboard reads this to display training history.
        """
        meta_file = getattr(
            config, "TRAINING_META_FILE",
            os.path.join(config.LOGS_DIR, "training_meta.json")
        )
        os.makedirs(config.LOGS_DIR, exist_ok=True)

        # Load existing history (if any)
        history: list = []
        if os.path.exists(meta_file):
            try:
                with open(meta_file, "r") as f:
                    history = json.load(f)
                if not isinstance(history, list):
                    history = []
            except Exception:
                history = []

        history.append(meta)

        with open(meta_file, "w") as f:
            json.dump(history, f, indent=2)

        logger.info(f"  Training metadata saved → {meta_file}  "
                    f"(total runs logged: {len(history)})")


# ── Standalone execution ──────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s | %(levelname)-8s | %(message)s",
    )
    trainer = ModelTrainer()
    result  = trainer.train()
    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Accuracy  : {result['accuracy']:.4f}")
    print(f"  Precision : {result['precision_weighted']:.4f}")
    print(f"  Recall    : {result['recall_weighted']:.4f}")
    print(f"  F1-Score  : {result['f1_weighted']:.4f}")
    print(f"  Model     : {result['model_path']}")
    print(f"{'='*60}\n")
