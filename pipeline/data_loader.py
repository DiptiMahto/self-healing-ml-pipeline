"""
pipeline/data_loader.py
-----------------------
MODULE : Data Loader
PURPOSE: Reads training_data.csv and incoming_data.csv from disk.
         Also generates synthetic incoming data if no file exists yet,
         so the pipeline can run end-to-end on the first launch.

CONNECTS TO PIPELINE:
    main.py calls DataLoader().load_incoming() and .load_training()
    The returned DataFrames flow into DataValidator and DriftDetector.
"""

import os
import logging
import numpy  as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

logger = logging.getLogger(__name__)


class DataLoader:
    """Loads and (if necessary) generates datasets for the pipeline."""

    # ── public interface ──────────────────────────────────────────────────────

    def load_training(self) -> pd.DataFrame:
        """Return the training DataFrame, generating it if absent."""
        if not os.path.exists(config.TRAINING_DATA_PATH):
            logger.info("training_data.csv not found — generating from sklearn …")
            self._generate_training_data()
        df = pd.read_csv(config.TRAINING_DATA_PATH)
        logger.debug(f"Training data loaded: {df.shape}")
        return df

    def load_incoming(self) -> pd.DataFrame:
        """Return the incoming-batch DataFrame, generating it if absent."""
        if not os.path.exists(config.INCOMING_DATA_PATH):
            logger.info("incoming_data.csv not found — generating synthetic batch …")
            self._generate_incoming_data()
        df = pd.read_csv(config.INCOMING_DATA_PATH)
        # Drop target column from incoming data (we predict it, not receive it)
        if config.TARGET_COLUMN in df.columns:
            df = df.drop(columns=[config.TARGET_COLUMN])
        logger.debug(f"Incoming data loaded: {df.shape}")
        return df

    # ── private helpers ───────────────────────────────────────────────────────

    def _generate_training_data(self):
        """Save the Breast Cancer dataset as training_data.csv."""
        os.makedirs(config.DATA_DIR, exist_ok=True)
        bc   = load_breast_cancer()
        df   = pd.DataFrame(bc.data, columns=bc.feature_names)
        df[config.TARGET_COLUMN] = bc.target
        df.to_csv(config.TRAINING_DATA_PATH, index=False)
        logger.info(f"training_data.csv saved ({df.shape[0]} rows, {df.shape[1]} cols)")

    def _generate_incoming_data(self):
        """
        Create a synthetic incoming batch that partially mimics training data
        but with slight noise injected into some features — this simulates the
        real-world scenario where production data drifts gradually.
        """
        os.makedirs(config.DATA_DIR, exist_ok=True)
        bc         = load_breast_cancer()
        df_full    = pd.DataFrame(bc.data, columns=bc.feature_names)
        df_full[config.TARGET_COLUMN] = bc.target

        rng = np.random.default_rng(seed=99)

        # Sample 120 rows from the original data
        sample = df_full.sample(n=120, random_state=99).copy()

        # Inject drift into the first 5 features to simulate distribution shift
        drifting_cols = list(df_full.columns[:5])
        for col in drifting_cols:
            col_mean = sample[col].mean()
            col_std  = sample[col].std()
            noise    = rng.normal(loc=col_mean * 0.40, scale=col_std * 0.60,
                                  size=len(sample))
            sample[col] = sample[col] + noise

        # Drop target — we never reveal it in incoming batches
        sample = sample.drop(columns=[config.TARGET_COLUMN])
        sample.to_csv(config.INCOMING_DATA_PATH, index=False)
        logger.info(f"incoming_data.csv saved ({sample.shape[0]} rows, "
                    f"{sample.shape[1]} cols) — drift injected in: {drifting_cols}")
