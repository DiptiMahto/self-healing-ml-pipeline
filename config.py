"""
config.py
---------
Central configuration file for the Self-Healing ML Pipeline.
All paths, thresholds, and constants live here so every module
imports from ONE place. Changing a value here changes it everywhere.
"""

import os

# ─────────────────────────────────────────────
#  ROOT PATH
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────
#  DATA PATHS
# ─────────────────────────────────────────────
DATA_DIR            = os.path.join(BASE_DIR, "data")
TRAINING_DATA_PATH  = os.path.join(DATA_DIR, "training_data.csv")
INCOMING_DATA_PATH  = os.path.join(DATA_DIR, "incoming_data.csv")

# ─────────────────────────────────────────────
#  MODEL PATHS
# ─────────────────────────────────────────────
MODELS_DIR      = os.path.join(BASE_DIR, "models")
MODEL_PATH      = os.path.join(MODELS_DIR, "model_v1.pkl")
MODEL_VERSION   = "v1.0"

# ─────────────────────────────────────────────
#  LOGS
# ─────────────────────────────────────────────
LOGS_DIR        = os.path.join(BASE_DIR, "logs")
PIPELINE_LOG    = os.path.join(LOGS_DIR, "pipeline.log")
DRIFT_LOG       = os.path.join(LOGS_DIR, "drift_log.csv")
PREDICTIONS_LOG = os.path.join(LOGS_DIR, "predictions_log.csv")

# ─────────────────────────────────────────────
#  DRIFT DETECTION THRESHOLDS
# ─────────────────────────────────────────────
# Fraction of features allowed to drift before retraining is triggered.
# e.g. 0.3 means: if more than 30% of features show drift → retrain
DRIFT_THRESHOLD = 0.3

# p-value threshold for the KS test (per feature).
# If p < KS_P_VALUE_THRESHOLD → that feature is considered drifted.
KS_P_VALUE_THRESHOLD = 0.05

# ─────────────────────────────────────────────
#  DATA VALIDATION
# ─────────────────────────────────────────────
# Maximum allowed fraction of missing values per column
MAX_MISSING_FRACTION = 0.20

# ─────────────────────────────────────────────
#  MODEL TRAINING
# ─────────────────────────────────────────────
RANDOM_STATE        = 42
TEST_SIZE           = 0.20
TARGET_COLUMN       = "target"          # label column in training CSV
N_ESTIMATORS        = 100               # Random Forest trees

# ─────────────────────────────────────────────
#  PIPELINE METADATA  (written to logs)
# ─────────────────────────────────────────────
PIPELINE_NAME    = "Self-Healing ML Pipeline"
PIPELINE_VERSION = "1.0.0"

# ─────────────────────────────────────────────
#  SELF-HEALING INTELLIGENCE  (Decision Engine)
# ─────────────────────────────────────────────

# Minimum gap (minutes) between consecutive retraining runs.
# Prevents model from re-training on every transient data spike.
RETRAIN_COOLDOWN_MINUTES = 10

# If avg max-probability across all predictions drops below this
# threshold, the Decision Engine treats it as a signal of model
# uncertainty and may trigger retraining.
CONFIDENCE_THRESHOLD = 0.50

# ─────────────────────────────────────────────
#  ADVANCED LOG PATHS
# ─────────────────────────────────────────────
# JSON file that stores the timestamp of the last retrain event.
RETRAIN_COOLDOWN_FILE = os.path.join(LOGS_DIR, "retrain_cooldown.json")

# JSON file that stores training run metadata (accuracy, metrics, time).
TRAINING_META_FILE    = os.path.join(LOGS_DIR, "training_meta.json")

