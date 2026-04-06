# Self-Healing ML Pipeline with Drift Monitoring

> **Academic Major Project** — Chandigarh University  
> Stack: Python · scikit-learn · Evidently · Streamlit · pandas · scipy

---

## What This Project Does

This system simulates a production-grade **self-healing ML pipeline** that:

1. Loads incoming data batches
2. Validates data quality
3. Detects statistical drift using the **Kolmogorov-Smirnov test**
4. Makes predictions using a trained **Random Forest** model
5. Uses a **rule-based Decision Engine** to decide whether to retrain
6. Displays everything in a **Streamlit monitoring dashboard**

---

## Project Structure

```
self_healing_ml_pipeline/
├── config.py                  ← All constants and thresholds
├── main.py                    ← Pipeline entry point
├── requirements.txt
│
├── data/
│   ├── training_data.csv      ← Breast Cancer Wisconsin dataset
│   └── incoming_data.csv      ← Simulated production batch (with drift)
│
├── models/
│   └── model_v1.pkl           ← Trained Random Forest (auto-generated)
│
├── pipeline/
│   ├── data_loader.py         ← Loads / generates datasets
│   ├── data_validator.py      ← Checks missing values, nulls, negatives
│   ├── drift_detector.py      ← KS test per feature → drift score
│   ├── model_predictor.py     ← Loads model, runs inference
│   └── decision_engine.py     ← Rule engine: continue or retrain
│
├── training/
│   └── train_model.py         ← Trains & saves model_v1.pkl
│
├── dashboard/
│   └── streamlit_app.py       ← Monitoring dashboard
│
└── logs/
    ├── pipeline.log           ← Full run logs
    ├── drift_log.csv          ← Drift scores over time
    └── predictions_log.csv    ← Prediction history
```

---

## Quick Start

### 1. Set up environment

```bash
cd self_healing_ml_pipeline
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run the pipeline

```bash
python main.py
```

This will:
- Auto-generate `training_data.csv` and `incoming_data.csv` if absent
- Train the model if `model_v1.pkl` doesn't exist
- Run all 6 pipeline stages
- Print a summary and write to `logs/`

### 3. Launch the dashboard

```bash
streamlit run dashboard/streamlit_app.py
```

Open `http://localhost:8501` in your browser.

### 4. Train model manually (optional)

```bash
python training/train_model.py
```

---

## Pipeline Flow

```
[incoming_data.csv]
        │
        ▼
  DataLoader.load_incoming()
        │
        ▼
  DataValidator.validate()
        │
        ▼
  DriftDetector.detect()         ←── compares with training_data.csv
        │
        ▼
  ModelPredictor.predict()
        │
        ▼
  DecisionEngine.decide()
        │
   ┌────┴─────┐
   │           │
   ▼           ▼
CONTINUE    RETRAIN
(log it)  (ModelTrainer.train())
```

---

## Drift Detection — How It Works

For **each numeric feature**, we run the **KS (Kolmogorov-Smirnov) two-sample test**:

- **H₀**: training and incoming data come from the same distribution  
- If `p-value < 0.05` → reject H₀ → feature is **drifted**

```
drift_score = drifted_features / total_features
```

| Score | Meaning |
|-------|---------|
| 0.0 – 0.2 | Low drift — healthy |
| 0.2 – 0.3 | Moderate drift — monitor |
| > 0.3 | High drift — **retraining triggered** |

---

## Decision Engine Rules

| Rule | Condition | Action |
|------|-----------|--------|
| 1 | Validation FAIL + Drift HIGH | RETRAIN |
| 2 | Drift score > threshold | RETRAIN |
| 3 | Validation FAIL (drift OK) | RETRAIN |
| 4 | All clear | CONTINUE |

---

## Key Configuration (`config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DRIFT_THRESHOLD` | 0.30 | Max drift fraction before retraining |
| `KS_P_VALUE_THRESHOLD` | 0.05 | p-value cutoff per feature |
| `MAX_MISSING_FRACTION` | 0.20 | Max allowed missing values |
| `N_ESTIMATORS` | 100 | Random Forest trees |
| `TEST_SIZE` | 0.20 | Train/test split ratio |

---

## Viva Q&A Cheat Sheet

**Q: What is data drift?**  
A: When the statistical distribution of production data changes compared to training data, degrading model performance.

**Q: Why KS test?**  
A: It is distribution-free (non-parametric), works for any continuous distribution, and requires no assumptions about normality.

**Q: What is a self-healing pipeline?**  
A: A pipeline that detects its own degradation (via drift) and automatically corrects itself (via retraining) without human intervention.

**Q: Why Random Forest?**  
A: Ensemble of decision trees, robust to overfitting, handles correlated features well, performs strongly on tabular medical data.

**Q: What does the Decision Engine do?**  
A: Applies a priority-ordered rule table to the drift score and validation report to decide whether to continue serving predictions or trigger retraining.

---

*Built for academic purposes — Breast Cancer Wisconsin Dataset (UCI ML Repository)*
