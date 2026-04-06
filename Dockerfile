# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — Self-Healing ML Pipeline
# ─────────────────────────────────────────────────────────────────────────────
# Build:  docker build -t self-healing-ml .
# Run:    docker run -p 8501:8501 -p 8000:8000 self-healing-ml
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# --- system dependencies ------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# --- working directory --------------------------------------------------------
WORKDIR /app

# --- install Python dependencies first (layer-cached) -------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# --- copy project files -------------------------------------------------------
COPY . .

# --- create runtime directories -----------------------------------------------
RUN mkdir -p data models logs

# --- expose ports -------------------------------------------------------------
# 8501 : Streamlit dashboard
# 8000 : FastAPI  REST API
EXPOSE 8501 8000

# --- start-up script ----------------------------------------------------------
# 1. Run the pipeline once to generate data + train initial model
# 2. Start FastAPI  in the background
# 3. Start Streamlit in the foreground (keeps the container alive)
CMD ["sh", "-c", \
     "python main.py && \
      uvicorn api.app:app --host 0.0.0.0 --port 8000 & \
      streamlit run dashboard/streamlit_app.py \
          --server.port=8501 \
          --server.address=0.0.0.0 \
          --server.headless=true \
          --browser.gatherUsageStats=false"]
