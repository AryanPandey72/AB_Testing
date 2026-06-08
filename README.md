# Causal Inference & Multi-Armed Bandit Optimization Engine

A local-data experimentation engine that combines causal inference for unbiased treatment effect estimation with Bayesian bandits for real-time optimization.

This project is built around the local Criteo Uplift dataset file:

```text
criteo-uplift-v2.1.csv
```

No synthetic data is used. The raw CSV is ingested once into optimized Parquet files and small JSON artifacts. FastAPI and Streamlit never fetch or parse the raw dataset at runtime.

## Architecture

```text
Data Layer
  raw Criteo CSV
  processed Parquet files
  cached analysis artifacts
  SQLite operational state

Core Engines
  causal inference: DAG, propensity scores, IPW, PSM, ATE
  bandit optimization: Thompson Sampling, Beta updates, regret replay

Serving Layer
  FastAPI endpoints for actions, rewards, causal analysis, and replay

Frontend Layer
  Streamlit dashboard with DAG, posterior curves, regret, and KPI views
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## One-Time Criteo ETL

Run this once before starting the API or dashboard:

```bash
python scripts/ingest_criteo.py --csv criteo-uplift-v2.1.csv
```

The ETL writes:

```text
data/processed/criteo_full.parquet
data/processed/criteo_holdout.parquet
data/processed/criteo_causal_sample.parquet
artifacts/data/dataset_profile.json
artifacts/bandit/initial_priors.json
```

The bandit priors use empirical conversion rates compressed into a configurable prior strength, so the live demo remains responsive instead of being overwhelmed by millions of historical observations.

## Run API

```bash
uvicorn app.api.main:app --reload
```

Endpoints:

```text
POST /get_action
POST /log_reward
GET  /bandit_state
POST /reset_bandit
GET  /causal_analysis
GET  /dag
POST /replay/run
```

## Run Dashboard

```bash
streamlit run app/dashboard/streamlit_app.py
```

## Tests

```bash
python -m unittest
```

## Dataset Notes

The project expects the Criteo uplift schema:

```text
f0, f1, ..., f11, treatment, conversion, visit, exposure
```

Recommended modes:

```text
treatment = treatment, outcome = conversion
treatment = treatment, outcome = visit
treatment = exposure, outcome = conversion
treatment = exposure, outcome = visit
```

`treatment` is best for randomized campaign assignment analysis. `exposure` is useful for demonstrating confounding-aware methods, but it should be interpreted carefully because exposure is downstream of ad auction mechanics.

