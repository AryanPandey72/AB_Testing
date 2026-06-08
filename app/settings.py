from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATASET_PATH = PROJECT_ROOT / "criteo-uplift-v2.1.csv"

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_ARTIFACTS_DIR = ARTIFACTS_DIR / "data"
BANDIT_ARTIFACTS_DIR = ARTIFACTS_DIR / "bandit"
CAUSAL_ARTIFACTS_DIR = ARTIFACTS_DIR / "causal"

CRITEO_FULL_PARQUET = PROCESSED_DATA_DIR / "criteo_full.parquet"
CRITEO_HOLDOUT_PARQUET = PROCESSED_DATA_DIR / "criteo_holdout.parquet"
CRITEO_CAUSAL_SAMPLE_PARQUET = PROCESSED_DATA_DIR / "criteo_causal_sample.parquet"

DATASET_PROFILE_PATH = DATA_ARTIFACTS_DIR / "dataset_profile.json"
INITIAL_PRIORS_PATH = BANDIT_ARTIFACTS_DIR / "initial_priors.json"
ATE_REPORT_PATH = CAUSAL_ARTIFACTS_DIR / "ate_report.json"
DAG_ARTIFACT_PATH = CAUSAL_ARTIFACTS_DIR / "dag.json"

DB_DIR = PROJECT_ROOT / "db"
BANDIT_DB_PATH = DB_DIR / "bandit_state.sqlite"

DEFAULT_OUTCOME_COL = "conversion"
DEFAULT_TREATMENT_COL = "treatment"

