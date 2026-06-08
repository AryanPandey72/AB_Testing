from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from app import settings
from app.core.causal.dag import build_criteo_dag
from app.core.causal.estimators import CausalConfig, estimate_treatment_effects
from app.core.data.criteo_schema import FEATURE_COLUMNS, OUTCOME_COLUMNS, TREATMENT_COLUMNS


def run_causal_analysis(
    parquet_path: Path = settings.CRITEO_CAUSAL_SAMPLE_PARQUET,
    outcome_col: str = settings.DEFAULT_OUTCOME_COL,
    treatment_col: str = settings.DEFAULT_TREATMENT_COL,
    max_rows: int = 250_000,
) -> dict[str, Any]:
    if outcome_col not in OUTCOME_COLUMNS:
        raise ValueError(f"Unsupported outcome column: {outcome_col}")
    if treatment_col not in TREATMENT_COLUMNS:
        raise ValueError(f"Unsupported treatment column: {treatment_col}")
    if not parquet_path.exists():
        return {
            "status": "missing_data",
            "message": (
                "Criteo processed data was not found. Run "
                "`python scripts/ingest_criteo.py --csv criteo-uplift-v2.1.csv` first."
            ),
            "parquet_path": str(parquet_path),
            "dag": build_criteo_dag(treatment_col=treatment_col, outcome_col=outcome_col),
        }

    columns = FEATURE_COLUMNS + [treatment_col, outcome_col]
    frame = pl.scan_parquet(parquet_path).select(columns).head(max_rows).collect().to_pandas()
    result = estimate_treatment_effects(
        frame,
        CausalConfig(outcome_col=outcome_col, treatment_col=treatment_col),
    )
    result["status"] = "ok"
    result["dag"] = build_criteo_dag(treatment_col=treatment_col, outcome_col=outcome_col)
    return result

