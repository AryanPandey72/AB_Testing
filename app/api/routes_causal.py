from __future__ import annotations

from fastapi import APIRouter, Query

from app.core.causal.analysis import run_causal_analysis
from app.core.causal.dag import build_criteo_dag

router = APIRouter()

@router.get("/dag")
def dag(
    outcome_col: str = Query(default="conversion"),
    treatment_col: str = Query(default="treatment"),
) -> dict:
    return build_criteo_dag(treatment_col=treatment_col, outcome_col=outcome_col)


@router.get("/causal_analysis")
def causal_analysis(
    outcome_col: str = Query(default="conversion"),
    treatment_col: str = Query(default="treatment"),
    max_rows: int = Query(default=250_000, ge=100, le=1_000_000),
) -> dict:
    return run_causal_analysis(
        outcome_col=outcome_col,
        treatment_col=treatment_col,
        max_rows=max_rows,
    )
