from __future__ import annotations

from typing import Iterable


FEATURE_COLUMNS = [f"f{i}" for i in range(12)]
BINARY_COLUMNS = ["treatment", "conversion", "visit", "exposure"]
CRITEO_COLUMNS = FEATURE_COLUMNS + BINARY_COLUMNS
OUTCOME_COLUMNS = ["conversion", "visit"]
TREATMENT_COLUMNS = ["treatment", "exposure"]


def validate_criteo_columns(columns: Iterable[str]) -> None:
    missing = sorted(set(CRITEO_COLUMNS) - set(columns))
    if missing:
        raise ValueError(f"Criteo dataset is missing required columns: {missing}")


def polars_schema() -> dict[str, object]:
    import polars as pl

    schema: dict[str, object] = {col: pl.Float64 for col in FEATURE_COLUMNS}
    schema.update({col: pl.Int8 for col in BINARY_COLUMNS})
    return schema

