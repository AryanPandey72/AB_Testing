from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import settings
from app.core.data.artifacts import write_json
from app.core.data.criteo_schema import (
    BINARY_COLUMNS,
    CRITEO_COLUMNS,
    FEATURE_COLUMNS,
    validate_criteo_columns,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest local Criteo uplift CSV into Parquet.")
    parser.add_argument("--csv", type=Path, default=settings.RAW_DATASET_PATH)
    parser.add_argument("--full-output", type=Path, default=settings.CRITEO_FULL_PARQUET)
    parser.add_argument("--holdout-output", type=Path, default=settings.CRITEO_HOLDOUT_PARQUET)
    parser.add_argument("--causal-sample-output", type=Path, default=settings.CRITEO_CAUSAL_SAMPLE_PARQUET)
    parser.add_argument("--split-mod", type=int, default=5)
    parser.add_argument("--holdout-bucket", type=int, default=0)
    parser.add_argument("--causal-sample-every", type=int, default=50)
    parser.add_argument("--prior-strength", type=float, default=200.0)
    parser.add_argument("--prior-alpha", type=float, default=1.0)
    parser.add_argument("--prior-beta", type=float, default=1.0)
    parser.add_argument("--skip-full", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.csv.exists():
        raise FileNotFoundError(f"Criteo CSV not found: {args.csv}")

    print(f"Validating schema from {args.csv}")
    header = pl.read_csv(args.csv, n_rows=0).columns
    validate_criteo_columns(header)
    if args.validate_only:
        print("Schema validation complete.")
        return

    lf = _scan_criteo(args.csv)
    lf_with_row_id = _with_row_id(lf)

    settings.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    settings.DATA_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    settings.BANDIT_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_full:
        print(f"Writing full Parquet: {args.full_output}")
        _write_parquet_lazy(lf.select(CRITEO_COLUMNS), args.full_output)

    print(f"Writing holdout Parquet: {args.holdout_output}")
    holdout = (
        lf_with_row_id.filter((pl.col("row_id") % args.split_mod) == args.holdout_bucket)
        .with_columns(_shuffle_key_expr())
        .sort("shuffle_key")
        .select(CRITEO_COLUMNS)
    )
    _write_parquet_lazy(holdout, args.holdout_output)

    print(f"Writing causal sample Parquet: {args.causal_sample_output}")
    causal_sample = (
        lf_with_row_id.filter((pl.col("row_id") % args.causal_sample_every) == 0)
        .with_columns(_shuffle_key_expr())
        .sort("shuffle_key")
        .select(CRITEO_COLUMNS)
    )
    _write_parquet_lazy(causal_sample, args.causal_sample_output)

    print("Writing dataset profile artifact")
    write_json(settings.DATASET_PROFILE_PATH, _dataset_profile(lf))

    print("Writing empirical bandit priors artifact")
    write_json(
        settings.INITIAL_PRIORS_PATH,
        _initial_priors(
            lf=lf,
            prior_strength=args.prior_strength,
            prior_alpha=args.prior_alpha,
            prior_beta=args.prior_beta,
        ),
    )

    print("Criteo ingestion complete.")


def _scan_criteo(csv_path: Path) -> pl.LazyFrame:
    schema = {col: pl.Float64 for col in FEATURE_COLUMNS}
    schema.update({col: pl.Int8 for col in BINARY_COLUMNS})
    try:
        return pl.scan_csv(csv_path, schema_overrides=schema)
    except TypeError:
        return pl.scan_csv(csv_path, dtypes=schema)


def _with_row_id(lf: pl.LazyFrame) -> pl.LazyFrame:
    try:
        return lf.with_row_index("row_id")
    except AttributeError:
        return lf.with_row_count("row_id")


def _shuffle_key_expr() -> pl.Expr:
    row_id = pl.col("row_id").cast(pl.UInt64)
    return ((row_id * 1_103_515_245 + 12_345) % 2_147_483_647).alias("shuffle_key")


def _write_parquet_lazy(lf: pl.LazyFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        lf.sink_parquet(path)
    except (AttributeError, TypeError):
        lf.collect(streaming=True).write_parquet(path)


def _dataset_profile(lf: pl.LazyFrame) -> dict[str, Any]:
    profile = (
        lf.select(
            [
                pl.len().alias("rows"),
                pl.col("treatment").mean().alias("treatment_rate"),
                pl.col("exposure").mean().alias("exposure_rate"),
                pl.col("conversion").mean().alias("conversion_rate"),
                pl.col("visit").mean().alias("visit_rate"),
            ]
        )
        .collect()
        .to_dicts()[0]
    )
    return {k: _json_number(v) for k, v in profile.items()}


def _initial_priors(
    lf: pl.LazyFrame,
    prior_strength: float,
    prior_alpha: float,
    prior_beta: float,
) -> dict[str, Any]:
    stats = (
        lf.group_by("treatment")
        .agg(
            [
                pl.len().alias("observed_pulls"),
                pl.col("conversion").sum().alias("observed_rewards"),
                pl.col("conversion").mean().alias("conversion_rate"),
            ]
        )
        .sort("treatment")
        .collect()
        .to_dicts()
    )

    arms = []
    for row in stats:
        arm_id = int(row["treatment"])
        conversion_rate = float(row["conversion_rate"])
        arms.append(
            {
                "arm_id": arm_id,
                "arm_name": "control" if arm_id == 0 else "treatment",
                "alpha": prior_alpha + conversion_rate * prior_strength,
                "beta": prior_beta + (1.0 - conversion_rate) * prior_strength,
                "observed_pulls": int(row["observed_pulls"]),
                "observed_rewards": int(row["observed_rewards"]),
                "observed_conversion_rate": conversion_rate,
            }
        )

    return {
        "source": "criteo-uplift-v2.1.csv",
        "target": "conversion",
        "treatment": "treatment",
        "prior_strength": prior_strength,
        "arms": arms,
    }


def _json_number(value: Any) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    return float(value)


if __name__ == "__main__":
    main()
