from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from app import settings
from app.core.bandit.service import default_arms_from_artifact
from app.core.bandit.thompson import ArmState, ThompsonSamplingBandit


def run_logged_policy_replay(
    parquet_path: Path = settings.CRITEO_HOLDOUT_PARQUET,
    outcome_col: str = settings.DEFAULT_OUTCOME_COL,
    max_rows: int = 100_000,
    seed: int = 7,
) -> dict[str, Any]:
    if not parquet_path.exists():
        return {
            "status": "missing_data",
            "message": (
                "Criteo holdout data was not found. Run "
                "`python scripts/ingest_criteo.py --csv criteo-uplift-v2.1.csv` first."
            ),
            "parquet_path": str(parquet_path),
        }

    frame = (
        pl.scan_parquet(parquet_path)
        .select(["treatment", outcome_col])
        .head(max_rows)
        .collect()
    )
    rows = frame.to_dicts()
    if not rows:
        return {"status": "empty_data", "message": "Holdout replay frame is empty."}

    reward_rates = (
        frame.group_by("treatment")
        .agg(pl.col(outcome_col).mean().alias("mean_reward"))
        .sort("treatment")
        .to_dicts()
    )
    mean_by_arm = {int(row["treatment"]): float(row["mean_reward"]) for row in reward_rates}
    optimal_rate = max(mean_by_arm.values())

    bandit = ThompsonSamplingBandit(_fresh_arms(), seed=seed)

    ts_reward = 0
    ts_accepted = 0
    ab_reward = 0
    ab_accepted = 0
    ts_history = []
    ab_history = []

    for i, row in enumerate(rows):
        logged_arm = int(row["treatment"])
        reward = int(row[outcome_col])

        decision = bandit.choose_arm()
        if decision.selected_arm_id == logged_arm:
            ts_accepted += 1
            ts_reward += reward
            bandit.update(logged_arm, reward)
            ts_history.append(
                {
                    "accepted_events": ts_accepted,
                    "cumulative_reward": ts_reward,
                    "cumulative_regret": ts_accepted * optimal_rate - ts_reward,
                }
            )

        fixed_arm = i % 2
        if fixed_arm == logged_arm:
            ab_accepted += 1
            ab_reward += reward
            ab_history.append(
                {
                    "accepted_events": ab_accepted,
                    "cumulative_reward": ab_reward,
                    "cumulative_regret": ab_accepted * optimal_rate - ab_reward,
                }
            )

    revenue_per_conversion = 100.0
    reward_delta = ts_reward - ab_reward

    return {
        "status": "ok",
        "rows_scanned": len(rows),
        "outcome_col": outcome_col,
        "empirical_reward_rates": mean_by_arm,
        "optimal_reward_rate": optimal_rate,
        "thompson": {
            "accepted_events": ts_accepted,
            "cumulative_reward": ts_reward,
            "cumulative_regret": ts_history[-1]["cumulative_regret"] if ts_history else 0.0,
            "history": ts_history,
        },
        "fixed_ab": {
            "accepted_events": ab_accepted,
            "cumulative_reward": ab_reward,
            "cumulative_regret": ab_history[-1]["cumulative_regret"] if ab_history else 0.0,
            "history": ab_history,
        },
        "revenue_saved": reward_delta * revenue_per_conversion,
    }


def _fresh_arms() -> list[ArmState]:
    return [
        ArmState(
            arm_id=arm.arm_id,
            arm_name=arm.arm_name,
            alpha=arm.alpha,
            beta=arm.beta,
            pulls=0,
            rewards=0,
        )
        for arm in default_arms_from_artifact()
    ]

