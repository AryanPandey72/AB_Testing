from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.bandit.thompson import ArmState


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteBanditStateStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS arms (
                    arm_id INTEGER PRIMARY KEY,
                    arm_name TEXT NOT NULL,
                    alpha REAL NOT NULL,
                    beta REAL NOT NULL,
                    pulls INTEGER NOT NULL,
                    rewards INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS decisions (
                    decision_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    arm_id INTEGER NOT NULL,
                    samples_json TEXT NOT NULL,
                    context_json TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS reward_events (
                    event_id TEXT PRIMARY KEY,
                    decision_id TEXT,
                    user_id TEXT,
                    arm_id INTEGER NOT NULL,
                    reward INTEGER NOT NULL,
                    context_json TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )

    def has_arms(self) -> bool:
        with self._connect() as conn:
            count = conn.execute("SELECT COUNT(*) AS n FROM arms").fetchone()["n"]
        return bool(count)

    def reset_arms(self, arms: list[ArmState]) -> None:
        now = utc_now()
        with self._connect() as conn:
            conn.execute("DELETE FROM arms")
            conn.execute("DELETE FROM decisions")
            conn.execute("DELETE FROM reward_events")
            conn.executemany(
                """
                INSERT INTO arms
                    (arm_id, arm_name, alpha, beta, pulls, rewards, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        arm.arm_id,
                        arm.arm_name,
                        arm.alpha,
                        arm.beta,
                        arm.pulls,
                        arm.rewards,
                        now,
                        now,
                    )
                    for arm in arms
                ],
            )

    def load_arms(self) -> list[ArmState]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT arm_id, arm_name, alpha, beta, pulls, rewards FROM arms ORDER BY arm_id"
            ).fetchall()
        return [
            ArmState(
                arm_id=int(row["arm_id"]),
                arm_name=str(row["arm_name"]),
                alpha=float(row["alpha"]),
                beta=float(row["beta"]),
                pulls=int(row["pulls"]),
                rewards=int(row["rewards"]),
            )
            for row in rows
        ]

    def save_arm(self, arm: ArmState) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE arms
                SET alpha = ?, beta = ?, pulls = ?, rewards = ?, updated_at = ?
                WHERE arm_id = ?
                """,
                (arm.alpha, arm.beta, arm.pulls, arm.rewards, utc_now(), arm.arm_id),
            )

    def record_decision(
        self,
        decision_id: str,
        arm_id: int,
        samples: dict[int, float],
        user_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO decisions
                    (decision_id, user_id, arm_id, samples_json, context_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    decision_id,
                    user_id,
                    arm_id,
                    json.dumps(samples, sort_keys=True),
                    json.dumps(context or {}, sort_keys=True),
                    utc_now(),
                ),
            )

    def record_reward(
        self,
        event_id: str,
        arm_id: int,
        reward: int,
        decision_id: str | None = None,
        user_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO reward_events
                    (event_id, decision_id, user_id, arm_id, reward, context_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    decision_id,
                    user_id,
                    arm_id,
                    reward,
                    json.dumps(context or {}, sort_keys=True),
                    utc_now(),
                ),
            )

