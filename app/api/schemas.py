from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ActionRequest(BaseModel):
    user_id: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)


class RewardRequest(BaseModel):
    arm_id: int
    reward: int = Field(ge=0, le=1)
    decision_id: str | None = None
    user_id: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)


class ReplayRequest(BaseModel):
    outcome_col: str = "conversion"
    max_rows: int = Field(default=100_000, ge=100, le=1_000_000)
    seed: int = 7

