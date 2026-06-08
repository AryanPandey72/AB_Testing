from __future__ import annotations

from threading import Lock
from typing import Any
from uuid import uuid4

from app import settings
from app.core.bandit.state_store import SQLiteBanditStateStore
from app.core.bandit.thompson import ArmState, ThompsonSamplingBandit
from app.core.data.artifacts import read_json


def default_arms_from_artifact() -> list[ArmState]:
    payload = read_json(settings.INITIAL_PRIORS_PATH, default=None)
    if payload and "arms" in payload:
        return [
            ArmState(
                arm_id=int(arm["arm_id"]),
                arm_name=str(arm["arm_name"]),
                alpha=float(arm["alpha"]),
                beta=float(arm["beta"]),
                pulls=0,
                rewards=0,
            )
            for arm in payload["arms"]
        ]

    return [
        ArmState(arm_id=0, arm_name="control", alpha=1.0, beta=1.0),
        ArmState(arm_id=1, arm_name="treatment", alpha=1.0, beta=1.0),
    ]


class BanditService:
    def __init__(self, store: SQLiteBanditStateStore | None = None) -> None:
        self.store = store or SQLiteBanditStateStore(settings.BANDIT_DB_PATH)
        self.lock = Lock()
        if not self.store.has_arms():
            self.store.reset_arms(default_arms_from_artifact())
        self.bandit = ThompsonSamplingBandit(self.store.load_arms())

    def choose_action(
        self,
        user_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self.lock:
            decision = self.bandit.choose_arm()
            self.store.record_decision(
                decision_id=decision.decision_id,
                arm_id=decision.selected_arm_id,
                samples=decision.samples,
                user_id=user_id,
                context=context,
            )
            return {
                "decision_id": decision.decision_id,
                "arm_id": decision.selected_arm_id,
                "arm_name": decision.selected_arm_name,
                "samples": {str(k): v for k, v in decision.samples.items()},
            }

    def log_reward(
        self,
        arm_id: int,
        reward: int,
        decision_id: str | None = None,
        user_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self.lock:
            arm = self.bandit.update(arm_id=arm_id, reward=reward)
            self.store.save_arm(arm)
            event_id = str(uuid4())
            self.store.record_reward(
                event_id=event_id,
                decision_id=decision_id,
                user_id=user_id,
                arm_id=arm_id,
                reward=reward,
                context=context,
            )
            return {"event_id": event_id, "updated_arm": arm.to_dict()}

    def state(self) -> dict[str, Any]:
        with self.lock:
            arms = self.bandit.state()
            curves = self.bandit.posterior_curves(points=160)
        return {"arms": arms, "posterior_curves": curves}

    def reset(self) -> dict[str, Any]:
        with self.lock:
            arms = default_arms_from_artifact()
            self.store.reset_arms(arms)
            self.bandit = ThompsonSamplingBandit(self.store.load_arms())
        return self.state()

