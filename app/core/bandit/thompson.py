from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
from uuid import uuid4

import numpy as np
from scipy.stats import beta as beta_distribution


@dataclass
class ArmState:
    arm_id: int
    arm_name: str
    alpha: float = 1.0
    beta: float = 1.0
    pulls: int = 0
    rewards: int = 0

    @property
    def posterior_mean(self) -> float:
        return float(self.alpha / (self.alpha + self.beta))

    @property
    def observed_conversion_rate(self) -> float:
        if self.pulls == 0:
            return 0.0
        return float(self.rewards / self.pulls)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["posterior_mean"] = self.posterior_mean
        payload["observed_conversion_rate"] = self.observed_conversion_rate
        return payload


@dataclass(frozen=True)
class BanditDecision:
    decision_id: str
    selected_arm_id: int
    selected_arm_name: str
    samples: dict[int, float]


class ThompsonSamplingBandit:
    def __init__(self, arms: list[ArmState], seed: int | None = None) -> None:
        if not arms:
            raise ValueError("At least one arm is required.")
        self.arms = {arm.arm_id: arm for arm in arms}
        self.rng = np.random.default_rng(seed)

    def choose_arm(self) -> BanditDecision:
        samples = {
            arm_id: float(beta_distribution.rvs(arm.alpha, arm.beta, random_state=self.rng))
            for arm_id, arm in self.arms.items()
        }
        selected_arm_id = max(samples, key=samples.get)
        selected = self.arms[selected_arm_id]
        return BanditDecision(
            decision_id=str(uuid4()),
            selected_arm_id=selected.arm_id,
            selected_arm_name=selected.arm_name,
            samples=samples,
        )

    def update(self, arm_id: int, reward: int) -> ArmState:
        if reward not in (0, 1):
            raise ValueError("Reward must be 0 or 1.")
        if arm_id not in self.arms:
            raise KeyError(f"Unknown arm_id: {arm_id}")

        arm = self.arms[arm_id]
        arm.pulls += 1
        arm.rewards += reward
        if reward == 1:
            arm.alpha += 1
        else:
            arm.beta += 1
        return arm

    def state(self) -> list[dict[str, Any]]:
        return [arm.to_dict() for arm in sorted(self.arms.values(), key=lambda x: x.arm_id)]

    def posterior_curve(self, arm_id: int, points: int = 200) -> dict[str, list[float]]:
        if arm_id not in self.arms:
            raise KeyError(f"Unknown arm_id: {arm_id}")
        arm = self.arms[arm_id]
        x = np.linspace(0.0001, 0.9999, points)
        y = beta_distribution.pdf(x, arm.alpha, arm.beta)
        return {"x": x.tolist(), "y": y.tolist()}

    def posterior_curves(self, points: int = 200) -> dict[str, dict[str, list[float]]]:
        return {
            str(arm.arm_id): self.posterior_curve(arm.arm_id, points=points)
            for arm in self.arms.values()
        }

