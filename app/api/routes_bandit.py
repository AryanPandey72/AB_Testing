from __future__ import annotations

from fastapi import APIRouter

from app.api.schemas import ActionRequest, ReplayRequest, RewardRequest
from app.core.bandit.replay import run_logged_policy_replay
from app.core.bandit.service import BanditService

router = APIRouter()
bandit_service = BanditService()


@router.post("/get_action")
def get_action(request: ActionRequest) -> dict:
    return bandit_service.choose_action(user_id=request.user_id, context=request.context)


@router.post("/log_reward")
def log_reward(request: RewardRequest) -> dict:
    return bandit_service.log_reward(
        arm_id=request.arm_id,
        reward=request.reward,
        decision_id=request.decision_id,
        user_id=request.user_id,
        context=request.context,
    )


@router.get("/bandit_state")
def bandit_state() -> dict:
    return bandit_service.state()


@router.post("/reset_bandit")
def reset_bandit() -> dict:
    return bandit_service.reset()


@router.post("/simulation/run")
@router.post("/replay/run")
def replay(request: ReplayRequest) -> dict:
    return run_logged_policy_replay(
        outcome_col=request.outcome_col,
        max_rows=request.max_rows,
        seed=request.seed,
    )
