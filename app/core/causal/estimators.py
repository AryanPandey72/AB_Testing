from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

import numpy as np
import pandas as pd

from app.core.data.criteo_schema import FEATURE_COLUMNS


@dataclass(frozen=True)
class CausalConfig:
    outcome_col: str = "conversion"
    treatment_col: str = "treatment"
    clip_min: float = 0.01
    clip_max: float = 0.99
    max_matching_rows: int = 50_000


def estimate_treatment_effects(frame: pd.DataFrame, config: CausalConfig) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    required = FEATURE_COLUMNS + [config.treatment_col, config.outcome_col]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Causal analysis frame is missing columns: {missing}")

    analysis_frame = frame[required].dropna().copy()
    if analysis_frame.empty:
        raise ValueError("Causal analysis frame is empty after dropping nulls.")

    treatment = analysis_frame[config.treatment_col].astype(int).to_numpy()
    outcome = analysis_frame[config.outcome_col].astype(float).to_numpy()
    features = analysis_frame[FEATURE_COLUMNS].astype(float)

    if len(np.unique(treatment)) != 2:
        raise ValueError("Treatment column must contain both 0 and 1.")

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("logit", LogisticRegression(max_iter=1_000, solver="lbfgs")),
        ]
    )
    model.fit(features, treatment)

    raw_propensity = model.predict_proba(features)[:, 1]
    propensity = np.clip(raw_propensity, config.clip_min, config.clip_max)

    naive_ate = _safe_mean(outcome[treatment == 1]) - _safe_mean(outcome[treatment == 0])
    ipw_values = treatment * outcome / propensity - (1 - treatment) * outcome / (1 - propensity)
    ipw_ate = float(np.mean(ipw_values))
    ipw_se = float(np.std(ipw_values, ddof=1) / sqrt(len(ipw_values))) if len(ipw_values) > 1 else 0.0

    psm_ate = _estimate_psm_ate(
        outcome=outcome,
        treatment=treatment,
        propensity=propensity,
        max_rows=config.max_matching_rows,
    )

    ipw_weights = treatment / propensity + (1 - treatment) / (1 - propensity)
    balance_before = _covariate_balance(features, treatment)
    balance_after = _covariate_balance(features, treatment, ipw_weights)

    return {
        "row_count": int(len(analysis_frame)),
        "outcome_col": config.outcome_col,
        "treatment_col": config.treatment_col,
        "naive_ate": float(naive_ate),
        "ipw_ate": float(ipw_ate),
        "ipw_standard_error": ipw_se,
        "ipw_confidence_interval_95": [
            float(ipw_ate - 1.96 * ipw_se),
            float(ipw_ate + 1.96 * ipw_se),
        ],
        "psm_ate": float(psm_ate),
        "confounding_bias_removed": float(naive_ate - ipw_ate),
        "propensity": {
            "min": float(np.min(propensity)),
            "mean": float(np.mean(propensity)),
            "max": float(np.max(propensity)),
        },
        "covariate_balance": {
            "mean_abs_smd_before": _mean_abs(balance_before),
            "mean_abs_smd_after_ipw": _mean_abs(balance_after),
            "before": balance_before,
            "after_ipw": balance_after,
        },
    }


def _estimate_psm_ate(
    outcome: np.ndarray,
    treatment: np.ndarray,
    propensity: np.ndarray,
    max_rows: int,
) -> float:
    if len(outcome) > max_rows:
        idx = np.linspace(0, len(outcome) - 1, num=max_rows, dtype=int)
        outcome = outcome[idx]
        treatment = treatment[idx]
        propensity = propensity[idx]

    treated_idx = np.where(treatment == 1)[0]
    control_idx = np.where(treatment == 0)[0]
    if len(treated_idx) == 0 or len(control_idx) == 0:
        return float("nan")

    control_scores = propensity[control_idx]
    order = np.argsort(control_scores)
    sorted_control_idx = control_idx[order]
    sorted_scores = control_scores[order]

    matched_effects = []
    for idx in treated_idx:
        score = propensity[idx]
        pos = int(np.searchsorted(sorted_scores, score))
        candidates = []
        if pos < len(sorted_scores):
            candidates.append(sorted_control_idx[pos])
        if pos > 0:
            candidates.append(sorted_control_idx[pos - 1])
        nearest_control = min(candidates, key=lambda c: abs(propensity[c] - score))
        matched_effects.append(outcome[idx] - outcome[nearest_control])

    return float(np.mean(matched_effects))


def _covariate_balance(
    features: pd.DataFrame,
    treatment: np.ndarray,
    weights: np.ndarray | None = None,
) -> dict[str, float]:
    weights = np.ones(len(treatment), dtype=float) if weights is None else weights.astype(float)
    balance = {}

    for column in features.columns:
        values = features[column].to_numpy(dtype=float)
        treated = treatment == 1
        control = ~treated

        mt, vt = _weighted_mean_var(values[treated], weights[treated])
        mc, vc = _weighted_mean_var(values[control], weights[control])
        pooled_sd = sqrt(max((vt + vc) / 2, 1e-12))
        balance[column] = float((mt - mc) / pooled_sd)

    return balance


def _weighted_mean_var(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0:
        return 0.0, 0.0
    mean = float(np.sum(weights * values) / weight_sum)
    variance = float(np.sum(weights * (values - mean) ** 2) / weight_sum)
    return mean, variance


def _safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else float("nan")


def _mean_abs(values: dict[str, float]) -> float:
    return float(np.mean([abs(v) for v in values.values()])) if values else float("nan")

