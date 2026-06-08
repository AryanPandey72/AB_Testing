from __future__ import annotations

import math
import socket
import subprocess
import time
from typing import Any

import networkx as nx
import plotly.graph_objects as go
import requests
import streamlit as st


import sys

@st.cache_resource
def start_api_server() -> subprocess.Popen | None:
    def is_port_in_use(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('127.0.0.1', port)) == 0

    if not is_port_in_use(8000):
        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.api.main:app", "--host", "127.0.0.1", "--port", "8000"],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        time.sleep(3) # Give it an extra second to boot
        return process
    return None

start_api_server()

st.set_page_config(
    page_title="Causal Bandit Engine",
    page_icon=None,
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.4rem; }
    [data-testid="stMetricValue"] { font-size: 1.65rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def api_base() -> str:
    return st.sidebar.text_input("API", value="http://127.0.0.1:8000").rstrip("/")


def get_json(base: str, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        response = requests.get(f"{base}{path}", params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        return {"status": "api_error", "message": str(exc)}


def post_json(base: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        response = requests.post(f"{base}{path}", json=payload or {}, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        return {"status": "api_error", "message": str(exc)}


def posterior_figure(state: dict[str, Any]) -> go.Figure:
    fig = go.Figure()
    arms = {str(arm["arm_id"]): arm for arm in state.get("arms", [])}
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]

    for i, (arm_id, curve) in enumerate(state.get("posterior_curves", {}).items()):
        arm = arms.get(arm_id, {})
        fig.add_trace(
            go.Scatter(
                x=curve["x"],
                y=curve["y"],
                mode="lines",
                name=arm.get("arm_name", f"arm {arm_id}"),
                line={"width": 3, "color": palette[i % len(palette)]},
            )
        )

    fig.update_layout(
        height=360,
        margin={"l": 24, "r": 24, "t": 24, "b": 24},
        xaxis_title="Conversion probability",
        yaxis_title="Posterior density",
        legend_title=None,
    )
    return fig


def dag_figure(dag: dict[str, Any]) -> go.Figure:
    graph = nx.DiGraph()
    kinds = {}
    for node in dag.get("nodes", []):
        graph.add_node(node["id"])
        kinds[node["id"]] = node.get("kind", "variable")
    for edge in dag.get("edges", []):
        graph.add_edge(edge["source"], edge["target"])

    if not graph.nodes:
        return go.Figure()

    pos = nx.spring_layout(graph, seed=11, k=0.9)
    edge_x = []
    edge_y = []
    for source, target in graph.edges():
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    color_by_kind = {
        "confounder": "#64748b",
        "treatment": "#d62728",
        "outcome": "#2ca02c",
        "mediator": "#ff7f0e",
    }
    node_x = []
    node_y = []
    labels = []
    colors = []
    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        labels.append(node)
        colors.append(color_by_kind.get(kinds.get(node, "variable"), "#1f77b4"))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line={"width": 1.3, "color": "#94a3b8"},
            hoverinfo="none",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=labels,
            textposition="top center",
            marker={"size": 15, "color": colors, "line": {"width": 1, "color": "#0f172a"}},
        )
    )
    fig.update_layout(
        height=430,
        margin={"l": 8, "r": 8, "t": 20, "b": 8},
        showlegend=False,
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return fig


def replay_figure(replay: dict[str, Any]) -> go.Figure:
    fig = go.Figure()
    for label, key, color in [
        ("Thompson Sampling", "thompson", "#1f77b4"),
        ("Fixed A/B", "fixed_ab", "#d62728"),
    ]:
        history = replay.get(key, {}).get("history", [])
        fig.add_trace(
            go.Scatter(
                x=[row["accepted_events"] for row in history],
                y=[row["cumulative_regret"] for row in history],
                mode="lines",
                name=label,
                line={"width": 3, "color": color},
            )
        )
    fig.update_layout(
        height=380,
        margin={"l": 24, "r": 24, "t": 24, "b": 24},
        xaxis_title="Accepted logged events",
        yaxis_title="Cumulative regret",
        legend_title=None,
    )
    return fig


base = api_base()
st.title("Causal Inference & Multi-Armed Bandit Optimization Engine")

health = get_json(base, "/health")
state = get_json(base, "/bandit_state")

if health.get("status") == "api_error":
    st.error(health["message"])
    st.stop()

arms = state.get("arms", [])
winner = max(arms, key=lambda arm: arm["posterior_mean"]) if arms else {}
total_pulls = sum(arm.get("pulls", 0) for arm in arms)
total_rewards = sum(arm.get("rewards", 0) for arm in arms)
conversion_rate = total_rewards / total_pulls if total_pulls else 0.0

cols = st.columns(4)
cols[0].metric("Live Pulls", f"{total_pulls:,}")
cols[1].metric("Live Rewards", f"{total_rewards:,}")
cols[2].metric("Observed CVR", f"{conversion_rate:.4f}")
cols[3].metric("Posterior Leader", winner.get("arm_name", "n/a"))

tabs = st.tabs(["Bandit", "Causal", "Replay"])

with tabs[0]:
    left, right = st.columns([2, 1])
    with left:
        st.plotly_chart(posterior_figure(state), use_container_width=True)
    with right:
        action = post_json(base, "/get_action", {"user_id": "dashboard"})
        st.metric("Served Arm", action.get("arm_name", "n/a"))
        reward_cols = st.columns(2)
        if reward_cols[0].button("Reward 1", use_container_width=True):
            post_json(
                base,
                "/log_reward",
                {
                    "arm_id": action.get("arm_id", 0),
                    "reward": 1,
                    "decision_id": action.get("decision_id"),
                    "user_id": "dashboard",
                },
            )
            st.rerun()
        if reward_cols[1].button("Reward 0", use_container_width=True):
            post_json(
                base,
                "/log_reward",
                {
                    "arm_id": action.get("arm_id", 0),
                    "reward": 0,
                    "decision_id": action.get("decision_id"),
                    "user_id": "dashboard",
                },
            )
            st.rerun()
        if st.button("Reset Bandit", use_container_width=True):
            post_json(base, "/reset_bandit")
            st.rerun()

    if arms:
        st.dataframe(arms, hide_index=True, use_container_width=True)

with tabs[1]:
    causal_cols = st.columns([1, 1, 2])
    outcome = causal_cols[0].selectbox("Outcome", ["conversion", "visit"], index=0)
    treatment = causal_cols[1].selectbox("Treatment", ["treatment", "exposure"], index=0)
    max_rows = causal_cols[2].slider("Rows", 10_000, 1_000_000, 250_000, step=10_000)
    causal = get_json(
        base,
        "/causal_analysis",
        params={"outcome_col": outcome, "treatment_col": treatment, "max_rows": max_rows},
    )

    if causal.get("status") == "missing_data":
        st.warning(causal["message"])
    elif causal.get("status") == "api_error":
        st.error(causal["message"])
    else:
        metric_cols = st.columns(4)
        metric_cols[0].metric("Naive ATE", f"{causal.get('naive_ate', math.nan):.6f}")
        metric_cols[1].metric("IPW ATE", f"{causal.get('ipw_ate', math.nan):.6f}")
        metric_cols[2].metric("PSM ATE", f"{causal.get('psm_ate', math.nan):.6f}")
        metric_cols[3].metric("Bias Removed", f"{causal.get('confounding_bias_removed', math.nan):.6f}")
        st.plotly_chart(dag_figure(causal.get("dag", {})), use_container_width=True)
        balance = causal.get("covariate_balance", {})
        st.dataframe(
            [
                {
                    "metric": "mean_abs_smd_before",
                    "value": balance.get("mean_abs_smd_before"),
                },
                {
                    "metric": "mean_abs_smd_after_ipw",
                    "value": balance.get("mean_abs_smd_after_ipw"),
                },
            ],
            hide_index=True,
            use_container_width=True,
        )

with tabs[2]:
    replay_cols = st.columns([1, 1, 2])
    replay_outcome = replay_cols[0].selectbox("Replay Outcome", ["conversion", "visit"], index=0)
    replay_rows = replay_cols[1].slider("Replay Rows", 1_000, 500_000, 100_000, step=1_000)
    run_replay = replay_cols[2].button("Run Logged Replay", use_container_width=True)

    if run_replay:
        replay = post_json(base, "/replay/run", {"outcome_col": replay_outcome, "max_rows": replay_rows})
        if replay.get("status") == "missing_data":
            st.warning(replay["message"])
        elif replay.get("status") == "api_error":
            st.error(replay["message"])
        else:
            replay_metrics = st.columns(4)
            replay_metrics[0].metric("Rows Scanned", f"{replay.get('rows_scanned', 0):,}")
            replay_metrics[1].metric(
                "TS Reward",
                f"{replay.get('thompson', {}).get('cumulative_reward', 0):,}",
            )
            replay_metrics[2].metric(
                "A/B Reward",
                f"{replay.get('fixed_ab', {}).get('cumulative_reward', 0):,}",
            )
            replay_metrics[3].metric("Revenue Saved", f"${replay.get('revenue_saved', 0):,.0f}")
            st.plotly_chart(replay_figure(replay), use_container_width=True)

