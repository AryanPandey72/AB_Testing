from __future__ import annotations

from app.core.data.criteo_schema import FEATURE_COLUMNS


def build_criteo_dag(treatment_col: str = "treatment", outcome_col: str = "conversion") -> dict:
    import networkx as nx

    graph = nx.DiGraph()

    for feature in FEATURE_COLUMNS:
        graph.add_node(feature, kind="confounder", label=feature)
        graph.add_edge(feature, treatment_col)
        graph.add_edge(feature, outcome_col)

    graph.add_node(treatment_col, kind="treatment", label=treatment_col)
    graph.add_node(outcome_col, kind="outcome", label=outcome_col)
    graph.add_edge(treatment_col, outcome_col)

    if treatment_col != "exposure":
        graph.add_node("exposure", kind="mediator", label="exposure")
        graph.add_edge(treatment_col, "exposure")
        graph.add_edge("exposure", outcome_col)

    nodes = [
        {
            "id": node,
            "label": attrs.get("label", node),
            "kind": attrs.get("kind", "variable"),
        }
        for node, attrs in graph.nodes(data=True)
    ]
    edges = [{"source": source, "target": target} for source, target in graph.edges()]

    return {"nodes": nodes, "edges": edges}

