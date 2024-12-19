from typing import List

from torch import fx


def backwards_pass_aware_graph_solver(
    memory: List[float],
    joint_graph: fx.Graph,
    max_memory: float,
    node_info,
    all_recomputable_banned_nodes: List[fx.Node],
):
    raise ValueError(
        "Backwards pass aware graph solver implementation is currently not supported."
    )
