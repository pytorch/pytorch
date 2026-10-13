"""Minimum cut specialized for activation-checkpointing graphs.

The activation-checkpointing partitioner represents each FX node by an ``in``
and an ``out`` vertex.  The edge between them is the cost of saving that
activation, dependency edges constrain which values can be recomputed, and
the source and sink represent the forward and backward sides of the
partition.  A minimum cut therefore selects the least expensive valid set of
activations to save.

Why not use ``networkx.minimum_cut`` directly?
------------------------------------------------
NetworkX supports arbitrary graph and node types, several interchangeable
flow algorithms, reusable residual graphs, and extensive validation.  To
provide that generality it constructs a dictionary-backed residual
``DiGraph`` and performs the flow computation through nested Python mappings.
Activation-checkpointing graphs can contain tens of thousands of vertices and
are solved several times while choosing a memory budget, making that object
and traversal overhead a visible part of compilation.  Selecting NetworkX's
``dinitz`` flow function changes the flow algorithm but retains this generic
representation.

This implementation first maps arbitrary node names to dense integer indices,
then stores the residual network in flat lists.  The partitioner only needs a
directed graph with nonnegative capacities, so the additional flexibility of
the generic implementation is unnecessary here.  NetworkX remains the graph
builder and a correctness oracle in tests.

Dinic's algorithm
-----------------
Dinic's algorithm repeatedly performs two operations:

1. A breadth-first search builds a *level graph*.  It assigns each reachable
   vertex its shortest residual distance from the source.  Flow is allowed to
   move only from level ``i`` to level ``i + 1``.
2. The algorithm sends a *blocking flow* through those level-respecting edges.
   Once no source-to-sink path remains in the level graph, another breadth-
   first search starts the next phase.

When the sink is no longer reachable, the flow is maximal.  By the max-flow
min-cut theorem, residual reachability then identifies a minimum cut.  To
match ``networkx.minimum_cut`` when several cuts tie, this implementation
finds the vertices that can still reach the sink and uses them as the sink
side of the partition.

The general complexity is ``O(V^2 E)``, but the partitioner's graphs are
sparse and highly structured.  The implementation is iterative so long FX
graphs cannot overflow Python's recursion limit.  ``current`` pointers avoid
rescanning exhausted edges within a level-graph phase, and reverse edges are
stored next to their forward edges so ``edge ^ 1`` finds the reverse edge.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import networkx as nx


def minimum_cut(
    graph: Any,
    source: Any,
    sink: Any,
) -> tuple[float, tuple[set[Any], set[Any]]]:
    """Return the minimum-cut value and ``(source_side, sink_side)`` partition.

    ``graph`` must provide NetworkX-style node iteration and ``edges`` access.
    Capacities must be nonnegative.  An all-infinite source-to-sink path has no
    finite cut and raises ``NetworkXUnbounded``, matching NetworkX.  Missing
    capacities are infinite under the NetworkX graph contract.

    The returned partition matches ``networkx.minimum_cut``, including its
    choice of the cut closest to the sink when several minimum cuts tie.
    """
    if not graph.is_directed():
        raise nx.NetworkXError("minimum_cut only supports directed graphs")
    if graph.is_multigraph():
        raise nx.NetworkXError("MultiGraph and MultiDiGraph not supported")
    if source not in graph:
        raise nx.NetworkXError(f"node {source} not in graph")
    if sink not in graph:
        raise nx.NetworkXError(f"node {sink} not in graph")
    if source == sink:
        raise nx.NetworkXError("source and sink are the same node")

    nodes = list(graph)
    node_index = {node: index for index, node in enumerate(nodes)}
    node_count = len(nodes)
    source_index = node_index[source]
    sink_index = node_index[sink]

    # Match ``build_residual_network``: omit self-loops and nonpositive edges,
    # and replace infinity with three times the sum of finite capacities.
    finite_capacity = sum(
        capacity
        for start, end, capacity in graph.edges(data="capacity", default=math.inf)
        if start != end and capacity > 0 and not math.isinf(capacity)
    )
    infinite_capacity = 3 * finite_capacity if finite_capacity else 1
    adjacency: list[list[int]] = [[] for _ in nodes]
    destinations: list[int] = []
    residual: list[float] = []
    infinite: list[bool] = []

    def add_edge(start: int, end: int, capacity: float, *, is_infinite: bool) -> None:
        # Forward and reverse residual edges are adjacent, so toggling the low
        # bit moves between them without another lookup table.
        edge = len(destinations)
        destinations.extend((end, start))
        residual.extend((capacity, 0))
        infinite.extend((is_infinite, False))
        adjacency[start].append(edge)
        adjacency[end].append(edge + 1)

    for start, end, capacity in graph.edges(data="capacity", default=math.inf):
        if start == end or capacity <= 0:
            continue
        add_edge(
            node_index[start],
            node_index[end],
            min(capacity, infinite_capacity),
            is_infinite=math.isinf(capacity),
        )

    reachable = {source_index}
    queue = deque([source_index])
    while queue:
        node = queue.popleft()
        for edge in adjacency[node]:
            end = destinations[edge]
            if infinite[edge] and end not in reachable:
                if end == sink_index:
                    raise nx.NetworkXUnbounded(
                        "Infinite capacity path, flow unbounded above."
                    )
                reachable.add(end)
                queue.append(end)

    flow = 0.0
    while True:
        levels = [-1] * node_count
        levels[source_index] = 0
        queue = deque([source_index])
        while queue and levels[sink_index] < 0:
            node = queue.popleft()
            next_level = levels[node] + 1
            for edge in adjacency[node]:
                end = destinations[edge]
                if residual[edge] > 0 and levels[end] < 0:
                    levels[end] = next_level
                    queue.append(end)
        if levels[sink_index] < 0:
            break

        current = [0] * node_count

        def send_one() -> float:
            """Send flow through one level-respecting path without recursion."""
            path_nodes = [source_index]
            path_edges: list[int] = []
            bottlenecks = [math.inf]
            while path_nodes:
                node = path_nodes[-1]
                if node == sink_index:
                    amount = bottlenecks[-1]
                    for edge in path_edges:
                        residual[edge] -= amount
                        residual[edge ^ 1] += amount
                    return amount
                edges = adjacency[node]
                while current[node] < len(edges):
                    edge = edges[current[node]]
                    end = destinations[edge]
                    if residual[edge] > 0 and levels[end] == levels[node] + 1:
                        path_edges.append(edge)
                        path_nodes.append(end)
                        bottlenecks.append(min(bottlenecks[-1], residual[edge]))
                        break
                    current[node] += 1
                else:
                    levels[node] = -1
                    path_nodes.pop()
                    bottlenecks.pop()
                    if path_edges:
                        path_edges.pop()
            return 0

        while amount := send_one():
            flow += amount

    # Match ``minimum_cut``: the sink side contains every vertex that can reach
    # the sink through a residual edge.  Traverse those edges backwards.
    sink_side_indices = {sink_index}
    queue = deque([sink_index])
    while queue:
        node = queue.popleft()
        for edge in adjacency[node]:
            predecessor = destinations[edge]
            if residual[edge ^ 1] > 0 and predecessor not in sink_side_indices:
                sink_side_indices.add(predecessor)
                queue.append(predecessor)
    sink_side = {nodes[index] for index in sink_side_indices}
    return flow, (set(nodes) - sink_side, sink_side)
