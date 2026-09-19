from __future__ import annotations

import math
from collections import deque
from typing import Any


def minimum_cut(
    graph: Any,
    source: Any,
    sink: Any,
) -> tuple[float, tuple[set[Any], set[Any]]]:
    nodes = list(graph)
    node_index = {node: index for index, node in enumerate(nodes)}
    node_count = len(nodes)
    source_index = node_index[source]
    sink_index = node_index[sink]

    finite_capacity = sum(
        capacity
        for start, end, capacity in graph.edges(data="capacity", default=math.inf)
        if start != end and capacity > 0 and not math.isinf(capacity)
    )
    infinite_capacity = 3 * finite_capacity if finite_capacity else 1
    adjacency: list[list[int]] = [[] for _ in nodes]
    destinations: list[int] = []
    residual: list[float] = []

    def add_edge(start: int, end: int, capacity: float) -> None:
        edge = len(destinations)
        destinations.extend((end, start))
        residual.extend((capacity, 0))
        adjacency[start].append(edge)
        adjacency[end].append(edge + 1)

    for start, end, capacity in graph.edges(data="capacity", default=math.inf):
        if start == end or capacity <= 0:
            continue
        add_edge(
            node_index[start],
            node_index[end],
            min(capacity, infinite_capacity),
        )

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

    reachable_indices = {source_index}
    queue = deque([source_index])
    while queue:
        node = queue.popleft()
        for edge in adjacency[node]:
            end = destinations[edge]
            if residual[edge] > 0 and end not in reachable_indices:
                reachable_indices.add(end)
                queue.append(end)
    reachable = {nodes[index] for index in reachable_indices}
    return flow, (reachable, set(nodes) - reachable)
