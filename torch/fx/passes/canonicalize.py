"""Canonicalize an FX graph's node order and names.

Provides two passes:

- ``canonicalize_graph``: reorders nodes into a deterministic topological order
  using Kahn's algorithm with a caller-supplied canonical key function and
  barrier predicate.
- ``rename_nodes_to_canonical``: renames all nodes to canonical names derived
  from their target, using the same naming scheme as ``Graph.create_node``.
"""

import collections
import heapq
import itertools
from collections.abc import Callable

import torch.fx as fx


__all__ = ["canonicalize_graph", "rename_nodes_to_canonical"]


def rename_nodes_to_canonical(graph: fx.Graph) -> None:
    """Rename all nodes in the graph to canonical names based on their target.

    Uses the same naming scheme as FX ``Graph.create_node`` (auto-generated
    names from the target string). After renaming, replaces the graph's
    namespace so future node creation stays consistent.
    """
    from torch.fx.graph import _Namespace

    ns = _Namespace()
    for node in graph.nodes:
        candidate = graph._target_to_str(node.target)
        new_name = ns.create_name(candidate, node)
        node.name = new_name
    graph._graph_namespace = ns


def _sink_get_attr_nodes(order: list[fx.Node]) -> None:
    """Move each get_attr node to right before its earliest consumer.

    By default, Kahn's algorithm places get_attr nodes (which have no data
    dependencies) at the top of the graph.  This post-processing step sinks
    each one to just before its first consumer, keeping definitions close to
    their uses.
    """
    non_ga = [n for n in order if n.op != "get_attr"]
    gas = [n for n in order if n.op == "get_attr"]
    if not gas:
        return

    pos = {n: i for i, n in enumerate(non_ga)}
    inserts: dict[int, list[fx.Node]] = collections.defaultdict(list)
    for ga in gas:
        if ga.users:
            target = min(pos.get(u, len(non_ga)) for u in ga.users)
        else:
            target = (
                len(non_ga) - 1 if non_ga and non_ga[-1].op == "output" else len(non_ga)
            )
        inserts[target].append(ga)

    order.clear()
    for i, node in enumerate(non_ga):
        order.extend(inserts.pop(i, ()))
        order.append(node)
    for remaining in inserts.values():
        order.extend(remaining)


def canonicalize_graph(
    graph: fx.Graph,
    canonical_key_fn: Callable[[fx.Node, dict[fx.Node, int]], object],
    is_safe_to_reorder: Callable[[fx.Node], bool],
) -> fx.Graph:
    """Reorder graph nodes into a canonical topological order and rename them.

    This ensures that structurally equivalent graphs produce identical node
    names and ordering, regardless of the order in which nodes were originally
    traced.

    Uses Kahn's algorithm with a canonical tiebreaker provided by
    ``canonical_key_fn``.

    Args:
        graph: The FX graph to canonicalize. Modified in-place.
        canonical_key_fn: ``(node, canonical_idx) -> comparable tuple``.
            Called when a node becomes ready.  ``canonical_idx`` maps already-
            ordered nodes to their position.  The returned tuple is used as the
            primary heap key.
        is_safe_to_reorder: ``(node) -> bool``.  Nodes for which this returns
            ``False`` act as barriers: they are chained in original order, and
            pure nodes are confined to their barrier segment.

    Returns:
        The same ``graph`` object, reordered and renamed in-place.
    """
    indeg: dict[fx.Node, int] = {
        node: len(node.all_input_nodes) for node in graph.nodes
    }

    # Nodes that aren't provably pure act as barriers. We partition the graph
    # into segments separated by barrier nodes and add synthetic edges:
    #   prev_barrier -> reorderable_nodes_in_segment -> next_barrier
    extra_users: dict[fx.Node, list[fx.Node]] = collections.defaultdict(list)
    prev_barrier: fx.Node | None = None
    segment_reorderable: list[fx.Node] = []
    for node in graph.nodes:
        if node.op in ("placeholder", "get_attr", "output"):
            continue
        is_barrier = not is_safe_to_reorder(node)
        if is_barrier:
            for reorderable in segment_reorderable:
                extra_users[reorderable].append(node)
                indeg[node] += 1
            segment_reorderable = []
        if prev_barrier is not None:
            extra_users[prev_barrier].append(node)
            indeg[node] += 1
        if is_barrier:
            prev_barrier = node
        else:
            segment_reorderable.append(node)

    canonical_idx: dict[fx.Node, int] = {}

    # The counter is a tiebreaker that prevents heapq from comparing
    # fx.Node objects (which have no __lt__). It only affects nodes with
    # identical canonical keys -- i.e., structurally equivalent operations
    # (same target, same input indices). Those are CSE candidates and
    # genuinely interchangeable, so any ordering between them is canonical.
    counter = 0
    ready: list[tuple[object, int, fx.Node]] = []
    for node in graph.nodes:
        if indeg[node] == 0:
            ready.append((canonical_key_fn(node, canonical_idx), counter, node))
            counter += 1
    heapq.heapify(ready)

    canonical_order: list[fx.Node] = []

    while ready:
        _, _, cur = heapq.heappop(ready)
        canonical_order.append(cur)
        canonical_idx[cur] = len(canonical_idx)

        for user in itertools.chain(cur.users, extra_users.get(cur, ())):
            indeg[user] -= 1
            if indeg[user] == 0:
                heapq.heappush(
                    ready,
                    (canonical_key_fn(user, canonical_idx), counter, user),
                )
                counter += 1

    if len(canonical_order) != len(graph.nodes):
        remaining = [n for n in indeg if indeg[n] != 0]
        raise RuntimeError(
            f"Canonicalization failed: processed {len(canonical_order)} of "
            f"{len(graph.nodes)} nodes. Remaining: {remaining}"
        )

    _sink_get_attr_nodes(canonical_order)

    # Reorder nodes in-place to preserve node object identity.
    cursor = graph._root  # type: ignore[attr-defined]
    for node in canonical_order:
        cursor.append(node)
        cursor = node

    rename_nodes_to_canonical(graph)

    return graph
