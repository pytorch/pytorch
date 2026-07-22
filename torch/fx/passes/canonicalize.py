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

import torch
import torch.fx as fx


__all__ = ["canonicalize_graph", "rename_nodes_to_canonical"]


_IN_PLACE_OPERATORS = frozenset(
    {
        "iadd",
        "iand",
        "iconcat",
        "ifloordiv",
        "ilshift",
        "imatmul",
        "imod",
        "imul",
        "ior",
        "ipow",
        "irshift",
        "isub",
        "itruediv",
        "ixor",
    }
)


def _computation_node_key(
    node: fx.Node, canonical_idx: dict[fx.Node, int]
) -> tuple[int, str, tuple[int, ...]]:
    """Canonical heap key for a computation node (call_function / call_method / call_module)."""
    input_indices = tuple(canonical_idx[n] for n in node.all_input_nodes)
    return (2, node.graph._target_to_str(node.target), input_indices)


def _canonical_node_key(node: fx.Node, canonical_idx: dict[fx.Node, int]) -> object:
    """Canonical heap key for get_attr, output, and computation nodes.

    Callers must handle placeholder nodes themselves (the ordering strategy
    differs between Dynamo and export) and never pass them here.
    """
    if node.op == "placeholder":
        raise AssertionError("callers must handle placeholder nodes themselves")
    if node.op == "get_attr":
        return (1, str(node.target))
    elif node.op == "output":
        return (3,)
    else:
        return _computation_node_key(node, canonical_idx)


def _is_safe_to_reorder(node: fx.Node) -> bool:
    """Check if a node is safe to reorder during graph canonicalization.

    Builds on Node.is_impure() (used by DCE) with two additional checks for
    cases it doesn't cover: in-place call_method nodes and non-OpOverload
    state-changing functions detected by a no-node-arguments heuristic.
    """
    if node.op == "call_method":
        return not node.target.endswith("_")  # pyrefly: ignore[missing-attribute]
    if node.op == "call_module":
        return not node.is_impure()
    if node.op != "call_function":
        return True
    if node.is_impure():
        return False
    if not isinstance(node.target, torch._ops.OpOverload):
        name = getattr(node.target, "__name__", "")
        if name.endswith("_"):
            return False
        if (
            getattr(node.target, "__module__", "") == "_operator"
            and name in _IN_PLACE_OPERATORS
        ):
            return False
        if isinstance(node.kwargs.get("out"), fx.Node):
            return False
        # triton_kernel_wrapper_mutation mutates tensors via kwargs but
        # is not detected by is_impure() or trailing-underscore checks.
        if name == "triton_kernel_wrapper_mutation":
            return False
        # Non-OpOverload targets with no FX Node arguments are likely
        # state-changing (e.g., _vmap_increment_nesting,
        # _set_fwd_grad_enabled). This is intentionally conservative:
        # pure constant-producing ops would also be treated as barriers,
        # but those are rare in Dynamo output graphs (constants are
        # typically lifted as placeholders or get_attr nodes).
        if not node.all_input_nodes:
            return False
        # functorch batch dim ops modify the vmap interpreter stack.
        if name in ("_add_batch_dim", "_remove_batch_dim"):
            return False
    return True


def rename_nodes_to_canonical(
    graph: fx.Graph,
    skip_ops: frozenset[str] = frozenset(),
) -> dict[str, str]:
    """Rename all nodes in the graph to canonical names based on their target.

    Uses the same naming scheme as FX ``Graph.create_node`` (auto-generated
    names from the target string). After renaming, replaces the graph's
    namespace so future node creation stays consistent.

    Args:
        graph: The FX graph whose nodes to rename.
        skip_ops: Node ops to skip renaming (their names are reserved in the
            new namespace but left unchanged).

    Returns a mapping from old name to new name for nodes that were renamed.
    """
    from torch.fx.graph import _Namespace

    renamed: dict[str, str] = {}
    ns = _Namespace()
    for node in graph.nodes:
        if node.op in skip_ops:
            ns.create_name(node.name, node)
            continue
        old_name = node.name
        candidate = graph._target_to_str(node.target)
        new_name = ns.create_name(candidate, node)
        if new_name != old_name:
            renamed[old_name] = new_name
        node.name = new_name
    graph._graph_namespace = ns
    return renamed


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
    *,
    skip_rename_ops: frozenset[str] = frozenset(),
) -> dict[str, str]:
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
        skip_rename_ops: Node ops to skip renaming. Skipped nodes keep their
            original names.

    Returns:
        A mapping from old node name to new node name for nodes that were
        renamed.
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

    # Purge erased nodes that are still physically in the linked list.
    # erase_node() sets _erased=True and unlinks the node, but stale
    # _prev/_next pointers on the erased node can leave it reachable from
    # neighbors that were inserted later.  If such a ghost node sits between
    # `cursor` and the node being appended, cursor.append() (which goes via
    # cursor._next._prepend()) corrupts the chain.
    root = graph._root  # type: ignore[attr-defined]
    node = root._next
    while node is not root:
        nxt = node._next
        if node._erased:
            node._remove_from_list()
        node = nxt

    # Reorder nodes in-place to preserve node object identity.
    cursor = root
    for node in canonical_order:
        cursor.append(node)
        cursor = node

    return rename_nodes_to_canonical(graph, skip_ops=skip_rename_ops)
