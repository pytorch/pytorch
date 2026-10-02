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
import operator
import os
import sys
from collections.abc import Callable, Mapping

import torch
import torch.fx as fx
import torch.utils._pytree as pytree
from torch.fx._compatibility import compatibility
from torch.fx.immutable_collections import immutable_dict, immutable_list


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


# Operator namespaces whose ordering relative to surrounding compute is
# load-bearing (comm/compute overlap, in-place collective reuse). `symm_mem`
# additionally carries synchronization primitives (nvshmem_wait_for_signal),
# where reordering compute across the wait breaks the sync contract rather than
# just costing overlap.
_ORDER_SENSITIVE_NAMESPACES = frozenset(
    {
        "_c10d_functional",
        "_c10d_functional_autograd",
        "c10d_functional",
        "symm_mem",
        "_dtensor",
    }
)


def _computation_node_key(
    node: fx.Node, canonical_idx: dict[fx.Node, int]
) -> tuple[int, str, tuple[int, ...]]:
    """Canonical heap key for a computation node (call_function / call_method / call_module)."""
    input_indices = tuple(canonical_idx[n] for n in node.all_input_nodes)
    return (2, node.graph._target_to_str(node.target), input_indices)


def _target_identity(target: object) -> tuple[str, bool]:
    from torch.fx.node import _get_qualified_name

    if callable(target):
        try:
            qualified_name = _get_qualified_name(target)
        except Exception:
            pass
        else:
            parts = qualified_name.split(".")
            for prefix_len in range(len(parts), 0, -1):
                obj = sys.modules.get(".".join(parts[:prefix_len]))
                if obj is None:
                    continue
                try:
                    for part in parts[prefix_len:]:
                        obj = getattr(obj, part)
                except Exception:
                    break
                if obj is target:
                    return qualified_name, True
                break
        target_type = type(target)
        type_name = f"{target_type.__module__}.{target_type.__qualname__}"
        return f"{type_name}@{os.getpid()}:{id(target)}", False
    if isinstance(target, str):
        return target, True
    target_type = type(target)
    type_name = f"{target_type.__module__}.{target_type.__qualname__}"
    return f"{type_name}@{os.getpid()}:{id(target)}", False


def _stable_target_str(target: object) -> str:
    """Stringify globally resolvable targets, with a local fallback."""
    return _target_identity(target)[0]


def _target_has_stable_identity(target: object) -> bool:
    return _target_identity(target)[1]


def _stable_arg_key(value: object, node_key: Callable[[fx.Node], object]) -> object:
    """Represent an FX argument using stable, comparable Python values."""
    if isinstance(value, fx.Node):
        return node_key(value)
    if type(value) is tuple:
        return ("tuple", tuple(_stable_arg_key(item, node_key) for item in value))
    if type(value) is list:
        return ("list", tuple(_stable_arg_key(item, node_key) for item in value))
    if type(value) is dict:
        return (
            "dict",
            tuple(
                (_stable_arg_key(key, node_key), _stable_arg_key(item, node_key))
                for key, item in value.items()
            ),
        )
    if isinstance(value, torch.Size):
        return (
            "tuple_subclass",
            "torch.Size",
            tuple(_stable_arg_key(item, node_key) for item in value),
        )
    if isinstance(value, immutable_list):
        return (
            "list_subclass",
            "torch.fx.immutable_collections.immutable_list",
            tuple(_stable_arg_key(item, node_key) for item in value),
        )
    if isinstance(value, immutable_dict):
        return (
            "dict_subclass",
            "torch.fx.immutable_collections.immutable_dict",
            tuple(
                (_stable_arg_key(key, node_key), _stable_arg_key(item, node_key))
                for key, item in value.items()
            ),
        )
    if isinstance(value, (tuple, list, dict)):
        value_type = type(value)
        type_name = f"{value_type.__module__}.{value_type.__qualname__}"
        return ("opaque", type_name, os.getpid(), id(value))
    if isinstance(value, slice):
        return (
            "slice",
            _stable_arg_key(value.start, node_key),
            _stable_arg_key(value.stop, node_key),
            _stable_arg_key(value.step, node_key),
        )
    if callable(value):
        return ("callable", _stable_target_str(value))
    if isinstance(value, torch.device):
        return ("device", value.type)
    value_type = type(value)
    value_type_name = f"{value_type.__module__}.{value_type.__qualname__}"
    stable_torch_types = (
        torch.dtype,
        torch.layout,
        torch.memory_format,
        torch.SymInt,
        torch.SymFloat,
        torch.SymBool,
    )
    if type(value) in (type(None), bool, int, float, complex, str, bytes) or isinstance(
        value, stable_torch_types
    ):
        return ("constant", value_type_name, repr(value))
    return ("opaque", value_type_name, os.getpid(), id(value))


def _arg_has_stable_identity(value: object) -> bool:
    if isinstance(value, fx.Node):
        return True
    if type(value) is tuple:
        return all(_arg_has_stable_identity(item) for item in value)
    if type(value) is list:
        return all(_arg_has_stable_identity(item) for item in value)
    if type(value) is dict:
        return all(
            _arg_has_stable_identity(key) and _arg_has_stable_identity(item)
            for key, item in value.items()
        )
    if isinstance(value, (torch.Size, immutable_list)):
        return all(_arg_has_stable_identity(item) for item in value)
    if isinstance(value, immutable_dict):
        return all(
            _arg_has_stable_identity(key) and _arg_has_stable_identity(item)
            for key, item in value.items()
        )
    if isinstance(value, (tuple, list, dict)):
        return False
    if isinstance(value, slice):
        return all(
            _arg_has_stable_identity(item)
            for item in (value.start, value.stop, value.step)
        )
    if callable(value):
        return _target_has_stable_identity(value)
    if isinstance(value, torch.device):
        return True
    stable_torch_types = (
        torch.dtype,
        torch.layout,
        torch.memory_format,
        torch.SymInt,
        torch.SymFloat,
        torch.SymBool,
    )
    return type(value) in (
        type(None),
        bool,
        int,
        float,
        complex,
        str,
        bytes,
    ) or isinstance(value, stable_torch_types)


def _stable_kwarg_key(
    kwargs: Mapping[str, object], node_key: Callable[[fx.Node], object]
) -> object:
    """Represent keyword arguments independent of their insertion order."""
    return tuple(
        (key, _stable_arg_key(value, node_key)) for key, value in sorted(kwargs.items())
    )


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


# Higher order operators whose whole effect is the subgraphs they are handed:
# if every branch is pure, the HOP is pure. Anything not listed here keeps its
# trace-order position, because its effects can hide somewhere this pass cannot
# see -- a mutable custom op passed as a plain argument (auto_functionalized), a
# side-table kernel index (triton_kernel_wrapper_mutation), or a torchbind object.
_SUBGRAPH_ONLY_EFFECT_HOPS = frozenset(
    {
        "cond",
        "flex_attention_backward",
        "flex_gemm",
        "foreach_map",
        "hints_wrapper",
        "invoke_quant",
        "invoke_quant_packed",
        "local_map_hop",
        "switch",
        "while_loop",
        "while_loop_stack_output",
        "invoke_subgraph",
        "wrap",
        "tag_activation_checkpoint",
        "wrap_activation_checkpoint",
        "scan",
        "strict_mode",
        "associative_scan",
        "map_impl",
        "flex_attention",
    }
)


def _hop_effects_are_pure(
    node: fx.Node, *, owning_module: torch.nn.Module | None = None
) -> bool:
    """Whether a higher order operator node can be reordered.

    ``Node.is_impure()`` never looks inside a HOP's subgraphs. Conservatively
    keep the HOP fixed if any nested node is not independently reorderable. In
    the motivating ``cond`` failure, this prevents a read of a mutated input
    from hoisting above the HOP and observing its pre-mutation value.
    """
    name = getattr(node.target, "__name__", "")
    if name not in _SUBGRAPH_ONLY_EFFECT_HOPS:
        return False
    if owning_module is None:
        owning_module = node.graph.owning_module
    if owning_module is None:
        return False
    subgraphs: list[fx.GraphModule] = []
    for arg in pytree.tree_leaves((node.args, node.kwargs)):
        if not (isinstance(arg, fx.Node) and arg.op == "get_attr"):
            continue
        try:
            submodule = owning_module.get_submodule(str(arg.target))
        except AttributeError:
            continue
        if isinstance(submodule, fx.GraphModule):
            subgraphs.append(submodule)
    if not subgraphs:
        return False
    for submodule in subgraphs:
        if (
            name in ("tag_activation_checkpoint", "wrap_activation_checkpoint")
            and "_checkpoint_context_fn" in submodule.meta
        ):
            return False
        # Placeholders/outputs are structural; recursion covers nested HOPs.
        if not all(
            n.op in ("placeholder", "output", "get_attr") or _is_safe_to_reorder(n)
            for n in submodule.graph.nodes
        ):
            return False
    return True


def _is_safe_to_reorder(
    node: fx.Node, *, owning_module: torch.nn.Module | None = None
) -> bool:
    """Check if a node is safe to reorder during graph canonicalization.

    Builds on Node.is_impure() (used by DCE) with additional checks for cases
    it doesn't cover: in-place call_method nodes, higher order operators with
    non-reorderable subgraph nodes, functional collectives, nodes binding
    unbacked symbols, and non-OpOverload state-changing functions detected by
    no-node-arguments and no-tensor-or-symbolic-value heuristics, and opaque
    Dynamo callables.

    Returning False is a graph-scale decision, not a node-scale one: barriers
    partition the graph into segments (see ``canonicalize_graph``) and pure nodes
    are confined to their segment, so each barrier trades determinism for
    ordering safety across the whole graph. A graph with N barriers only
    canonicalizes within N+1 segments, which for data-dependent graphs degrades
    toward trace order. Add barriers only where ordering is load-bearing.
    """
    # Nodes that bind unbacked symbols (item()/_local_scalar_dense, nonzero,
    # ...) keep their trace-order position. Reordering them changes the order
    # the ShapeEnv resolves symbol replacements, which can blow up compile time
    # (superlinear SymNode.expr / _find work) even though the graph is
    # value-equivalent. Trace order is already deterministic across ranks.
    if node.meta.get("unbacked_bindings"):
        return False
    if node.op == "call_method":
        return not node.target.endswith("_")  # pyrefly: ignore[missing-attribute]
    if node.op == "call_module":
        if owning_module is None:
            return not node.is_impure()
        if not isinstance(node.target, str):
            raise AssertionError(f"Expected str target, got {type(node.target)}")
        target_module = owning_module.get_submodule(node.target)
        return not getattr(target_module, "_is_impure", False)
    if node.op != "call_function":
        return True
    if node.is_impure():
        return False
    if isinstance(node.target, torch._ops.HigherOrderOperator):
        return _hop_effects_are_pure(node, owning_module=owning_module)
    # Functional collectives (all_reduce, wait_tensor, ...) keep their
    # trace-order position. Reordering a collective relative to surrounding
    # compute changes comm/compute overlap and defeats Inductor's in-place
    # collective reuse. Trace order is already deterministic across ranks
    # (identical SPMD code), so canonicalization loses nothing here. In Dynamo
    # output graphs these appear as OpOverloadPackets, in aten graphs as
    # OpOverloads, so check both. OpOverloadPacket has no public namespace
    # accessor, hence _qualified_op_name (test_canonical_graph_collectives_are_barriers
    # covers both dispatch forms, so a rename there fails loudly).
    if isinstance(node.target, torch._ops.OpOverload):
        collective_namespace = node.target.namespace
    elif isinstance(node.target, torch._ops.OpOverloadPacket):
        collective_namespace = node.target._qualified_op_name.split("::", 1)[0]
    else:
        collective_namespace = None
    if collective_namespace in _ORDER_SENSITIVE_NAMESPACES:
        return False
    if not isinstance(node.target, torch._ops.OpOverload):
        from torch._dynamo import trace_rules

        # The allow_in_graph contract permits input mutation, but Dynamo does
        # not inspect the callable to record its effects.
        if trace_rules.is_callable_allowed(node.target):
            return False
        name = getattr(node.target, "__name__", "")
        if (
            name == "flat_apply_capture"
            and getattr(node.target, "__module__", "")
            == "torch._dynamo.variables.torch"
        ):
            return False
        if name.endswith("_"):
            return False
        if (
            getattr(node.target, "__module__", "") == "_operator"
            and name in _IN_PLACE_OPERATORS
        ):
            return False
        if isinstance(node.kwargs.get("out"), fx.Node):
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
        # State-changing functions that consume other nodes escape the
        # no-node-arguments heuristic above (e.g. _exit_inference_mode takes
        # the token produced by _enter_inference_mode, _sdpa_kernel takes
        # _backend_from_string nodes). Dynamo and export attach a tensor or
        # symbolic example_value/val to every node whose result feeds the
        # program; state-changing calls are never wrapped as data, so a node
        # carrying no such value only exists for its side effect.
        values = [node.meta.get("example_value"), node.meta.get("val")]
        if not any(
            isinstance(v, (torch.Tensor, torch.SymInt, torch.SymFloat, torch.SymBool))
            for v in pytree.tree_leaves(values)
        ):
            return False
    return True


@compatibility(is_backward_compatible=False)
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
    return _rename_nodes_to_canonical_in_order(graph, list(graph.nodes), skip_ops)


def _rename_nodes_to_canonical_in_order(
    graph: fx.Graph,
    order: list[fx.Node],
    skip_ops: frozenset[str],
) -> dict[str, str]:
    from torch.fx.graph import _Namespace

    renamed: dict[str, str] = {}
    ns = _Namespace()
    for node in order:
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


def _canonicalize_graph_node_names(
    graph: fx.Graph,
    canonical_key_fn: Callable[[fx.Node, dict[fx.Node, int]], object],
    *,
    skip_rename_ops: frozenset[str] = frozenset(),
) -> dict[str, str]:
    """Rename nodes according to an unconstrained canonical topological order.

    Unlike :func:`canonicalize_graph`, this leaves the graph's physical node
    order untouched. This is useful when raw node names need to provide stable
    identity but moving an otherwise independent node across a side-effect or
    ordering barrier would change semantics or scheduling.
    """
    indeg = {node: len(node.all_input_nodes) for node in graph.nodes}
    canonical_idx: dict[fx.Node, int] = {}
    counter = itertools.count()
    ready = [
        (canonical_key_fn(node, canonical_idx), next(counter), node)
        for node in graph.nodes
        if indeg[node] == 0
    ]
    heapq.heapify(ready)
    canonical_order: list[fx.Node] = []

    while ready:
        _, _, node = heapq.heappop(ready)
        canonical_order.append(node)
        canonical_idx[node] = len(canonical_idx)
        for user in node.users:
            indeg[user] -= 1
            if indeg[user] == 0:
                heapq.heappush(
                    ready,
                    (canonical_key_fn(user, canonical_idx), next(counter), user),
                )

    if len(canonical_order) != len(indeg):
        remaining = [node for node, degree in indeg.items() if degree != 0]
        raise RuntimeError(
            f"Canonical naming failed: processed {len(canonical_order)} of "
            f"{len(indeg)} nodes. Remaining: {remaining}"
        )

    return _rename_nodes_to_canonical_in_order(graph, canonical_order, skip_rename_ops)


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


def _group_getitem_nodes(order: list[fx.Node]) -> None:
    """Move each getitem node to immediately after its producer, in index order.

    Serialization does not store getitem nodes as free-standing nodes; the
    deserializer reconstructs them right after their producer, index-ordered
    (see ``generate_getitems`` in ``torch/_export/serde/serialize.py``).  If
    canonicalization leaves getitems interleaved with other computation, the
    serialize/deserialize roundtrip produces a different node order.  This pass
    normalizes getitem placement to match the deserializer so the canonical
    form is roundtrip-stable.  Nested getitems (from list-typed outputs) are
    emitted depth-first, matching the recursive reconstruction.

    Relies on each getitem's producer (``args[0]``) being a node present in
    ``order``, and on a getitem sharing its producer's ``nn_module_stack``
    scope (so pulling it adjacent stays within that scope).  Both hold for
    graphs FX/export produce; the count assertion below guards against silent
    node loss if they ever don't.
    """
    children: dict[fx.Node, list[fx.Node]] = collections.defaultdict(list)
    getitems: set[fx.Node] = set()
    for node in order:
        if node.op == "call_function" and node.target is operator.getitem:
            children[node.args[0]].append(node)  # type: ignore[index]
            getitems.add(node)
    if not getitems:
        return

    def _index(node: fx.Node) -> tuple[int, int, str]:
        idx = node.args[1]
        return (0, idx, "") if isinstance(idx, int) else (1, 0, str(idx))

    for group in children.values():
        group.sort(key=_index)

    new_order: list[fx.Node] = []

    def _emit(node: fx.Node) -> None:
        new_order.append(node)
        for child in children.get(node, ()):
            _emit(child)

    for node in order:
        if node not in getitems:
            _emit(node)
    if len(new_order) != len(order):
        raise AssertionError(
            f"getitem grouping lost nodes: {len(new_order)} != {len(order)} "
            "(a getitem's producer is missing from the order)"
        )
    order[:] = new_order


@compatibility(is_backward_compatible=False)
def canonicalize_graph(
    graph: fx.Graph,
    canonical_key_fn: Callable[[fx.Node, dict[fx.Node, int]], object],
    is_safe_to_reorder: Callable[[fx.Node], bool],
    *,
    skip_rename_ops: frozenset[str] = frozenset(),
    group_getitems: bool = False,
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
        group_getitems: If True, place getitem nodes immediately after their
            producer (index-ordered) so the order matches what serialization's
            deserializer reconstructs. Enable for graphs that get serialized
            (e.g. export) to keep the serialize/deserialize roundtrip stable.

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
    # (same target, same input indices). These are usually CSE-equivalent and
    # value-interchangeable, so the fallback to trace order is fine. This is
    # best-effort canonical labeling: it does not refine equal-key nodes by
    # their consumers' argument positions, so two graphs that are isomorphic
    # only under swapping such nodes may still get different (but value-equal)
    # labelings.
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

    if group_getitems:
        _group_getitem_nodes(canonical_order)

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
