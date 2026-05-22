from __future__ import annotations

import operator
from typing import Any, TYPE_CHECKING

import sympy

import torch
import torch.fx as fx
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_flatten, tree_iter


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch._ops import OpOverloadPacket
    from torch.utils._pytree import TreeSpec

aten = torch.ops.aten


def get_aten_target(node: fx.Node) -> OpOverloadPacket | Callable[..., Any] | str:
    if hasattr(node.target, "overloadpacket"):
        return node.target.overloadpacket
    return node.target


rand_ops = [
    aten.dropout,
    aten._fused_dropout,
    aten._standard_gamma,
    aten.bernoulli,
    aten.multinomial,
    aten.native_dropout,
    aten.normal,
    aten.poisson,
    aten.binomial,
    aten.rrelu,
    aten.rand_like,
    aten.rand,
    aten.randint,
    aten.randn,
    aten.randperm,
]

_MISSING = object()


def _storage_refs_from_value(value: object) -> set[StorageWeakRef] | None:
    # StorageWeakRef gives storage identity/hashing without extending the
    # lifetime of the underlying storage.
    storage_refs: set[StorageWeakRef] = set()
    for leaf in tree_iter(value):
        if not isinstance(leaf, torch.Tensor):
            continue
        try:
            storage = leaf.untyped_storage()
        except NotImplementedError:
            return None
        storage_refs.add(StorageWeakRef(storage))
    return storage_refs


def _node_storage_refs(node: fx.Node) -> set[StorageWeakRef] | None:
    if "val" not in node.meta:
        return None
    return _storage_refs_from_value(node.meta["val"])


def _node_has_tensor_storage(node: fx.Node) -> bool:
    """Check if a node carries a tensor meta value with accessible storage."""
    if "val" not in node.meta or not isinstance(node.meta["val"], torch.Tensor):
        return False
    try:
        node.meta["val"].untyped_storage()
    except NotImplementedError:
        return False
    return True


def _collect_storage_refs_from_graph_values(
    values: Any,
) -> set[StorageWeakRef] | None:
    storage_refs: set[StorageWeakRef] = set()
    for value in tree_iter(values):
        if isinstance(value, fx.Node):
            node_storage_refs = _node_storage_refs(value)
            if node_storage_refs is None:
                return None
            storage_refs.update(node_storage_refs)
        elif isinstance(value, torch.Tensor):
            value_storage_refs = _storage_refs_from_value(value)
            if value_storage_refs is None:
                return None
            storage_refs.update(value_storage_refs)
    return storage_refs


def _get_mutated_argument_values(node: fx.Node) -> tuple[Any, ...] | None:
    if (
        not isinstance(node.target, torch._ops.OpOverload)
        or node.target.namespace != "aten"
        or not node.target._schema.is_mutable
    ):
        return None

    mutated_values: list[Any] = []
    positional_idx = 0
    for schema_arg in node.target._schema.arguments:
        if schema_arg.kwarg_only:
            arg_value = node.kwargs.get(schema_arg.name, _MISSING)
        else:
            if positional_idx < len(node.args):
                arg_value = node.args[positional_idx]
            else:
                arg_value = node.kwargs.get(schema_arg.name, _MISSING)
                if arg_value is _MISSING and schema_arg.name == "self":
                    arg_value = node.kwargs.get("input", _MISSING)
            positional_idx += 1

        alias_info = schema_arg.alias_info
        if alias_info is not None and alias_info.is_write:
            if arg_value is _MISSING:
                return None
            mutated_values.append(arg_value)

    if not mutated_values:
        return None

    return tuple(mutated_values)


def _get_mutated_storage_refs(node: fx.Node) -> set[StorageWeakRef] | None:
    mutated_values = _get_mutated_argument_values(node)
    if mutated_values is None:
        return None
    return _collect_storage_refs_from_graph_values(mutated_values)


def _get_possibly_mutated_argument_values(node: fx.Node) -> tuple[Any, ...]:
    mutated_values = _get_mutated_argument_values(node)
    if mutated_values is not None:
        return mutated_values
    return (node.args, node.kwargs)


def _collect_graph_nodes_from_values(values: Any) -> set[fx.Node]:
    return {value for value in tree_iter(values) if isinstance(value, fx.Node)}


def _collect_nodes_and_ancestors(nodes: set[fx.Node]) -> set[fx.Node]:
    seen: set[fx.Node] = set()
    stack = list(nodes)
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(node.all_input_nodes)
    return seen


def _can_cse_across_mutation_regions(
    node: fx.Node,
    prev_node: fx.Node,
    is_mutation_op_fn: Callable[[fx.Node], bool],
) -> bool:
    """
    Check if ``node`` can be safely CSE'd with ``prev_node`` despite being in
    different mutation regions.

    Mutation regions are a coarse-grained mechanism: any mutable op anywhere
    in the graph increments the region counter for all subsequent nodes, even
    when the mutation is completely unrelated to the op being considered for
    CSE.  This helper performs a finer-grained check for pure aten ops.
    """
    if node.target != prev_node.target:
        return False
    if not isinstance(node.target, torch._ops.OpOverload):
        return False
    if node.target.namespace != "aten" or node.target._schema.is_mutable:
        return False
    if prev_node.meta["mutation_region_id"] > node.meta["mutation_region_id"]:
        raise AssertionError(
            "expected previous CSE candidate to precede the current node's "
            "mutation region"
        )

    input_storage_refs = _collect_storage_refs_from_graph_values(
        (node.args, node.kwargs)
    )
    node_output_storage_refs = _node_storage_refs(node)
    prev_output_storage_refs = _node_storage_refs(prev_node)
    if (
        input_storage_refs is None
        or node_output_storage_refs is None
        or prev_output_storage_refs is None
    ):
        return False

    storage_refs_to_protect = input_storage_refs | prev_output_storage_refs
    if not storage_refs_to_protect:
        return True

    cur = prev_node.next
    while cur is not node:
        if cur.op == "output":
            return False
        if is_mutation_op_fn(cur):
            mutated_storage_refs = _get_mutated_storage_refs(cur)
            # A mutable op with no schema-backed, discoverable tensor write is
            # ambiguous here, so keep the mutation-region barrier.
            if mutated_storage_refs is None or not mutated_storage_refs:
                return False
            if not mutated_storage_refs.isdisjoint(storage_refs_to_protect):
                return False
        cur = cur.next

    return True


# return a new copy of torch.fx.graph.Graph with CSE applied to the input graph
def fx_graph_cse(fx_g: torch.fx.graph.Graph) -> fx.Graph:
    new_graph = fx.Graph()
    env: dict[
        fx.Node, fx.Node
    ] = {}  # map from node in the old graph to node in the new graph
    hash_env: dict[
        tuple[str, int], fx.Node
    ] = {}  # map from hash to a node in the new graph
    token_map: dict[tuple[str, int], dict[str, Any]] = {}  # map from hash to token
    old_node_for_hash: dict[tuple[str, int], fx.Node] = {}

    from torch._inductor.pattern_matcher import (
        compute_mutation_region_ids,
        is_mutation_op,
        same_mutation_regions,
    )

    compute_mutation_region_ids(fx_g)  # type: ignore[arg-type]

    # Make a set of separate storages returned from the output, which will be preserved
    # when pruning.  This prevents us from deduplicating returned tensors which have
    # experienced identical operations, but are separate data structures in eager mode.
    output_node: fx.Node = list(fx_g.nodes)[-1]
    if output_node.op != "output":
        raise AssertionError(
            f"expected output_node.op to be 'output', got '{output_node.op}'"
        )

    output_storages = {
        StorageWeakRef(n.meta["val"].untyped_storage())
        for n in output_node.all_input_nodes
        if _node_has_tensor_storage(n)
    }
    nodes_that_alias_outputs = {
        n
        for n in fx_g.nodes
        if _node_has_tensor_storage(n)
        and StorageWeakRef(n.meta["val"].untyped_storage()) in output_storages
    }

    # If a candidate result is mutated later, CSE would redirect that mutation
    # to the earlier replacement result.  Keep such nodes distinct.
    mutated_storage_refs: set[StorageWeakRef] = set()
    nodes_that_alias_mutated_storages: set[fx.Node] = set()
    for n in fx_g.nodes:
        if is_mutation_op(n):
            possibly_mutated_values = _get_possibly_mutated_argument_values(n)
            nodes_that_alias_mutated_storages.update(
                _collect_nodes_and_ancestors(
                    _collect_graph_nodes_from_values(possibly_mutated_values)
                )
            )
            node_mutated_storage_refs = _collect_storage_refs_from_graph_values(
                possibly_mutated_values
            )
            if node_mutated_storage_refs is not None:
                mutated_storage_refs.update(node_mutated_storage_refs)

    if mutated_storage_refs:
        for n in fx_g.nodes:
            node_storage_refs = _node_storage_refs(n)
            if node_storage_refs is not None and not node_storage_refs.isdisjoint(
                mutated_storage_refs
            ):
                nodes_that_alias_mutated_storages.add(n)

    for n in fx_g.nodes:
        # The placeholder, output, and get_attr nodes are copied to the new graph without change
        # do not CSE away random operations
        if (
            n.op == "placeholder"
            or n.op == "output"
            or n.op == "get_attr"
            or get_aten_target(n) in rand_ops
            # aten.empty is non-deterministic, so don't CSE it.
            # Also, aten.empty is almost always fusible into its consumer,
            # so it's not worth CSEing.
            or get_aten_target(n) is aten.empty
            or n in nodes_that_alias_outputs
            or n in nodes_that_alias_mutated_storages
            # This CSE pass currently doesn't handle re-propagation of unbacked
            # meta where it'll sometimes eliminate a _local_scalar_dense but not
            # replace the meta of downstream users. eg. one bug we've seen is:
            #
            # _local_scalar_dense_11: "Sym(u14)" = torch.ops.aten._local_scalar_dense.default(select_10);
            # sym_sum_2: "Sym(u19 + u20 + u21)" = torch.sym_sum((_local_scalar_dense_11, _local_scalar_dense_12, _local_scalar_dense_13))
            #
            # Notice how _local_scalar_dense_11 is u14 but sym_sum_2's meta is incorrectly the old
            # pre-cse value of u19.
            or (
                "val" in n.meta
                and isinstance(n.meta["val"], sympy.Symbol)
                and free_unbacked_symbols(n.meta["val"])
            )
        ):
            new_node = new_graph.node_copy(n, lambda x: env[x])
            env[n] = new_node
        else:  # n.op == 'call_function', should never see n.op == 'call_module' or 'call_method'
            # substitute args and kwargs members to their mapping in env if exists
            # specs can be used to reconstruct nested list/dictionaries
            def substitute(
                arg_list: list[Any] | tuple[Any, ...],
            ) -> tuple[tuple[Any, ...], TreeSpec]:
                arg_list, spec = tree_flatten(arg_list)
                for i in range(len(arg_list)):
                    v = arg_list[i]
                    if isinstance(v, torch.fx.node.Node) and v in env:
                        arg_list[i] = env[v]
                    if isinstance(v, (torch.SymBool, torch.SymInt, torch.SymFloat)):
                        arg_list[i] = v.node
                return tuple(arg_list), spec

            args, args_spec = substitute(n.args)
            kwargs, kwargs_spec = substitute(n.kwargs)

            # each token corresponds to a unique node
            # nodes with the same token can be substituted
            token = {
                "target": n.target,
                "args": args,
                "args_spec": args_spec,
                "kwargs": kwargs,
                "kwargs_spec": kwargs_spec,
            }

            # hash substituted args to a number, do not hash specs because specs are not hashable
            # We need to add type into hash to avoid situations like:
            # hash((primals_2, 1.0)) == hash((primals_2, 1))
            hash_arg = hash(
                (tuple((a, type(a)) for a in args), tuple((a, type(a)) for a in kwargs))
            )
            hash_val = (n.target, hash_arg)

            # check if a node has a substitute and can be eliminated
            hash_val_in_hash_env = hash_val in hash_env
            overwrite_due_to_mutation = False
            if hash_val_in_hash_env and token_map[hash_val] == token:
                duplicate_n_prev = hash_env[hash_val]
                if same_mutation_regions(n, duplicate_n_prev):
                    env[n] = duplicate_n_prev
                    old_node_for_hash[hash_val] = n
                    continue
                elif _can_cse_across_mutation_regions(
                    n, old_node_for_hash[hash_val], is_mutation_op
                ):
                    env[n] = duplicate_n_prev
                    old_node_for_hash[hash_val] = n
                    continue
                else:
                    # any future duplicates should replace with n, not duplicate_n_prev
                    overwrite_due_to_mutation = True

            new_node = new_graph.node_copy(n, lambda x: env[x])
            env[n] = new_node
            if overwrite_due_to_mutation or not hash_val_in_hash_env:
                hash_env[hash_val] = new_node
                token_map[hash_val] = token
                old_node_for_hash[hash_val] = n

    return new_graph


def raise_getitems(gm: fx.GraphModule) -> fx.GraphModule:
    # Pre-create a list of nodes to iterate over, as modifying the node order
    # during the loop can lead to infinite loops if not handled properly.
    getitem_nodes = list(
        gm.graph.find_nodes(op="call_function", target=operator.getitem)
    )

    # loop through getitem nodes in the graph and raise them to the parent node
    # in reverse order to preserve their original relative order
    for node in reversed(getitem_nodes):
        if len(node.all_input_nodes) != 1:
            raise AssertionError(
                f"expected node {node.name} to have 1 input node, got {len(node.all_input_nodes)}"
            )
        parent = node.all_input_nodes[0]
        parent.append(node)

    gm.recompile()
    return gm


def strip_overloads(gm: fx.GraphModule) -> None:
    """
    Modifies the target of graph nodes in :attr:`gm` to strip overloads.

    Args:
        gm(fx.GraphModule): The input Fx graph module to be modified
    """
    for node in gm.graph.nodes:
        if isinstance(node.target, torch._ops.OpOverload):
            node.target = node.target.overloadpacket
    gm.recompile()


def get_placeholders(graph: fx.Graph) -> list[Any]:
    return graph.find_nodes(op="placeholder")


def get_outputs(graph: fx.Graph) -> list[fx.Node]:
    for node in graph.find_nodes(op="output"):
        return pytree.tree_leaves(node.args[0])
    raise AssertionError("No output node found")
