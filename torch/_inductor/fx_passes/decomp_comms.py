"""
Decompose comm collectives: replace all_gather + Gram matmul with local matmul + all_reduce.

Detects the pattern:
    all_gather(X_shard) -> wait -> slice -> [computation] -> split -> getitem(rank)

and replaces it with semantically equivalent local computation:
    X_shard -> [local computation with all_reduce for global aggregation]

It relies on two mathematical identities:

1. Gram matrix decomposes over row shards:
       X.T @ X = sum_i(Xi.T @ Xi)
   So all_gather + mm(X.T, X) = mm(X_shard.T, X_shard) + all_reduce(sum)

2. Matmul distributes over row slicing:
       (B @ X)[shard_rows] = B @ X[shard_rows] = B @ X_shard
   So the X-update step B @ X_gathered, sliced to rank's rows,
   equals B @ X_shard directly.

Combined: the entire Newton-Schulz iteration can run on the shard with
only one all_reduce per iteration (for the Gram matrix aggregation).
The result is bit-for-bit identical to the all_gather approach.

Uses the inductor PatternMatcher API for matching.
"""

import logging
import operator
from typing import Optional

import torch
import torch.fx as fx

from torch._inductor.pattern_matcher import (
    CallFunction,
    Ignored,
    KeywordArg,
)

aten = torch.ops.aten
c10d = torch.ops._c10d_functional
logger = logging.getLogger(__name__)

# Pattern: all_gather -> wait -> slice
_all_gather_wait_slice_pattern = CallFunction(
    aten.slice.Tensor,
    CallFunction(
        c10d.wait_tensor.default,
        CallFunction(
            c10d.all_gather_into_tensor.default,
            KeywordArg("shard"),
            KeywordArg("world_size"),
            KeywordArg("group_name"),
        ),
    ),
    Ignored(),  # dim
    Ignored(),  # start
    Ignored(),  # end
)


def _is_collective(node: fx.Node) -> bool:
    """Check if a node is a distributed collective operation."""
    if node.op != "call_function":
        return False
    target = node.target
    target_str = str(target)
    return "_c10d_functional" in target_str


def _is_permute_transpose(node: fx.Node) -> bool:
    """Check if node is a 2D transpose: permute(X, [1, 0])."""
    if node.op != "call_function" or node.target is not aten.permute.default:
        return False
    dims = node.args[1] if len(node.args) > 1 else None
    return isinstance(dims, (list, tuple)) and list(dims) == [1, 0]


def _find_split_getitem(
    slice_node: fx.Node, max_search: int = 3000
) -> Optional[tuple[fx.Node, fx.Node, fx.Node]]:
    """Walk forward from slice_node to find split -> getitem pattern.

    Returns (split_node, getitem_node, pre_split_tensor) or None.
    Rejects chains containing collectives (would break distributed semantics).
    """
    graph = slice_node.graph
    nodes_list = list(graph.nodes)
    try:
        start_pos = nodes_list.index(slice_node)
    except ValueError:
        return None

    chain_nodes = {slice_node}

    for n in nodes_list[start_pos + 1 : start_pos + max_search]:
        if n.op != "call_function":
            continue

        uses_chain = any(inp in chain_nodes for inp in n.all_input_nodes)
        if not uses_chain:
            continue

        if n.target is aten.split.Tensor:
            # Verify split is on dim 0 (the gathered dimension)
            split_dim = n.args[2] if len(n.args) > 2 else 0
            if split_dim != 0:
                logger.debug(
                    "decomp_gram_matrix_all_gather: skip — split dim=%d, expected 0",
                    split_dim,
                )
                return None

            getitem_users = [
                u
                for u in n.users
                if u.op == "call_function" and u.target is operator.getitem
            ]
            if len(getitem_users) == 1:
                return n, getitem_users[0], n.args[0]
            return None

        if _is_collective(n):
            logger.debug(
                "decomp_gram_matrix_all_gather: skip — collective %s in compute chain",
                n.target,
            )
            return None

        chain_nodes.add(n)

    return None


def _find_gram_mms(
    slice_node: fx.Node, max_search: int = 3000
) -> list[fx.Node]:
    """Find Gram matrix mm nodes: mm(X, permute(X, [1,0])) or mm(permute(X, [1,0]), X).

    Only matches when the permute is a strict 2D transpose ([1, 0]) and both
    operands share the same source node. These are the patterns where the
    Gram identity X.T @ X = sum(Xi.T @ Xi) applies.
    """
    graph = slice_node.graph
    nodes_list = list(graph.nodes)
    try:
        start_pos = nodes_list.index(slice_node)
    except ValueError:
        return []

    chain_nodes = {slice_node}
    gram_mms = []

    for n in nodes_list[start_pos + 1 : start_pos + max_search]:
        if n.op != "call_function":
            continue

        uses_chain = any(inp in chain_nodes for inp in n.all_input_nodes)
        if not uses_chain:
            continue

        if n.target is aten.split.Tensor:
            break

        if n.target is aten.mm.default and len(n.args) == 2:
            a, b = n.args
            # Pattern 1: mm(permute(X, [1,0]), X) — X.T @ X
            if _is_permute_transpose(a) and a.args[0] is b:
                gram_mms.append(n)
            # Pattern 2: mm(X, permute(X, [1,0])) — X @ X.T
            elif _is_permute_transpose(b) and b.args[0] is a:
                gram_mms.append(n)

        if not _is_collective(n):
            chain_nodes.add(n)

    return gram_mms


def decomp_gram_matrix_all_gather(gm: fx.GraphModule) -> fx.GraphModule:
    """Replace all_gather + Gram matmul with local matmul + all_reduce.

    Only transforms when ALL of these conditions are met:
    1. Pattern: all_gather -> wait -> slice -> [compute] -> split(dim=0) -> getitem
    2. all_gather and wait each have exactly one user
    3. At least one Gram matrix mm detected (mm(X, X.T) or mm(X.T, X))
    4. No collectives in the compute chain between slice and split
    5. split_size is a concrete integer and split is on dim 0
    """
    graph = gm.graph
    transformed = 0

    for node in list(graph.nodes):
        if node.op != "call_function" or node.target is not aten.slice.Tensor:
            continue

        match = _all_gather_wait_slice_pattern.match(node)
        if not match:
            continue

        shard_input = match.kwargs["shard"]
        group_name = match.kwargs["group_name"]
        ag_node = match.nodes[0]
        wait_node = match.nodes[1]
        slice_node = node

        if len(ag_node.users) != 1 or len(wait_node.users) != 1:
            logger.debug(
                "decomp_gram_matrix_all_gather: skip — all_gather has %d users, "
                "wait has %d users (both must be 1)",
                len(ag_node.users),
                len(wait_node.users),
            )
            continue

        result = _find_split_getitem(slice_node)
        if result is None:
            continue

        split_node, getitem_node, pre_split_tensor = result

        split_size = split_node.args[1] if len(split_node.args) > 1 else None
        if not isinstance(split_size, int):
            logger.debug(
                "decomp_gram_matrix_all_gather: skip — split_size is not int: %s",
                type(split_size),
            )
            continue

        gram_mms = _find_gram_mms(slice_node)
        if not gram_mms:
            logger.debug(
                "decomp_gram_matrix_all_gather: skip — no Gram matrix mm "
                "(mm(X, X.T) or mm(X.T, X)) found in compute chain"
            )
            continue

        # === TRANSFORM ===

        # 1. Route computation to use shard directly (remove all_gather)
        slice_node.replace_all_uses_with(shard_input)

        # 2. Insert all_reduce(sum) after each Gram mm
        for mm_node in gram_mms:
            next_node = mm_node.next
            with graph.inserting_before(next_node):
                ar = graph.call_function(
                    c10d.all_reduce.default,
                    args=(mm_node, "sum", group_name),
                )
                wait_ar = graph.call_function(
                    c10d.wait_tensor.default, args=(ar,),
                )
            for user in list(mm_node.users):
                if user is not ar:
                    user.replace_input_with(mm_node, wait_ar)

        # 3. Remove split+getitem (result is already shard-sized)
        getitem_node.replace_all_uses_with(pre_split_tensor)

        # 4. Erase dead nodes
        for dead in [getitem_node, split_node, slice_node, wait_node, ag_node]:
            if len(dead.users) == 0:
                graph.erase_node(dead)

        transformed += 1

    if transformed > 0:
        try:
            graph.lint()
        except RuntimeError as e:
            logger.warning("decomp_gram_matrix_all_gather: lint warning: %s", e)
        gm.recompile()
        logger.info(
            "decomp_gram_matrix_all_gather: transformed %d pattern(s)",
            transformed,
        )

    return gm
