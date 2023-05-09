# Owner(s): ["oncall: distributed"]
import collections
import logging
import operator
from typing import cast, Dict, List, Set

import torch
import torch.fx as fx
from torch.distributed._spmd.graph_utils import CommType
from torch.distributed._spmd.iter_graph_module import IterGraphModule
from torch.utils._pytree import tree_flatten, tree_unflatten

from .common import (
    _call_function,
    CommBlock,
    fake_tensor_mode,
    get_all_comm_blocks,
    get_comm_block,
    graph_optimization_pass,
)

aten = torch.ops.aten
logger: logging.Logger = logging.getLogger("comm_fusion")


def _scatter_wait_result(
    gm: IterGraphModule,
    fused_comm_block: CommBlock,
    comm_blocks: List[CommBlock],
    node_indices: Dict[fx.Node, int],
) -> None:
    """
    Scatters the result of the fused communication node to the original users --
    splitting the output and reshape each subitem.
    """
    last_wait_node_idx = 0
    for node in gm.graph.nodes:
        if node == fused_comm_block.comm_node:
            break
        last_wait_node_idx = max(
            node_indices.get(node, last_wait_node_idx), last_wait_node_idx
        )

    fused_comm_node = fused_comm_block.comm_node
    fused_wait_node = fused_comm_block.wait_nodes[0]

    with gm.graph.inserting_after(fused_wait_node):
        split_node = gm.graph.call_function(
            aten.split,
            (
                fused_wait_node,
                # TODO(@fegin): support symbolic shapes
                [int(cast(torch.Size, cb.shape).numel()) for cb in comm_blocks],
            ),
        )

    # Scatter the split result.
    need_sort_nodes = []
    last_split_reshape_node = split_node
    with gm.graph.inserting_after(split_node):
        for idx, comm_block in enumerate(comm_blocks):
            # Some users of the original allreduce and wait are scheduled
            # before the fused allreduce. We must move these users to a
            # correct topological sort order -- right after the last fused
            # allreduce result, the `last_split_reshape_node` variable.
            orig_wait = comm_block.wait_nodes[0]
            nodes = collections.deque(list(orig_wait.users))
            while nodes:
                user_node = nodes.popleft()
                if not isinstance(user_node, fx.Node):
                    continue
                if node_indices[user_node] < last_wait_node_idx:
                    need_sort_nodes.append(user_node)
                    nodes.extend(list(user_node.users))

            split_idx_node = gm.graph.call_function(operator.getitem, (split_node, idx))
            with gm.graph.inserting_after(split_idx_node):
                wait_output_node = gm.graph.call_function(
                    aten.reshape, (split_idx_node, comm_block.shape)
                )
            gm.graph.node_replace_all_uses_with(orig_wait, wait_output_node)

        if last_split_reshape_node == split_node:
            last_split_reshape_node = wait_output_node

    need_sort_nodes = sorted(need_sort_nodes, key=lambda node: node_indices[node])
    gm.graph.move_after(need_sort_nodes, last_split_reshape_node)

    gm.graph.eliminate_dead_code()


def _fuse_with_cat(
    gm: IterGraphModule,
    comm_blocks: List[CommBlock],
    node_indices: Dict[fx.Node, int],
) -> CommBlock:
    """
    Given a list of CommBlock (only allreduce), fuse the CommBlocks using concat.
    """
    # Find the last input node.
    last_input_node = comm_blocks[0].inputs[0]
    last_input_index = -1
    all_input_nodes = []
    for comm_block in comm_blocks:
        input_node = comm_block.inputs[0]
        # If the input node is a clone, this is CommTensor based implementation.
        if input_node.name.startswith("clone"):
            input_node = cast(fx.Node, input_node.args[0])
        all_input_nodes.append(input_node)
        index = node_indices[input_node]
        if index >= last_input_index:
            assert index != last_input_index
            last_input_node = input_node
            last_input_index = index

    # Flatten all the inputs right after the last input is ready.
    with gm.graph.inserting_after(last_input_node):
        cat_inputs = []
        for input_node in all_input_nodes:
            cat_inputs.append(
                _call_function(
                    gm, fake_tensor_mode, None, aten.flatten.using_ints, input_node
                )
            )

    with gm.graph.inserting_after(cat_inputs[0]):
        cat_node = _call_function(gm, fake_tensor_mode, None, aten.cat, cat_inputs)

    # Create a new Comm node.
    last_comm = comm_blocks[-1]
    last_comm_node = last_comm.comm_node
    last_wait_node = last_comm.wait_nodes[0]
    with gm.graph.inserting_after(cat_node):
        flatten_args, spec = tree_flatten((last_comm_node.args, last_comm_node.kwargs))
        flatten_args[0] = cat_node
        args, kwargs = tree_unflatten(flatten_args, spec)
        fused_comm_node = _call_function(
            gm,
            fake_tensor_mode,
            cat_node.meta["val"],
            last_comm_node.target,
            *args,
            **kwargs,
        )

    # Create a new Wait node.
    with gm.graph.inserting_after(fused_comm_node):
        flatten_args, spec = tree_flatten((last_wait_node.args, last_wait_node.kwargs))
        flatten_args[0] = fused_comm_node
        args, kwargs = tree_unflatten(flatten_args, spec)
        fused_wait_node = _call_function(
            gm,
            fake_tensor_mode,
            cat_node.meta["val"],
            last_wait_node.target,
            *args,
            **kwargs,
        )

    # Move the fused_comm_node and its args to right after the source node
    nodes_to_move = cat_inputs + [cat_node, fused_comm_node, fused_wait_node]
    gm.graph.move_after(nodes_to_move, last_input_node)

    tensor_meta = cat_node.meta.get("tensor_meta")
    fused_comm_block = CommBlock(
        shape=tensor_meta.shape,  # type: ignore[union-attr]
        node_list=[fused_comm_node, fused_wait_node],
        wait_nodes=[fused_wait_node],
        comm_node=fused_comm_node,
        inputs=[cat_node],
        outputs={fused_wait_node},
    )

    _scatter_wait_result(gm, fused_comm_block, comm_blocks, node_indices)

    return fused_comm_block


def _expedite_comm_ops(gm: IterGraphModule, comm_blocks: List[CommBlock]) -> None:
    node_indices = {node: i for i, node in enumerate(gm.graph.nodes)}
    for comm_block in comm_blocks:
        last_input = comm_block.comm_node
        last_input_idx = -1
        for input in comm_block.inputs:
            input_idx = node_indices[input]
            if input_idx > last_input_idx:
                last_input = input
                last_input_idx = input_idx
        gm.graph.node_append(last_input, comm_block.comm_node)


@graph_optimization_pass(
    prerequisites=[],
    apply_after=[],
)
def comm_fusion_with_concat(
    gm: IterGraphModule,
    bucket_size_mb: int,
) -> None:
    """
    Run fuse communication with concat.
    This implementation uses concat to concat the bucketed gradients.
    """
    comm_blocks = get_all_comm_blocks(gm, (CommType.ALLREDUCE, "all_reduce"))
    # First ensure the allreduce are scheduled immediately right after the gradients.
    _expedite_comm_ops(gm, comm_blocks)
    # Get the comm_blocks based on the new order.
    comm_blocks = get_all_comm_blocks(gm, (CommType.ALLREDUCE, "all_reduce"))
    node_indices = {node: i for i, node in enumerate(gm.graph.nodes)}

    bucket_size = 1 * 1024**2
    bucket_cap_size = bucket_size_mb * 1024**2
    begin = end = curr_size = 0
    while end < len(comm_blocks):
        # TODO: determine the dtype
        curr_size += cast(torch.Size, comm_blocks[end].shape).numel() * 4
        end += 1
        if curr_size < bucket_size:
            continue
        _fuse_with_cat(gm, comm_blocks[begin:end], node_indices)
        bucket_size = bucket_cap_size
        begin = end
        curr_size = 0
    else:
        if begin < len(comm_blocks):
            _fuse_with_cat(gm, comm_blocks[begin:end], node_indices)


@graph_optimization_pass(
    prerequisites=[comm_fusion_with_concat],
    apply_after=[],
)
def schedule_comm_wait(gm: IterGraphModule) -> None:
    """
    Delay the execution of wait tensors of allreduce until its first user.
    """
    comm_blocks = get_all_comm_blocks(gm, (CommType.ALLREDUCE, "all_reduce"))

    # Find all the end users.
    allreduce_users: Set[fx.Node] = set()
    for allreduce in comm_blocks:
        for output in allreduce.outputs:
            allreduce_users.update(output.users)

    node_indices = {node: i for i, node in enumerate(gm.graph.nodes)}
    for allreduce in comm_blocks:
        # Find the earliest users.
        assert (
            len(allreduce.outputs) >= 1
        ), f"Found a allreduce that has zero outputs/users -- {allreduce}."
        # Initialize the target_node to be the first user of the first output.
        target_node = next(iter(next(iter(allreduce.outputs)).users))
        target_node_index = 2**31
        for user in (user for output in allreduce.outputs for user in output.users):
            index = node_indices[user]
            if index < target_node_index:
                target_node = user
                target_node_index = index

        # Move wait nodes and all the subsequent output nodes before the
        # earliest user.
        wait_idx = -1
        for wait_idx, node in enumerate(allreduce.node_list):
            if node == allreduce.wait_nodes[0]:
                break
        assert wait_idx >= 0
        gm.graph.move_before(allreduce.node_list[wait_idx:], target_node)
