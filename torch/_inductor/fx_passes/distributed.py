# Owner(s): ["oncall: distributed"]
import collections
import logging
import operator
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, cast, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.fx as fx
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

from ..fx_utils import get_fake_args_kwargs
from ..virtualized import V

aten = torch.ops.aten
logger: logging.Logger = logging.getLogger("comm_fusion")


class CommType(str, Enum):
    ALLREDUCE = "allreduce_"
    ALLGATHER = "allgather_"
    BROADCAST = "broadcast_"
    REDUCESCATTER = "reduce_scatter_"
    SCATTER = "scatter_"


def block_move_after(block: List[fx.Node], target_node: fx.Node) -> None:
    for node in block:
        target_node.append(node)
        target_node = node


def block_move_before(block: List[fx.Node], target_node: fx.Node) -> None:
    for node in block:
        target_node.prepend(node)
        target_node = node


def call_function(
    graph: fx.Graph,
    target: Callable[..., Any],
    args: Optional[Tuple[fx.node.Argument, ...]] = None,
    kwargs: Optional[Dict[str, fx.node.Argument]] = None,
) -> fx.Node:
    node = graph.call_function(target, args, kwargs)
    is_valid, args, kwargs = get_fake_args_kwargs(node)
    assert is_valid
    with V.fake_mode:
        node.meta["val"] = target(*args, **kwargs)
        node.meta["tensor_meta"] = tree_map(
            _extract_tensor_metadata, (node.meta["val"],)
        )[0]
    return node


@dataclass(unsafe_hash=True)
class CommBlock:
    shape: Optional[torch.Size]
    node_list: List[fx.Node]
    inputs: List[fx.Node]
    wait_nodes: List[fx.Node]
    comm_node: fx.Node
    outputs: Set[fx.Node]


def get_comm_block(comm_node: fx.Node) -> CommBlock:
    """
    Given a collective node (e.g., allreduce), find out all the nodes belong to
    this communcation.

    Args:
        comm_node(fx.Node): The target communication/collective node.
    Returns:
        The CommBlock that encapsulates the related nodes (e.g., wait_node) of
        the given comm_node.
    """
    # We choose 5 to prevent some accidents that cause infinite loop. But
    # with functional collective, the distance is 1.
    MAX_WAIT_DISTANCE = 5
    node_list = []
    wait_nodes = []
    inputs, _ = tree_flatten((comm_node.args, comm_node.kwargs))
    input_nodes = [inp for inp in inputs if isinstance(inp, fx.Node)]
    distance = 0
    wait_prefixes = ("wait_comm", "wait_tensor")
    non_end_users_nodes = ("split", "reshape", "getitem", "detach", "alias")

    nodes = collections.deque([comm_node, None])
    while nodes and distance < 5:
        node = nodes.popleft()
        if node is None:
            distance += 1
            if nodes:
                nodes.append(None)
            continue
        node_list.append(node)
        if node.name.startswith(wait_prefixes):
            wait_nodes.append(node)
        else:
            for child in node.users:
                if isinstance(child, fx.Node):
                    nodes.append(child)

    if not wait_nodes:
        raise RuntimeError(
            "The wait nodes are too far away from the comm node {comm_node}."
        )

    # Identify all the outputs of this collective block.
    outputs: Set[fx.Node] = set()
    nodes = collections.deque(wait_nodes)
    while nodes:
        node = nodes.popleft()
        assert node is not None
        for user in node.users:
            if isinstance(user, fx.Node) and user.name.startswith(non_end_users_nodes):
                nodes.append(user)
                node_list.append(user)
            else:
                outputs.add(node)
                break

    # TODO: populate all the tensor metadata and remove the default.
    tensor_meta = input_nodes[0].meta.get("tensor_meta", None)
    if isinstance(tensor_meta, (tuple, list)):
        shape = tensor_meta[0]
    elif tensor_meta is not None:
        shape = torch.Size(int(s) for s in tensor_meta.shape)
    return CommBlock(
        # TODO: support symbolic shapes
        shape=shape,
        node_list=node_list,
        wait_nodes=wait_nodes,
        comm_node=comm_node,
        inputs=input_nodes,
        outputs=outputs,
    )


def get_all_comm_blocks(
    graph: fx.Graph, comm_ops: Union[Tuple[str, ...], str]
) -> List[CommBlock]:
    return [
        get_comm_block(node) for node in graph.nodes if node.name.startswith(comm_ops)
    ]


def _scatter_fused_allreduce_waits(
    graph: fx.Graph,
    fused_comm_block: CommBlock,
    comm_blocks: List[CommBlock],
    node_indices: Dict[fx.Node, int],
    split_and_reshape: bool = True,
) -> None:
    """
    Scatters the result of the fused communication node to the original users.
    If the fused method is concat splitting the output and reshape will be inserted,
    before inserting getitem. Otherwise getitem will be used as the users of the
    wait node.
    """
    last_wait_node_idx = 0
    for node in graph.nodes:
        last_wait_node_idx = max(
            node_indices.get(node, last_wait_node_idx), last_wait_node_idx
        )
        if node == comm_blocks[-1].wait_nodes[0]:
            break

    if split_and_reshape:
        fused_wait_node = fused_comm_block.wait_nodes[0]
        with graph.inserting_after(fused_wait_node):
            split_node = call_function(
                graph,
                aten.split,
                (
                    fused_wait_node,
                    # TODO(@fegin): support symbolic shapes
                    [int(cast(torch.Size, cb.shape).numel()) for cb in comm_blocks],
                ),
            )
        with graph.inserting_after(split_node):
            fused_outputs = []
            for idx, comm_block in enumerate(comm_blocks):
                split_idx_node = call_function(
                    graph, operator.getitem, (split_node, idx)
                )
                with graph.inserting_after(split_idx_node):
                    fused_outputs.append(
                        call_function(
                            graph, aten.reshape, (split_idx_node, comm_block.shape)
                        )
                    )
    else:
        fused_outputs = fused_comm_block.wait_nodes

    # Scatter the fused outputs.
    need_sort_nodes = []
    for comm_block, fused_output in zip(comm_blocks, fused_outputs):
        # Some users of the original allreduce and wait are scheduled
        # before the fused allreduce. We must move these users to a
        # correct topological sort order -- right after the last fused
        # allreduce result, the `last_fused_result` variable.
        orig_wait = comm_block.wait_nodes[0]
        nodes = collections.deque(list(orig_wait.users))
        while nodes:
            user_node = nodes.popleft()
            if not isinstance(user_node, fx.Node):
                continue
            if node_indices[user_node] < last_wait_node_idx:
                need_sort_nodes.append(user_node)
                nodes.extend(list(user_node.users))

        orig_wait.replace_all_uses_with(fused_output)

    last_fused_result = fused_outputs[0]
    fused_outputs_set = set(fused_outputs)
    for node in graph.nodes:
        if node in fused_outputs_set:
            last_fused_result = node

    need_sort_nodes = sorted(need_sort_nodes, key=lambda node: node_indices[node])
    block_move_after(need_sort_nodes, last_fused_result)



def _fuse_allreduce_by_concat(
    graph: fx.Graph,
    last_input_node: fx.Node,
    all_input_nodes: Union[fx.Node, List[fx.Node]],
    last_comm_block: CommBlock,
) -> CommBlock:
    """Given a list of inputs in order, create a fused allreduce using concat."""
    # Flatten all the inputs right after the last input is ready.
    with graph.inserting_after(last_input_node):
        cat_inputs = []
        for input_node in all_input_nodes:
            cat_inputs.append(
                call_function(graph, aten.flatten.using_ints, (input_node,))
            )

    with graph.inserting_after(cat_inputs[0]):
        cat_node = call_function(graph, aten.cat, (cat_inputs,))

    # Create a new Comm node.
    last_comm_node = last_comm_block.comm_node
    last_wait_node = last_comm_block.wait_nodes[0]
    with graph.inserting_after(cat_node):
        flatten_args, spec = tree_flatten((last_comm_node.args, last_comm_node.kwargs))
        flatten_args[0] = cat_node
        args, kwargs = tree_unflatten(flatten_args, spec)
        fused_comm_node = call_function(graph, last_comm_node.target, args, kwargs)

    # Create a new Wait node.
    with graph.inserting_after(fused_comm_node):
        flatten_args, spec = tree_flatten((last_wait_node.args, last_wait_node.kwargs))
        flatten_args[0] = fused_comm_node
        args, kwargs = tree_unflatten(flatten_args, spec)
        fused_wait_node = call_function(graph, last_wait_node.target, args, kwargs)

    # Move the fused_comm_node and its args to right after the source node
    nodes_to_move = cat_inputs + [cat_node, fused_comm_node, fused_wait_node]
    block_move_after(nodes_to_move, last_input_node)

    tensor_meta = cat_node.meta.get("tensor_meta")
    fused_comm_block = CommBlock(
        shape=tensor_meta.shape,  # type: ignore[union-attr]
        node_list=[fused_comm_node, fused_wait_node],
        wait_nodes=[fused_wait_node],
        comm_node=fused_comm_node,
        inputs=[cat_node],
        outputs={fused_wait_node},
    )

    return fused_comm_block


def _fuse_div(
    graph,
    last_input_node: fx.Node,
    all_input_nodes,
    node_indices: Dict[fx.Node, int],
) -> Optional[fx.Node]:
    if len(all_input_nodes) == 1:
        return None

    dividends = []
    divisor = 0
    for div in all_input_nodes:
        if div.target != aten.div.Tensor:
            return None

        if len(div.users) != 1:
            return None

        divisor = div.args[1] if divisor == 0 else divisor

        if div.args[1] != divisor:
            return None

        dividends.append(div.args[0])

    with graph.inserting_before(last_input_node):
        node = call_function(graph, aten._foreach_div.Scalar, (dividends, divisor))
        node_indices[node] = node_indices[last_input_node] - 1

    return node


def _fuse_allreduce_by_coalescing(
    graph: fx.GraphModule,
    last_input_node: fx.Node,
    all_input_nodes: Union[fx.Node, List[fx.Node]],
    last_comm_block: CommBlock,
    n_commblock: int
) -> CommBlock:
    """Given a list of inputs in order, create a fused allreduce by coalescing."""
    # Create a new Comm node.
    last_comm_node = last_comm_block.comm_node
    last_wait_node = last_comm_block.wait_nodes[0]
    with graph.inserting_after(last_comm_node):
        flatten_args, spec = tree_flatten((last_comm_node.args, last_comm_node.kwargs))
        flatten_args[0] = all_input_nodes
        args, kwargs = tree_unflatten(flatten_args, spec)
        fused_comm_node = call_function(
            graph, torch.ops.c10d_functional.all_reduce_coalesced.default, args, kwargs
        )

    # Create a new wait node.
    getitem_nodes = []
    wait_nodes = []
    flatten_args, spec = tree_flatten((last_wait_node.args, last_wait_node.kwargs))
    for idx in range(n_commblock):
        with graph.inserting_after(fused_comm_node):
            gi_node = call_function(graph, operator.getitem, (fused_comm_node, idx))
        getitem_nodes.append(gi_node)
        flatten_args[0] = gi_node
        args, kwargs = tree_unflatten(flatten_args, spec)
        with graph.inserting_after(gi_node):
            wait_nodes.append(call_function(graph, last_wait_node.target, args, kwargs))

    # Move the fused_comm_node and its args to right after the source node
    nodes_to_move = [fused_comm_node] + getitem_nodes + wait_nodes
    block_move_after(nodes_to_move, last_input_node)

    tensor_meta = last_comm_node.meta.get("tensor_meta")
    fused_comm_block = CommBlock(
        shape=tensor_meta.shape,  # type: ignore[union-attr]
        node_list=[fused_comm_node] + getitem_nodes + wait_nodes,
        wait_nodes=wait_nodes,
        comm_node=fused_comm_node,
        inputs=all_input_nodes,
        outputs=set(wait_nodes),
    )
    return fused_comm_block


def _fuse_allreduce(
    graph: fx.Graph,
    comm_blocks: List[CommBlock],
    node_indices: Dict[fx.Node, int],
    use_concat: bool,
) -> CommBlock:
    """Given a list of allreduce CommBlock, fuse the CommBlocks into one CommBlock."""
    last_input_node = comm_blocks[0].inputs[0]
    last_input_index = -1
    all_input_nodes = []
    for comm_block in comm_blocks:
        input_node = comm_block.inputs[0]
        all_input_nodes.append(input_node)
        index = node_indices[input_node]
        if index >= last_input_index:
            assert index != last_input_index
            last_input_node = input_node
            last_input_index = index

    if use_concat:
        input_node = _fuse_div(graph, last_input_node, all_input_nodes, node_indices)
        if input_node is not None:
            all_input_nodes = input_node
        fused_comm_block = _fuse_allreduce_by_concat(
            graph, last_input_node, all_input_nodes, comm_blocks[-1], len(comm_blocks)
        )
    else:
        input_node = _fuse_div(graph, last_input_node, all_input_nodes, node_indices)
        if input_node is not None:
            all_input_nodes = input_node
        fused_comm_block = _fuse_allreduce_by_coalescing(
            graph, last_input_node, all_input_nodes, comm_blocks[-1], len(comm_blocks)
        )

    _scatter_fused_allreduce_waits(
        graph, fused_comm_block, comm_blocks, node_indices, split_and_reshape=use_concat
    )

    for comm_block in comm_blocks:
        for wait in comm_block.wait_nodes:
            graph.erase_node(wait)
        graph.erase_node(comm_block.comm_node)
    graph.eliminate_dead_code()

    return fused_comm_block


def _expedite_comm_ops(graph: fx.Graph, comm_blocks: List[CommBlock]) -> None:
    node_indices = {node: i for i, node in enumerate(graph.nodes)}
    for comm_block in comm_blocks:
        last_input = comm_block.comm_node
        last_input_idx = -1
        for input in comm_block.inputs:
            input_idx = node_indices[input]
            if input_idx > last_input_idx:
                last_input = input
                last_input_idx = input_idx
        last_input.append(comm_block.comm_node)


def _optimize_input_output(comm_blocks, graph, node_indices):
    """
    This function detects the `copy_()` added when tracing DDP.  At this moment,
    Inductor is unable to detect this pattern and remove the `copy_()`.
    """
    copies = []
    divs = []
    for comm_block in comm_blocks:
        if len(comm_block.inputs) != 1:
            return

        input_ = next(iter(comm_block.inputs))
        if input_.target != aten.div.Tensor:
            continue
        divs.append(input_)

        """
        valid = True
        candidates = []
        for child in comm_block.outputs:
            if len(child.users) != 1:
                valid = False
                break
            grand_child = next(iter(child.users))
            if grand_child.target != aten.copy.default:
                valid = False
                break
            candidates.append(grand_child)

        if not valid:
            return
        copies.extend(candidates)
        """


    # Failed to detect the copies. Don't do any optimization.
    if not copies:
        return

    src = []
    dst = []
    first_user = None
    first_idx = 0
    for copy in copies:
        dst.append(copy.args[0])
        src.append(copy.args[1])
        for user in copy.users:
            idx = node_indices[user]
            if first_user is None or idx < first_idx:
                first_user = user
                first_idx = idx
        copy.replace_all_uses_with(copy.args[0])

    assert first_user is not None, "There are copies but no users use the copies."
    with graph.inserting_before(first_user):
        node = call_function(graph, aten._foreach_copy, (dst, src))
        node_indices[node] = first_idx - 1


def allreduce_comm_fusion(
    graph: fx.Graph,
    bucket_size_mb: int,
    use_concat: bool = True,
) -> None:
    """
    Fuse allreduce communication. There are 2 ways to fuse allreduce:
    1. concat based solution, 2 all_reduce_coalesced. The default is concat.
    If use_concat is set to False, all_reduce_coalesced will be used.
    """
    comm_blocks = get_all_comm_blocks(graph, (CommType.ALLREDUCE, "all_reduce"))
    # First ensure the allreduce are scheduled immediately right after the gradients.
    _expedite_comm_ops(graph, comm_blocks)
    # Get the comm_blocks based on the new order.
    comm_blocks = get_all_comm_blocks(graph, (CommType.ALLREDUCE, "all_reduce"))
    node_indices = {node: i for i, node in enumerate(graph.nodes)}

    bucket_size = 1 * 1024**2
    bucket_cap_size = bucket_size_mb * 1024**2
    begin = end = curr_size = 0
    while end < len(comm_blocks):
        # TODO: determine the dtype
        curr_size += cast(torch.Size, comm_blocks[end].shape).numel() * 4
        end += 1
        if curr_size < bucket_size:
            continue
        _optimize_input_output(comm_blocks[begin:end], graph, node_indices)
        _fuse_allreduce(graph, comm_blocks[begin:end], node_indices, use_concat)
        bucket_size = bucket_cap_size
        begin = end
        curr_size = 0
    else:
        if begin < len(comm_blocks):
            _optimize_input_output(comm_blocks[begin:end], graph, node_indices)
            _fuse_allreduce(graph, comm_blocks[begin:end], node_indices, use_concat)


def schedule_comm_wait(graph: fx.Graph) -> None:
    """
    Delay the execution of wait tensors of allreduce until its first user.
    """
    comm_blocks = get_all_comm_blocks(graph, (CommType.ALLREDUCE, "all_reduce"))

    # Find all the end users.
    allreduce_users: Set[fx.Node] = set()
    for allreduce in comm_blocks:
        for output in allreduce.outputs:
            allreduce_users.update(output.users)

    node_indices = {node: i for i, node in enumerate(graph.nodes)}
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
        block_move_before(allreduce.node_list[wait_idx:], target_node)
