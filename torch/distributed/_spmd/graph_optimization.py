# Owner(s): ["oncall: distributed"]
import collections
import logging
import operator
import tempfile
from dataclasses import dataclass
from functools import wraps
from typing import (
    Any,
    Callable,
    cast,
    DefaultDict,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import torch
import torch.fx as fx
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.distributed._spmd.graph_utils import CommType, dump_graphs_to_files
from torch.distributed._spmd.iter_graph_module import IterGraphModule
from torch.fx.passes.shape_prop import TensorMetadata
from torch.utils._pytree import tree_flatten, tree_unflatten

logger: logging.Logger = logging.getLogger("graph_optimization")
aten = torch.ops.aten
fake_tensor_mode = FakeTensorMode()

_optimized_func: Set[str] = set()
_run_before_sets: DefaultDict[str, Set[str]] = collections.defaultdict(set)
_dump_graph_folder: str = ""


def enable_graph_optimization_dump(folder: str = ""):
    global _dump_graph_folder
    if not folder:
        folder = tempfile.mkdtemp()
    _dump_graph_folder = folder


def graph_optimization_pass(run_after: Iterable[str]) -> Callable:
    """
    The contract of graph optimization pass. All the passes should be wrapped with
    this decorator. `run_after` is used to denote which optimizer passes must be
    run before the wrapped optimizer pass. Optimizer pass developers are required
    to add this field accordingly and users need to follow the restrictions to
    avoid the assert.

    Current design has one limitation: users can only apply the optimizations once.
    In some cases, we may need to run multiple the same optimization multiple time,
    e.g., optimization passes -> profiling the result -> apply optimization passes
    with the profiling result again. Will address this limitation in the future.

    Args:
        run_after (Iterable[str]): the list of string to the names of passes which
            must be run before this optimization pass.
    """

    def inner(func: Callable) -> Callable:
        for name in run_after:
            _run_before_sets[name].add(func.__name__)

        @wraps(func)
        def pass_wrapper(
            gm: Union[fx.GraphModule, IterGraphModule], *args: Any, **kwargs: Any
        ) -> None:
            assert isinstance(
                gm, (fx.GraphModule, IterGraphModule)
            ), "The first argument of the pass must be either fx.GraphModule or IterGraphModule"
            assert (
                func.__name__ not in _optimized_func
            ), f"Cannot apply {func.__name__} twice."
            invalid_passes = _run_before_sets[func.__name__].intersection(
                _optimized_func
            )
            assert (
                not invalid_passes
            ), f"{invalid_passes} must run after {func.__name__}"

            func(gm, *args, **kwargs)
            gm.graph.lint()
            gm.graph.eliminate_dead_code()
            gm.recompile()
            _optimized_func.add(func.__name__)

            prefix = f"after_{func.__name__}"
            if _dump_graph_folder:
                if isinstance(gm, IterGraphModule):
                    dump_graphs_to_files(
                        {
                            f"{prefix}_setup_gm": gm.setup_gm,
                            f"{prefix}_main_gm": gm.main_gm,
                            f"{prefix}_cleanup_gm": gm.cleanup_gm,
                        },
                        _dump_graph_folder,
                    )
                else:
                    dump_graphs_to_files({prefix: gm}, _dump_graph_folder)

        return pass_wrapper

    return inner


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
    tensor_meta = inputs[0].meta.get("tensor_meta", None)
    return CommBlock(
        shape=tensor_meta.shape if tensor_meta else None,
        node_list=node_list,
        wait_nodes=wait_nodes,
        comm_node=comm_node,
        inputs=inputs,
        outputs=outputs,
    )


def get_all_comm_blocks(
    gm: IterGraphModule, comm_ops: Tuple[str, ...]
) -> List[CommBlock]:
    return [
        get_comm_block(node)
        for node in gm.graph.nodes
        if node.name.startswith(comm_ops)
    ]


def _create_meta_val(
    fake_tensor_mode: FakeTensorMode,
    val: FakeTensor,
) -> FakeTensor:
    # TODO: fix the memory_format
    return FakeTensor(
        fake_tensor_mode,
        torch.empty(
            val.shape,
            dtype=val.dtype,
            device="meta",
            requires_grad=val.requires_grad,
        ),
        val.device,
    )


def _create_meta_tensor_meta(
    fake_tensor_mode: FakeTensorMode,
    val: FakeTensor,
) -> TensorMetadata:
    return TensorMetadata(
        shape=val.shape,
        dtype=val.dtype,
        requires_grad=val.requires_grad,
        stride=val.stride,  # type: ignore[arg-type]
        # TODO: fix these value
        memory_format=None,
        is_quantized=False,
        qparams={},
    )


def _call_function(
    gm: IterGraphModule,
    fake_tensor_mode: FakeTensorMode,
    meta_val: Optional[FakeTensor],
    function: Any,
    *args: Any,
    **kwargs: Any,
) -> fx.Node:
    node = gm.graph.call_function(function, args, kwargs)

    if meta_val is None:
        flat_args, spec = tree_flatten((args, kwargs))
        new_flat_args = []
        memory_format = None
        for arg in flat_args:
            if not isinstance(arg, fx.Node):
                new_flat_args.append(arg)
                continue
            val = arg.meta["val"]
            new_flat_args.append(_create_meta_val(fake_tensor_mode, val))

        fake_args, fake_kwargs = tree_unflatten(new_flat_args, spec)
        new_meta_val = function(*fake_args, **fake_kwargs)
    else:
        new_meta_val = meta_val
    node.meta["val"] = new_meta_val
    node.meta["tensor_meta"] = _create_meta_tensor_meta(fake_tensor_mode, new_meta_val)
    return node


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
                [cast(torch.Size, cb.shape).numel() for cb in comm_blocks],
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
        shape=tensor_meta.shape,  # type: [union-attr]
        node_list=[fused_comm_node, fused_wait_node],
        wait_nodes=[fused_wait_node],
        comm_node=fused_comm_node,
        inputs=[cat_node],
        outputs=set([fused_wait_node]),
    )

    _scatter_wait_result(gm, fused_comm_block, comm_blocks, node_indices)

    return fused_comm_block


@graph_optimization_pass(run_after=tuple())
def comm_fusion_with_concat(
    gm: IterGraphModule,
    bucket_size_mb: int,
) -> None:
    """
    Run fuse communication with concat.
    This implementation uses concat to concat the bucketed gradients.
    """
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


@graph_optimization_pass(run_after=("comm_fusion_with_concat",))
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
