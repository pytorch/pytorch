# Owner(s): ["oncall: distributed"]
import collections
import logging
import tempfile
import time
from dataclasses import dataclass
from functools import wraps
from typing import (
    Any,
    Callable,
    DefaultDict,
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
from torch.distributed._spmd.graph_utils import dump_graphs_to_files
from torch.distributed._spmd.iter_graph_module import IterGraphModule
from torch.fx.passes.shape_prop import TensorMetadata
from torch.utils._pytree import tree_flatten, tree_unflatten

logger: logging.Logger = logging.getLogger("graph_transformations")
fake_tensor_mode = FakeTensorMode()

_optimized_func: Set[str] = set()
# The key is the target pass and the value is the prerequisites of the pass.
_prerequisite_sets: DefaultDict[str, Set[str]] = collections.defaultdict(set)
# The key is the target pass and the value is the passes that must applied before
# the key.
_apply_before_sets: DefaultDict[str, Set[str]] = collections.defaultdict(set)
_dump_graph_folder: str = ""


def enable_graph_optimization_dump(folder: str = ""):
    global _dump_graph_folder
    if not folder:
        folder = tempfile.mkdtemp()
    _dump_graph_folder = folder


# TODO(@fegin): Support multiple runs of graph optimization
# TODO(@fegin): With this design, circular imports will happen when a pass
# developer accidentally create a pass dependency cycle. As a result, we need to
# break this file into a finer granularity to avoid incorrect circular import.
def graph_optimization_pass(
    prerequisites: Iterable[Callable],
    apply_after: Iterable[Callable],
) -> Callable:
    """
    The contract of graph optimization pass. All the passes should be wrapped
    with this decorator.

    `prerequisites` is used to annotate the prerequisite passes of the this pass.
    `apply_after` means that this wrapped pass must be applied after the passes
    in `apply_after`. The difference between `prerequisites` and `apply_after`
    is that all the passes in `prerequisites` must be applied to the graph and
    must be applifed before the wrapped pass while the passes `apply_after` are
    optional. But if a pass in `apply_after` is applied to the graph, it has to
    be done before the wrapped pass.
    Optimizer pass developers are required to add these fields accordingly and
    users need to follow the restrictions to avoid the assert.

    Current design has one limitation: users can only apply the optimizations
    once.  In some cases, we may need to run multiple the same optimization
    multiple time, e.g., optimization passes -> profiling the result -> apply
    optimization passes with the profiling result again. This limitation will be
    addressed limitation in the future.

    Args:
        prerequisites (Iterable[Callable]): the list of string to the names of
            passes which are the prerequisites of this pass.
        apply_after (Iterable[Callable]): the list of string to the names of
            passes that can not be applied after the wrapped pass.
    """

    def inner(func: Callable) -> Callable:
        def make_key(func: Callable) -> str:
            return f"{func.__module__}.{func.__name__}"

        func_key = make_key(func)
        _prerequisite_sets[func_key] = {make_key(f) for f in prerequisites}
        for apply_after_pass in apply_after:
            _apply_before_sets[make_key(apply_after_pass)].add(func_key)

        @wraps(func)
        def pass_wrapper(
            gm: Union[fx.GraphModule, IterGraphModule], *args: Any, **kwargs: Any
        ) -> None:
            begin = time.time()
            assert isinstance(gm, (fx.GraphModule, IterGraphModule)), (
                "The first argument of the pass must be either "
                "fx.GraphModule or IterGraphModule."
            )
            assert func_key not in _optimized_func, f"Cannot apply {func_key} twice."
            invalid_passes = _apply_before_sets[func_key].intersection(_optimized_func)
            assert (
                not invalid_passes
            ), f"{invalid_passes} must be applied after {func_key}."
            assert _prerequisite_sets[func_key].issubset(_optimized_func), (
                f"{_prerequisite_sets[func_key] - _optimized_func} are the "
                f"prerequisites of {func_key} but are not applified. "
                f"Applied passes are {_optimized_func}."
            )

            func(gm, *args, **kwargs)
            gm.graph.lint()
            gm.graph.eliminate_dead_code()
            gm.recompile()
            _optimized_func.add(func_key)

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

            logger.info("Spent %f seconds applying %s", time.time() - begin, func_key)

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
    return CommBlock(
        # TODO: support symbolic shapes
        shape=torch.Size(int(s) for s in tensor_meta.shape) if tensor_meta else None,
        node_list=node_list,
        wait_nodes=wait_nodes,
        comm_node=comm_node,
        inputs=input_nodes,
        outputs=outputs,
    )


def get_all_comm_blocks(
    gm: IterGraphModule, comm_ops: Union[Tuple[str, ...], str]
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
