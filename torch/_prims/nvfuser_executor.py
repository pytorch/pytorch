import operator
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from warnings import warn

import torch
import torch.overrides
from torch._prims_common import (
    _torch_dtype_to_nvfuser_dtype_map,
    getnvFuserDtype,
    Number,
    number_type,
)

from torch.fx import GraphModule
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

if torch.cuda.is_available():
    from torch._C._nvfuser import (  # type: ignore[import]
        DataType,
        Fusion,
        FusionDefinition,
    )
else:
    DataType = None

DEFAULT_NVFUSER_PYTHON_CONFIG = MappingProxyType(
    {
        "use_python_fusion_cache": True,
        "allow_single_op_fusion": False,
    }
)

# nvFuserTensorTemplate and nvFuserScalarTemplate are helper objects
# for cached construction of the nvFuser's Fusion
# TODO: change what is stored in the cache for nvFuser's Tensor objects
# https://github.com/pytorch/pytorch/issues/80551
@dataclass(frozen=True)
class nvFuserTensorTemplate:
    symbolic_shape: tuple
    contiguity: tuple
    dtype: DataType
    is_cpu: bool


@dataclass(frozen=True)
class nvFuserScalarTemplate:
    dtype: DataType


@lru_cache(maxsize=2048)
def compute_symbolic_shape(shape):
    """Computes the symbolic shape of a tensor.
    nvFuser specializes on size-1 dimensions as broadcasted dimensions.
    -1 is used to represent any size."""
    return tuple(1 if s == 1 else -1 for s in shape)


@lru_cache(maxsize=2048)
def compute_contiguity(shape, strides):
    """Computes the contiguity information to simplify internal indexing.
    Contiguous dimensions are represented by True, strided dimensions
    are represented by False.
    """
    return torch._C._nvfuser.compute_contiguity(shape, strides)


def to_nvfuser_template_args(args):
    def to_nvfuser(arg):
        if isinstance(arg, torch.Tensor):
            return nvFuserTensorTemplate(
                compute_symbolic_shape(arg.size()),
                compute_contiguity(arg.size(), arg.stride()),
                getnvFuserDtype(arg.dtype),
                arg.is_cpu,  # type: ignore[attr-defined]
            )
        elif isinstance(arg, Number):
            return nvFuserScalarTemplate(getnvFuserDtype(number_type(arg)))
        else:
            return arg

    return tree_map(to_nvfuser, args)


def _any_get_attr_used(call_function_nodes):
    return any(
        filter(
            # bug in mypy https://github.com/python/mypy/issues/12682
            lambda n: any(  # type: ignore[arg-type]
                a.op == "get_attr" for a in n.args if isinstance(a, torch.fx.Node)  # type: ignore[attr-defined]
            ),
            call_function_nodes,
        )
    )


# MyPy bug: https://github.com/python/mypy/issues/5107
@lru_cache(maxsize=1024)  # type: ignore[arg-type]
def make_nvfuser_fusion(gm: GraphModule, *nv_args_templates):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Attempting to use nvFuser trace executor but CUDA is not available!"
        )

    # Everything in the graph must support nvfuser
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target == operator.getitem:
            continue
        if (
            node.op == "call_function"
            and getattr(node.target, "impl_nvfuser", None) is None
        ):
            raise ValueError(
                "All call_function nodes in the graph must support nvfuser. "
                f"Node {node} with target {node.target} does not support nvfuser"
            )

    graph_input_nodes = list(filter(lambda n: n.op == "placeholder", gm.graph.nodes))
    call_function_nodes = list(
        filter(lambda n: n.op == "call_function", gm.graph.nodes)
    )
    assert len(graph_input_nodes) == len(
        nv_args_templates
    ), "Number of placeholder nodes in the graph must match number of args"
    assert len(nv_args_templates) > 0, "There must be at least one argument"
    assert (
        len(call_function_nodes) > 0
    ), "Graph must contain at least one call_function node"
    assert not _any_get_attr_used(
        call_function_nodes
    ), "Constant tensors that are saved in the graph and used as arguments are not supported yet"

    fusion = Fusion()
    with FusionDefinition(fusion) as fd:

        def _to_nvfuser_constant(arg):
            if isinstance(arg, Number):
                return fd.define_constant(arg)
            else:
                return arg

        class FusionInterpreter(torch.fx.Interpreter):
            def run_node(self, node):
                # Squeeze requires original shape of args[0]
                if node.target in [
                    torch.ops.nvprims.squeeze,
                    torch.ops.nvprims.squeeze.default,
                ]:
                    original_shape = list(node.args[0].meta["tensor_meta"].shape)
                    assert len(node.args) == 2
                    args, kwargs = self.fetch_args_kwargs_from_env(node)
                    args = [args[0], original_shape, args[1]]
                    return self.call_function(node.target, args, node.kwargs)

                if node.target in [
                    torch.ops.nvprims.native_batch_norm,
                    torch.ops.nvprims.native_batch_norm.default,
                ]:
                    args, kwargs = self.fetch_args_kwargs_from_env(node)
                    assert len(args) == 8
                    training = args[5]
                    args6_end = tuple(map(_to_nvfuser_constant, args[6:]))
                    args = args[:5] + (training,) + args6_end
                    return node.target.impl_nvfuser(fd, *args, **kwargs)

                return super().run_node(node)

            def call_function(self, target, args, kwargs):
                # This handles tuple unpacking
                if target == operator.getitem:
                    assert isinstance(args[0], tuple)
                    return target(*args, **kwargs)
                args = tuple(map(_to_nvfuser_constant, args))
                target = target.impl_nvfuser
                args = (fd,) + args
                return target(*args, **kwargs)

        def templates_to_nvfuser_inputs(arg):
            if isinstance(arg, nvFuserTensorTemplate):
                x = fd.define_tensor(
                    arg.symbolic_shape, arg.contiguity, arg.dtype, arg.is_cpu
                )
                return x
            elif isinstance(arg, nvFuserScalarTemplate):
                x = fd.define_scalar(arg.dtype)
                return x
            else:
                return arg

        # Transforms graph to call nvfuser lowerings
        nv_args = tuple(map(templates_to_nvfuser_inputs, nv_args_templates))
        out = FusionInterpreter(gm).run(*nv_args)
        flat_out, unflatten_spec = tree_flatten(out)
        for o in flat_out:
            fd.add_output(o)

    return fusion, unflatten_spec


def nvfuser_execute(gm: GraphModule, *args, executor_parameters=None):
    executor_parameters = executor_parameters or DEFAULT_NVFUSER_PYTHON_CONFIG
    flat_args, _ = tree_flatten(args)

    # check for cuda only fusion
    if any(isinstance(arg, torch.Tensor) and arg.is_cuda for arg in flat_args) and all(  # type: ignore[attr-defined]
        (
            not isinstance(arg, torch.Tensor)
            or (arg.is_cpu and arg.ndim == 0)  # type: ignore[attr-defined]
            or arg.is_cuda  # type: ignore[attr-defined]
        )
        for arg in flat_args
    ):

        # Construction of the fusion is expensive and cached based on the GraphModule
        # and symbolic nvFuser args.
        nv_template_args = to_nvfuser_template_args(flat_args)
        use_cache = executor_parameters.get(
            "use_python_fusion_cache",
            DEFAULT_NVFUSER_PYTHON_CONFIG["use_python_fusion_cache"],
        )
        if use_cache:
            fusion, unflatten_spec = make_nvfuser_fusion(gm, *nv_template_args)  # type: ignore[misc]
        else:
            fusion, unflatten_spec = make_nvfuser_fusion.__wrapped__(gm, *nv_template_args)  # type: ignore[misc]

        # Inputs to fusion.execute correspond to the same template/symbolic inputs
        # marked with `define_tensor/scalar`
        concrete_fusion_inputs = tuple(
            arg for arg in flat_args if isinstance(arg, (torch.Tensor, Number))
        )

        return tree_unflatten(
            fusion.execute(concrete_fusion_inputs),  # type: ignore[has-type]
            unflatten_spec,  # type: ignore[has-type]
        )
    else:
        warn(
            "nvfuser_executor is executed with non-cuda args, fallback to aten executor"
        )
        return gm.forward(*args)


class NvfuserPrimOperatorSupport(torch.fx.passes.operator_support.OperatorSupport):
    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
        # special case to stop lowering to nvprim when converting to an unsupported type
        if (
            node.op == "call_function"
            and node.target == torch.ops.nvprims.convert_element_type.default
        ):
            return (
                _torch_dtype_to_nvfuser_dtype_map.get(node.args[1]) is not None
                and _torch_dtype_to_nvfuser_dtype_map.get(
                    node.args[0].meta["tensor_meta"].dtype  # type: ignore[union-attr]
                )
                is not None
            )
        return node.op == "call_function" and (
            getattr(node.target, "impl_nvfuser", None) is not None
            or node.target == operator.getitem
        )


class PartitionedInterpreter(torch.fx.Interpreter):
    def call_module(self, target, args, kwargs):
        assert isinstance(target, str)
        assert len(kwargs) == 0
        submod = self.fetch_attr(target)
        # CapabilityBasedPartitioner hardcodes the name of the subgraphs with supported_ops as "fused_" + subgraph id
        if target.startswith("fused_"):
            return nvfuser_execute(submod, *args)
        else:
            return super().call_module(target, args, kwargs)


class NvfuserGraphModule(torch.nn.Module):
    def __init__(self, gm, use_python_fusion_cache):
        super().__init__()
        self.gm = gm
        self.executor_parameters = {"use_python_fusion_cache": use_python_fusion_cache}

    def __call__(self, *args):
        return nvfuser_execute(
            self.gm, *args, executor_parameters=self.executor_parameters
        )


# A set of operators that are supported by nvFuser
# but should not form a fusion group solely on their own
_non_compute_ops = [
    "torch.ops." + str(getattr(torch.ops.nvprims, prim).default)
    for prim in dir(torch.ops.nvprims)
    if isinstance(getattr(torch.ops.nvprims, prim), torch._ops.OpOverloadPacket)
    and getattr(torch.ops.nvprims, prim).return_type
    == torch._prims_common.RETURN_TYPE.VIEW
]

_allowed_single_node_partition_ops = [
    "torch.ops.nvprims.native_batch_norm.default",
    "torch.ops.nvprims.var_mean.default",
    "torch.ops.nvprims.var_mean.main",
]


def _remove_empty_like_fill(gm: GraphModule):
    # Remove empty_like + fill nodes that prevent lowering to nvprims
    # This is a workaround for nonoptimal traces of C++ code `(1 - tensor)`
    # https://github.com/pytorch/pytorch/issues/86612

    # Here when we see a `sub` node, we check if the first input is a result of
    # filling a tensor with a scalar
    # If so, we replace the first argument of the `sub` node with a scalar
    for node in gm.graph.nodes:
        if node.op == "call_function":
            if node.target == torch.ops.nvprims.sub.default:
                # check if the first argument is a fill
                if (
                    isinstance(node.args[0], torch.fx.Node)
                    and node.args[0].op == "call_function"
                    and node.args[0].target == torch.ops.aten.fill.Scalar
                ):
                    # Replace the first argument with the second argument of fill
                    # aten.fill.Scalar(tensor, scalar)
                    fill_node = node.args[0]
                    scalar = fill_node.args[1]
                    node.args = (scalar, *node.args[1:])
    gm.graph.eliminate_dead_code()
    gm.recompile()
    return gm


# MyPy bug: https://github.com/python/mypy/issues/5107
@lru_cache(maxsize=1024)  # type: ignore[arg-type]
def maybe_partition_graph(
    gm: GraphModule, allow_single_op_fusion: bool, use_python_fusion_cache: bool
):
    gm = _remove_empty_like_fill(gm)
    supported_ops = NvfuserPrimOperatorSupport()
    call_function_nodes = list(
        filter(lambda n: n.op == "call_function", gm.graph.nodes)
    )
    # the graph is partitioned only if at least one node is not supported by nvFuser
    any_unsupported = any(
        not supported_ops.is_node_supported(None, node) for node in call_function_nodes
    )
    any_unsupported |= len(call_function_nodes) == 0

    # When there are constant tensors in the graph, we can't partition it
    # because deepcopy fails. Here we just return the original graph to be
    # executed by eager mode
    # https://github.com/pytorch/pytorch/issues/84415
    if (
        _any_get_attr_used(call_function_nodes)
        or len(list(filter(lambda n: n.op == "placeholder", gm.graph.nodes))) == 0
    ):
        return gm, True

    if any_unsupported:
        # CapabilityBasedPartitioner modifies the graph in-place so we need to make a copy of the graph
        gm = deepcopy(gm)
        partitioner = CapabilityBasedPartitioner(
            gm,
            supported_ops,
            allows_single_node_partition=allow_single_op_fusion,
            non_compute_ops=_non_compute_ops,
            allowed_single_node_partition_ops=_allowed_single_node_partition_ops,
        )
        partitions = partitioner.propose_partitions()
        partitioner.remove_bookend_non_compute_ops(partitions)
        if len(partitions) == 0:
            warn(
                "No partition found for the graph. "
                + "This is likely because the graph is not supported by nvFuser. "
                + "Please use the eager ATen mode to execute the graph.",
                category=RuntimeWarning,
            )
        partitioned_graph = partitioner.fuse_partitions(partitions)

        # Replacing graph's fused submodules with a wrapper module with
        # __call__() method that calls nvfuser_execute.
        # This avoids the need to call the interpreter on the graph
        for node in partitioned_graph.graph.nodes:
            # TODO: use a better way to identify fused submodule
            if node.op == "call_module" and "fused_" in node.name:
                nvfuser_submodule = getattr(partitioned_graph, node.name)
                partitioned_graph.delete_submodule(node.target)
                gm.add_submodule(
                    node.target,
                    NvfuserGraphModule(nvfuser_submodule, use_python_fusion_cache),
                )

        # Go through the graph and replace all the nodes that were converted to
        # nvprims but won't be sent to nvFuser with a call to PyTorch's eager
        # mode. This is necessary because torch.ops.* have higher overhead than
        # calling the eager mode directly.
        for node in partitioned_graph.graph.nodes:
            if node.op == "call_function" and str(node.target).startswith("nvprims."):
                if getattr(node.target, "impl_aten", None) is not None:
                    node.target = node.target.impl_aten
        partitioned_graph.graph.eliminate_dead_code()
        partitioned_graph.recompile()
        return partitioned_graph, any_unsupported
    else:
        return gm, any_unsupported


def nvfuser_execute_partitioned(gm: GraphModule, *args, executor_parameters=None):
    executor_parameters = executor_parameters or DEFAULT_NVFUSER_PYTHON_CONFIG
    # maybe_partition_graph function is cached so we can't use non-hashable arguments
    allow_single_op_fusion = executor_parameters.get(
        "allow_single_op_fusion",
        DEFAULT_NVFUSER_PYTHON_CONFIG["allow_single_op_fusion"],
    )
    use_python_fusion_cache = executor_parameters.get(
        "use_python_fusion_cache",
        DEFAULT_NVFUSER_PYTHON_CONFIG["use_python_fusion_cache"],
    )
    # When possible it's better to use nvfuser_execute directly
    # because it avoids GraphModule's overhead
    gm, is_partitioned = maybe_partition_graph(
        gm,
        allow_single_op_fusion=allow_single_op_fusion,
        use_python_fusion_cache=use_python_fusion_cache,
    )
    if is_partitioned:
        return gm(*args)
    else:
        return nvfuser_execute(gm, *args, executor_parameters=executor_parameters)
