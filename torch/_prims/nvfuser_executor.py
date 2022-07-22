from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from warnings import warn

import torch
import torch.overrides
from torch._prims_common import getnvFuserDtype, Number

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


# nvFuserTensorTemplate and nvFuserScalarTemplate are helper objects
# for cached construction of the nvFuser's Fusion
# TODO: change what is stored in the cache for nvFuser's Tensor objects
# https://github.com/pytorch/pytorch/issues/80551
@dataclass(frozen=True)
class nvFuserTensorTemplate:
    size: tuple
    stride: tuple
    dtype: DataType


@dataclass(frozen=True)
class nvFuserScalarTemplate:
    dtype: DataType


def to_nvfuser_template_args(args):
    def to_nvfuser(arg):
        if isinstance(arg, torch.Tensor):
            return nvFuserTensorTemplate(
                arg.size(), arg.stride(), getnvFuserDtype(arg.dtype)
            )
        elif isinstance(arg, Number):
            return nvFuserScalarTemplate(getnvFuserDtype(type(arg)))
        else:
            return arg

    return tree_map(to_nvfuser, args)


# MyPy bug: https://github.com/python/mypy/issues/5107
@lru_cache(maxsize=1024)  # type: ignore[arg-type]
def make_nvfuser_fusion(gm: GraphModule, *nv_args_templates):
    # PROTOTYPE nvfuser executor
    # Everything in the graph must support nvfuser
    for node in gm.graph.nodes:
        if (
            node.op == "call_function"
            and getattr(node.target, "impl_nvfuser", None) is None
        ):
            raise ValueError(
                "All call_function nodes in the graph must support nvfuser. "
                f"Node {node} with target {node.target} does not support nvfuser"
            )

    fusion = Fusion()
    with FusionDefinition(fusion) as fd:

        def _to_nvfuser_constant(arg):
            if isinstance(arg, Number):
                return fd.define_constant(arg)
            else:
                return arg

        class FusionInterpreter(torch.fx.Interpreter):
            def call_function(self, target, args, kwargs):
                args = tuple(map(_to_nvfuser_constant, args))
                target = target.impl_nvfuser
                args = (fd,) + args
                return target(*args, **kwargs)

        def templates_to_nvfuser_inputs(arg):
            if isinstance(arg, nvFuserTensorTemplate):
                x = fd.define_tensor(arg.size, arg.stride, arg.dtype)
                fd.add_input(x)
                return x
            elif isinstance(arg, nvFuserScalarTemplate):
                x = fd.define_scalar(arg.dtype)
                fd.add_input(x)
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


def nvfuser_execute(gm: GraphModule, *args):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Attempting to use nvFuser trace executor but CUDA is not available!"
        )

    flat_args, _ = tree_flatten(args)

    # Construction of the fusion is expensive and cached based on the GraphModule
    # and symbolic nvFuser args.
    nv_template_args = to_nvfuser_template_args(flat_args)
    fusion, unflatten_spec = make_nvfuser_fusion(gm, *nv_template_args)  # type: ignore[misc]

    # Inputs to fusion.execute correspond to the same template/symbolic inputs marked with `fd.add_input`
    concrete_fusion_inputs = tuple(
        arg for arg in flat_args if isinstance(arg, (torch.Tensor, Number))
    )

    return tree_unflatten(
        fusion.execute(concrete_fusion_inputs),  # type: ignore[has-type]
        unflatten_spec,  # type: ignore[has-type]
    )


class NvfuserPrimOperatorSupport(torch.fx.passes.operator_support.OperatorSupport):
    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
        return (
            node.op == "call_function"
            and getattr(node.target, "impl_nvfuser", None) is not None
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


# MyPy bug: https://github.com/python/mypy/issues/5107
@lru_cache()  # type: ignore[arg-type]
def maybe_partition_graph(gm: GraphModule):
    supported_ops = NvfuserPrimOperatorSupport()
    call_function_nodes = filter(lambda n: n.op == "call_function", gm.graph.nodes)
    # the graph is partitioned only if at least one node is not supported by nvFuser
    any_unsupported = any(
        not supported_ops.is_node_supported(None, node) for node in call_function_nodes
    )
    if any_unsupported:
        # CapabilityBasedPartitioner modifies the graph in-place so we need to make a copy of the graph
        gm = deepcopy(gm)
        partitioner = CapabilityBasedPartitioner(
            gm, supported_ops, allows_single_node_partition=True
        )
        partitions = partitioner.propose_partitions()
        if len(partitions) == 0:
            warn(
                "No partition found for the graph. "
                + "This is likely because the graph is not supported by nvFuser. "
                + "Please use the eager ATen mode to execute the graph.",
                category=RuntimeWarning,
            )
        partitioned_graph = partitioner.fuse_partitions(partitions)
        return partitioned_graph, any_unsupported
    else:
        return gm, any_unsupported


def nvfuser_execute_partitioned(gm: GraphModule, *args):
    # When possible it's better to use nvfuser_execute directly
    # because it avoids PartitionedInterpreter's overhead
    gm, is_partitioned = maybe_partition_graph(gm)
    if is_partitioned:
        return PartitionedInterpreter(gm).run(*args)
    else:
        return nvfuser_execute(gm, *args)
