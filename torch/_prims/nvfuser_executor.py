from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from warnings import warn

import torch
import torch.overrides
from torch._prims_common import getnvFuserDtype, Number, number_type

from torch.fx import GraphModule
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.utils._pytree import tree_any, tree_flatten, tree_map, tree_unflatten

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
            return nvFuserScalarTemplate(getnvFuserDtype(number_type(arg)))
        else:
            return arg

    return tree_map(to_nvfuser, args)


def _is_node_in_output(gm, candidate_node):
    for node in gm.graph.nodes:
        if node.op == "output":
            return tree_any(lambda x: x == candidate_node, node.args[0])


def _is_node_in_input(gm, candidate_node):
    return tree_any(
        lambda x: x == candidate_node,
        [node for node in gm.graph.nodes if node.op == "placeholder"],
    )


# Current implementation of in-place copy_to in nvFuser implicitly adds a new
# output to the fusion. We don't want to expose this tensor to the outer world,
# so we count the number of outputs to be dropped. We don't need to drop the
# outputs that are actually included in the output of the graph.
def _count_outputs_to_drop(gm):
    return sum(
        1
        for node in gm.graph.nodes
        if node.op == "call_function"
        and node.target
        in [torch.ops.nvprims.copy_to, torch.ops.nvprims.copy_to.default]
        and (
            not _is_node_in_output(gm, node.args[1])
            or _is_node_in_input(gm, node.args[1])
        )
    )


# MyPy bug: https://github.com/python/mypy/issues/5107
@lru_cache(maxsize=1024)  # type: ignore[arg-type]
def make_nvfuser_fusion(gm: GraphModule, *nv_args_templates):
    # PROTOTYPE nvfuser executor
    # Everything in the graph must support nvfuser
    for node in gm.graph.nodes:
        if node.op == "call_function" and "getitem" in node.name:
            continue
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
                    return self.call_function(node.target, args, kwargs)

                # copy_to requires special handling of the output tensors
                elif node.target in [
                    torch.ops.nvprims.copy_to,
                    torch.ops.nvprims.copy_to.default,
                ]:
                    assert len(node.args) == 2
                    source_node = node.args[1]
                    args, kwargs = self.fetch_args_kwargs_from_env(node)
                    source = args[1]
                    # If source is a fusion input, we need to place an operation
                    # before the copy_to so that the copy is actually performed
                    if _is_node_in_input(gm, source_node):
                        source = fd.ops.set(source)
                    result = self.call_function(node.target, [args[0], source], kwargs)
                    # Check whether the source tensor is an output tensor
                    # If it is, we need to drop it now from the fusion output
                    # (it was implicitly added by the copy_to op)
                    # it will be added back later using correct expected order
                    if _is_node_in_output(gm, source_node) and not _is_node_in_input(
                        gm, source_node
                    ):
                        fd.remove_output(source)
                    return result

                return super().run_node(node)

            def call_function(self, target, args, kwargs):
                # This handles tuple unpacking
                if "getitem" in str(target):
                    assert isinstance(args[0], tuple)
                    return target(*args, **kwargs)
                args = tuple(map(_to_nvfuser_constant, args))
                target = target.impl_nvfuser
                args = (fd,) + args
                return target(*args, **kwargs)

        def templates_to_nvfuser_inputs(arg):
            if isinstance(arg, nvFuserTensorTemplate):
                x = fd.define_tensor(arg.size, arg.stride, arg.dtype)
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

    return fusion, unflatten_spec, _count_outputs_to_drop(gm)


def nvfuser_execute(gm: GraphModule, *args):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Attempting to use nvFuser trace executor but CUDA is not available!"
        )

    flat_args, _ = tree_flatten(args)

    # Construction of the fusion is expensive and cached based on the GraphModule
    # and symbolic nvFuser args.
    nv_template_args = to_nvfuser_template_args(flat_args)
    fusion, unflatten_spec, drop_output_count = make_nvfuser_fusion(gm, *nv_template_args)  # type: ignore[misc]

    # Inputs to fusion.execute correspond to the same template/symbolic inputs
    # marked with `define_tensor/scalar`
    concrete_fusion_inputs = tuple(
        arg for arg in flat_args if isinstance(arg, (torch.Tensor, Number))
    )

    return tree_unflatten(
        fusion.execute(concrete_fusion_inputs)[drop_output_count:],  # type: ignore[has-type]
        unflatten_spec,  # type: ignore[has-type]
    )


class NvfuserPrimOperatorSupport(torch.fx.passes.operator_support.OperatorSupport):
    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
        return (
            node.op == "call_function"
            and getattr(node.target, "impl_nvfuser", None) is not None
            or "getitem" in node.name  # getitem is a special case
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
    def __init__(self, gm):
        super().__init__()
        self.gm = gm

    def __call__(self, *args):
        return nvfuser_execute(self.gm, *args)


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

        # Replacing graph's fused submodules with a wrapper module with
        # __call__() method that calls nvfuser_execute.
        # This avoids the need to call the interpreter on the graph
        for node in partitioned_graph.graph.nodes:
            # TODO: use a better way to identify fused submodule
            if node.op == "call_module" and "fused_" in node.name:
                nvfuser_submodule = getattr(partitioned_graph, node.name)
                partitioned_graph.delete_submodule(node.target)
                gm.add_submodule(node.target, NvfuserGraphModule(nvfuser_submodule))

        return partitioned_graph, any_unsupported
    else:
        return gm, any_unsupported


def nvfuser_execute_partitioned(gm: GraphModule, *args):
    # When possible it's better to use nvfuser_execute directly
    # because it avoids PartitionedInterpreter's overhead
    gm, is_partitioned = maybe_partition_graph(gm)
    if is_partitioned:
        return gm(*args)
    else:
        return nvfuser_execute(gm, *args)
