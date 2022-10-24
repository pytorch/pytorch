from typing import Dict

import torch
from torch.nn import Module
from torch._ops import OpOverload

from torch.fx import GraphModule
from torch.fx.node import Node, _get_qualified_name
from torch.fx.passes.operator_support import OperatorSupport
from torch.fx.passes.tools_common import CALLABLE_NODE_OPS
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch._prims.executor import execute
from torch.fx.experimental.proxy_tensor import DecompositionInterpreter
from torch._decomp import decomposition_table

import typing as t

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

def aten_to_dtype(self, dtype: torch.dtype, **kwargs):
    if len(kwargs) > 0 or not dtype:
        raise RuntimeError("No support for other to.dtype() formats other than to.dtype(self, dtype)")
    return torch._prims.convert_element_type(self, dtype)

# decomposition_table currently contains both aten2aten and aten2prim decomposition
# this is a hack to separate them, as we only need aten2prim decomposition for nvfuser-supported aten graph lowering
aten2aten_decomp = {}
aten2prim_decomp = {}

for op, decomp_fn in decomposition_table.items():
    if "torch._refs" in decomp_fn.__module__:
        aten2prim_decomp[op] = decomp_fn
    else:
        aten2aten_decomp[op] = decomp_fn

aten2aten_decomp_skips = {
    "aten.native_layer_norm_backward.default",
    "aten.embedding_dense_backward.default",   # This is hurting nvfuser's perf
    "aten.addmm.default"
}

for op, decomp_fn in decomposition_table.items():
    if "torch._refs" in decomp_fn.__module__:
        aten2prim_decomp[op] = decomp_fn
    else:
        if str(op) not in aten2aten_decomp_skips:
            aten2aten_decomp[op] = decomp_fn


aten2prim_decomp[torch.ops.aten.to.dtype] = aten_to_dtype


class NvFuserOperatorSupport(OperatorSupport):
    """
    Operator support for nvFuser backend.

    Currently, partitioning is based on FX ATen graph. The fused subgraph will latter be decomposed into prims.
    To determine if an ATen ops is supported by nvFuser, we shall check the prim ops used in its ref decomposition.
    Only if all the prim ops in the ref has a nvfuser_impl, we say this Aten op is suppported by nvFuser.

    Note: When adding a rule, please add it to the corresponding section and follow the
    alphabetical order.
    """

    def __init__(self):

        # TODO: current list copied from torch/csrc/jit/codegen/cuda/parser.cpp is incorrect,
        # as that file is solely for TorchScript and doesn't represent the actual status
        # whether operation would be runnable by primTorch+nvFuser.
        # We will iterate on this list to reflect the the reality.
        support_dict = {
            # ===============================================================
            # call_function aten
            # ===============================================================
            # Following supported aten ops is copied from torch/csrc/jit/codegen/cuda/parser.cpp
            # TODO: might need to update according to supported input types
            "torch.ops.aten.add": None,
            "torch.ops.aten.sub": None,
            # "torch.ops.aten.rsub": None,    # rsub decomp is supported at aten2aten level
            "torch.ops.aten.div": None,
            "torch.ops.aten.atan2": None,
            "torch.ops.aten.mul": None,
            "torch.ops.aten.max": None,
            "torch.ops.aten.min": None,
            "torch.ops.aten.pow": None,
            "torch.ops.aten.remainder": None,
            "torch.ops.aten.fmod": None,
            "torch.ops.aten.bitwise_and": None,
            "torch.ops.aten.__and__": None,
            "torch.ops.aten.bitwise_or": None,
            "torch.ops.aten.__or__": None,
            "torch.ops.aten.bitwise_xor": None,
            "torch.ops.aten.__xor__": None,
            "torch.ops.aten.bitwise_left_shift": None,
            "torch.ops.aten.__lshift__": None,
            "torch.ops.aten.bitwise_right_shift": None,
            "torch.ops.aten.__rshift__": None,
            "torch.ops.aten.eq": None,
            "torch.ops.aten.ne": None,
            "torch.ops.aten.ge": None,
            "torch.ops.aten.gt": None,
            "torch.ops.aten.le": None,
            "torch.ops.aten.lt": None,
            "torch.ops.aten.abs": None,
            "torch.ops.aten.bitwise_not": None,
            "torch.ops.aten.ceil": None,
            "torch.ops.aten.floor": None,
            "torch.ops.aten.frac": None,
            "torch.ops.aten.neg": None,
            "torch.ops.aten.relu": None,
            "torch.ops.aten.round": None,
            "torch.ops.aten.silu": None,
            "torch.ops.aten.trunc": None,
            "torch.ops.aten.log": None,
            "torch.ops.aten.log10": None,
            "torch.ops.aten.log1p": None,
            "torch.ops.aten.log2": None,
            "torch.ops.aten.lgamma": None,
            "torch.ops.aten.exp": None,
            "torch.ops.aten.expm1": None,
            "torch.ops.aten.erf": None,
            "torch.ops.aten.erfc": None,
            "torch.ops.aten.cos": None,
            "torch.ops.aten.acos": None,
            "torch.ops.aten.cosh": None,
            "torch.ops.aten.sin": None,
            "torch.ops.aten.asin": None,
            "torch.ops.aten.sinh": None,
            "torch.ops.aten.tan": None,
            "torch.ops.aten.atan": None,
            "torch.ops.aten.tanh": None,
            "torch.ops.aten.atanh": None,
            "torch.ops.aten.sqrt": None,
            "torch.ops.aten.rsqrt": None,
            "torch.ops.aten.reciprocal": None,
            "torch.ops.aten.sigmoid": None,
            "torch.ops.aten.isfinite": None,
            "torch.ops.aten.isinf": None,
            "torch.ops.aten.isnan": None,
            "torch.ops.aten.isneginf": None,
            "torch.ops.aten.isposinf": None,
            "torch.ops.aten.isreal": None,
            # "torch.ops.aten.rand_like": None,  # causing Node empty_like_default does not support nvfuser
            "torch.ops.aten.softplus": None,
            "torch.ops.aten.threshold": None,
            # relying on aten->aten->prim decomp, aten2aten is using unsupported aten.new_zero op
            # "torch.ops.aten.threshold_backward": None,
            "torch.ops.aten.clamp": None,
            # "torch.ops.aten.clone": None,
            # Failing with where(): incompatible function arguments: \
            # [<torch._C._nvfuser.TensorView, tensor, <torch._C._nvfuser.TensorView]
            # failing with BERT_pytorch_forward_0, which has aten.where.ScalarSelf in the decomps
            # "torch.ops.aten.where": None,
            # However, aten.where.self overload is fully supported
            "torch.ops.aten.where.self": None,
            "torch.ops.aten.lerp": None,
            "torch.ops.aten.addcmul": None,
            # "torch.ops.aten.native_dropout": None,    # missing refs for aten.rank_like
            "torch.ops.aten.dropout": None,
            # "torch.ops.aten.native_dropout_backward": None,   # missing refs for aten.type_as
            "torch.ops.aten.instance_norm": None,
            "torch.ops.aten._batch_norm_impl_index": None,
            # "torch.ops.aten.native_batch_norm": None,     # missing refs for aten.var
            "torch.ops.aten.batch_norm": None,
            "torch.ops.aten.cudnn_batch_norm": None,
            "torch.ops.aten._batch_norm_impl_index_backward": None,
            # "torch.ops.aten.native_batch_norm_backward": None,    # should have been handled at aten2aten decomp
            "torch.ops.aten.native_layer_norm": None,
            "torch.ops.aten.layer_norm": None,
            # relying on aten->aten->prim decomp, aten2aten is using unsupported aten.div
            # "torch.ops.aten.native_layer_norm_backward": None,
            "torch.ops.aten.softmax.int": None,
            "torch.ops.aten.log_softmax.int": None,
            # relying on aten->aten->prim decomp, aten2aten is using unsupported aten.amax
            # "torch.ops.aten._softmax": None,
            "torch.ops.aten._log_softmax_backward_data": None,
            # "torch.ops.aten._softmax_backward_data": None,  # Node _softmax_backward_data_default does not support nvfuser
            # "torch.ops.aten.var.dim": None,       # missing refs
            "torch.ops.aten.std.dim": None,
            "torch.ops.aten.sum": None,
            # "torch.ops.aten.mean.dim": None,      # missing refs
            "torch.ops.aten._grad_sum_to_size": None,
            "torch.ops.aten.sum_to_size": None,
            "torch.ops.aten._autocast_to_reduced_precision": None,
            "torch.ops.aten._autocast_to_full_precision": None,
            # "torch.ops.aten.to.dtype": None,      # causing segfault
            # "torch.ops.aten.type_as": None,       # missing refs
            "torch.ops.aten.linear": None,
            "torch.ops.aten.gelu": None,
            # "torch.ops.aten.gelu_backward": None,       # gelu_backward is handled at aten2aten decomp
            # "torch.ops.aten.hardtanh": None,        # has functional ref, using unsupported aten.clamp
            "torch.ops.aten.leaky_relu": None,
            "torch.ops.aten.square": None,
            # relying on aten->aten->prim decomp, aten2aten is using unsupported aten.conj_physical
            "torch.ops.aten.tanh_backward": None,
            # "torch.ops.aten.amax": None,      # missing prim decomp
            # "torch.ops.aten.amin": None,      # missing prim decomp
            # "torch.ops.aten.reshape": None,
            # "torch.ops.aten.view": None,      # missing prim decomp
            "torch.ops.aten.flatten.using_ints": None,

            # ===============================================================
            # call_function builtins and operator
            # ===============================================================
            "getattr": None,
            "_operator.getitem": None,
        }

        super().__init__(support_dict)

    def is_node_supported(
        self, submodules: t.Mapping[str, Module], node: Node
    ) -> bool:

        # nvFuser FX subgraph should be purely functional
        if node.op not in CALLABLE_NODE_OPS:
            return False

        # ops in supported_dict doesn't have overload name
        # use overloadpacket's qualified_name for OpOverload
        if isinstance(node.target, OpOverload):
            target = _get_qualified_name(node.target.overloadpacket)
            if target in self._support_dict:
                return True

        return super().is_node_supported(submodules, node)


class NvFuserBackend:
    def __init__(self):
        self.supported_ops = NvFuserOperatorSupport()

        # TODO: this is a naive implementation of cache without proper guard
        self.partitioner_cache: Dict[GraphModule, GraphModule] = {}

        # TODO: this is a naive implementation of cache without proper guard, this will only work for identical inputs
        self.prim_decomp_cache: Dict[GraphModule, GraphModule] = {}

    def lower_to_prims_and_execute(self, graph_module: GraphModule, *args, **kwargs):
        # `graph_module` is an Aten-Fx graph
        # "lowering to prims" and "trace execution" are grouped into this function, as they are both input dependent

        if graph_module in self.prim_decomp_cache:
            logger.debug("prim_decomp_cache hit!")
            prim_module = self.prim_decomp_cache[graph_module]
        else:
            prim_graph = torch.fx.Graph()
            DecompositionInterpreter(graph_module, prim_graph, decomposition_table=aten2prim_decomp).run(*args, **kwargs)
            prim_module = torch.fx.GraphModule(graph_module, prim_graph)
            self.prim_decomp_cache[graph_module] = prim_module

            logger.debug("Lower to prims graph: ", prim_module.code)

        # invokes trace executor for running the prim graph
        return execute(prim_module, *args, executor="nvfuser")

    def compile(self, graph_module: GraphModule) -> GraphModule:
        # entry function for nvFuser backend
        logger.debug("Compiling graph_module: ", graph_module.code)

        # FX graph based partitioning based on nvfuser supported ops
        if graph_module in self.partitioner_cache:
            logger.debug("partitioner_cache hit!")
            fused_graph_module = self.partitioner_cache[graph_module]
        else:
            partitioner = CapabilityBasedPartitioner(
                graph_module, self.supported_ops, allows_single_node_partition=False)
            fused_graph_module = partitioner.partition_and_fuse()

            self.partitioner_cache[graph_module] = fused_graph_module

        # Overriding fused_module's __call__() function with lower_to_prims_and_execute()
        for node in fused_graph_module.graph.nodes:
            # TODO: use a better way to identify fused submodule
            if node.op == "call_module" and "fused_" in node.name:
                fused_module = getattr(fused_graph_module, node.name)
                fused_module._wrapped_call = self.lower_to_prims_and_execute

        return fused_graph_module

    def __call__(self, graph_module: GraphModule, _) -> GraphModule:
        # wrap self.compile as __call__ function to fit the interface for AOTAutograd's fw_compiler
        return self.compile(graph_module)
