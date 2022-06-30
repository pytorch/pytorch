from typing import Dict

import torch
from torch.fx import GraphModule
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.backends.nvfuser.operator_support import NvFuserOperatorSupport
from torch._prims.executor import execute
from torch.fx.experimental.proxy_tensor import DecompositionInterpreter
from torch._decomp import decomposition_table

import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def aten_to_dtype(self, dtype: torch.dtype, **kwargs):
    if len(kwargs) > 0 or not dtype:
        raise RuntimeError("No support for other to.dtype() formats other than to.dtype(self, dtype)")
    return torch._prims.convert_element_type(self, dtype)

# decomposition_table currently contains both aten2aten and aten2prim decomposition
# this is a hack to seperate them, as we only need aten2prim decomposition for nvfuser-supported aten graph lowering
aten2aten_decomp = {}
aten2prim_decomp = {}

for op, decomp_fn in decomposition_table.items():
    if "torch._refs" in decomp_fn.__module__:
        aten2prim_decomp[op] = decomp_fn
    else:
        aten2aten_decomp[op] = decomp_fn

aten2prim_decomp[torch.ops.aten.to.dtype] = aten_to_dtype

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
            logging.debug("prim_decomp_cache hit!")
            prim_module = self.prim_decomp_cache[graph_module]
        else:
            prim_graph = torch.fx.Graph()
            DecompositionInterpreter(graph_module, prim_graph, decomposition_table=aten2prim_decomp).run(*args, **kwargs)
            prim_module = torch.fx.GraphModule(graph_module, prim_graph)
            self.prim_decomp_cache[graph_module] = prim_module

            logging.debug("Lower to prims graph: ", prim_module.code)

        # invokes trace executor for running the prim graph
        return execute(prim_module, *args, executor="nvfuser")

    def compile(self, graph_module: GraphModule) -> GraphModule:
        # entry function for nvFuser bsackend
        logging.debug("Compiling graph_module: ", graph_module.code)

        # FX graph based partitioning based on nvfuser supported ops
        if graph_module in self.partitioner_cache:
            logging.debug("partitioner_cache hit!")
            fused_graph_module = self.partitioner_cache[graph_module]
        else:
            partitioner = CapabilityBasedPartitioner(graph_module, self.supported_ops)
            fused_graph_module = partitioner.partition_and_fuse()

            self.partitioner_cache[graph_module] = fused_graph_module

        # Replace fused submodules's __call__() function with lower_to_prims_and_execute()
        num_partitions = 0
        for node in fused_graph_module.graph.nodes:
            # TODO: use a better way to identify fused submodule
            if "fused_" in node.name:
                fused_module = getattr(fused_graph_module, node.name)
                fused_module._wrapped_call = self.lower_to_prims_and_execute
                num_partitions += 1
        # print(num_partitions)

        return fused_graph_module

    def __call__(self, graph_module: GraphModule, _) -> GraphModule:
        # wrap self.compile as __call__ function to fit the interface for AOTAutograd's fw_compiler
        return self.compile(graph_module)
