import logging
import copy

import torch
from torch.fx._symbolic_trace import symbolic_trace

from torch.fx.passes.backends.nvfuser import NvFuserBackend

from torch._decomp import decomposition_table

from functorch import make_fx
from functorch.experimental import functionalize

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# decomposition_table currently contains both aten2aten and aten2prim decomposition
# this is a hack to seperate them, as we only need aten2prim decomposition for nvfuser-supported aten graph lowering
aten2aten_decomp = {}
aten2prim_decomp = {}

aten2aten_decomp_skips = {
    "aten.native_layer_norm_backward.default",
    "aten.embedding_dense_backward.default",   # this aten2aten is hurting nvfuser's perf
    "aten.addmm.default"
}

for op, decomp_fn in decomposition_table.items():
    if "torch._refs" in decomp_fn.__module__:
        aten2prim_decomp[op] = decomp_fn
    else:
        if str(op) not in aten2aten_decomp_skips:
            aten2aten_decomp[op] = decomp_fn
        else:
            print(f"skipping {op}")

def fn(x):
    hardtanh_ = torch.ops.aten.where(x > 0, x, 0)
    a = hardtanh_ + 1
    b = a + 1
    return b

device = 'cuda'
results = []

input = torch.rand(4, device = device)

aten_decomp_gm = make_fx(functionalize(fn), decomposition_table=aten2aten_decomp)(input)
# a hack to work around functionalization bug
aten_decomp_gm.graph.eliminate_dead_code()
aten_decomp_gm.recompile()


nvfuser = NvFuserBackend()
fused_graph_module = nvfuser.compile(copy.deepcopy(aten_decomp_gm))

# count number of partitions
num_partitions = 0
for node in fused_graph_module.graph.nodes:
    if "fused_" in node.name:
        num_partitions += 1
print("num_partitions: ", num_partitions)


nvfuser_result = fused_graph_module(input)
eager_result = fn(input)

print("eager_result", eager_result)
print("nvfuser_result", nvfuser_result)
