import copy
import torch
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupport
from torch.fx.passes.tools_common import CALLABLE_NODE_OPS
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.utils._pytree import tree_map

import operator


import difflib

red = lambda text: f"\033[38;2;255;0;0m{text}\033[38;2;255;255;255m"
green = lambda text: f"\033[38;2;0;255;0m{text}\033[38;2;255;255;255m"
blue = lambda text: f"\033[38;2;0;0;255m{text}\033[38;2;255;255;255m"
white = lambda text: f"\033[38;2;255;255;255m{text}\033[38;2;255;255;255m"

def get_edits_string(old, new):
    result = ""
    codes = difflib.SequenceMatcher(a=old, b=new).get_opcodes()
    for code in codes:
        if code[0] == "equal": 
            result += white(old[code[1]:code[2]])
        elif code[0] == "delete":
            result += red(old[code[1]:code[2]])
        elif code[0] == "insert":
            result += green(new[code[3]:code[4]])
        elif code[0] == "replace":
            result += (red(old[code[1]:code[2]]) + green(new[code[3]:code[4]]))
    return result

class CudaGraphsSupport(OperatorSupport):
    # TODO: why is submodules passed here
    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
        if node.op not in CALLABLE_NODE_OPS:
            return False

        if node.target in [torch.ops.aten.embedding_dense_backward.default]:
            return False

        if node.target in [operator.getitem]:
            return True

        found_not_cuda = False

        def find_not_cuda(t):
            nonlocal found_not_cuda
            if isinstance(t, torch.Tensor) and t.device.type != 'cuda':
                found_not_cuda = True

        for n in node.all_input_nodes:
            tree_map(find_not_cuda, n.meta['fake_result'])

        tree_map(find_not_cuda, node.meta['fake_result'])

        # NB: factory function is accounted for because the result would be
        # cpu or cuda

        return not found_not_cuda

def partition_cudagraphs(gm, inputs):
    """
    Partition an FX graph into sub-GraphModules that can be validly run under
    CUDA graphs.  For a subgraph to be runnable under CUDA, all of the operations
    must involve CUDA tensors only/
    """

    keep_gm = copy.deepcopy(gm)

    FakeTensorProp(gm).propagate(*inputs)
    supported_ops = CudaGraphsSupport()
    print(f"supported ops are {supported_ops}")
    # TODO: single node partition may be wrong due to the pessimization
    # from copying in and out the data.  Check in benchmarks, perhaps
    partitioner = CapabilityBasedPartitioner(gm, supported_ops, allows_single_node_partition=True)
    partitions = partitioner.propose_partitions()
    fused_graph = partitioner.fuse_partitions(partitions)

    # print(f"partitioner is {partitioner}")
    # print(f"partitions are {partitions}")
    # print(f"fused graph is {fused_graph}")

    # print("diff between gm and partitioner")
    # print(get_edits_string(repr(gm), str(partitioner)))

        
    # print("diff between partitioner and partition")
    # print(get_edits_string(str(partitioner), partitions))

    # print("diff between partitions and fused graph")
    # print(get_edits_string(partitions, str(fused_graph)))
    

    # print("diff between gm and fused_graph")
    print(get_edits_string(str(keep_gm), str(fused_graph)))

    print(f"original graph {keep_gm}")
    print(f"fused graph {fused_graph}")
    # print(gm)

    # print(fused_graph)

    return fused_graph


