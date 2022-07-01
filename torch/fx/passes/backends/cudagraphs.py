import torch
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupport
from torch.fx.passes.tools_common import CALLABLE_NODE_OPS
from torch.fx import Node
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils._pytree import tree_map

class CudaGraphsSupport(OperatorSupport):
    # TODO: why is submodules passed here
    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
        if not node.op in CALLABLE_NODE_OPS:
            return False

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

class FakeTensorProp(torch.fx.Interpreter):
    def run_node(self, n: Node):
        result = super().run_node(n)
        n.meta['fake_result'] = result
        return result

    def propagate(self, *args):
        # TODO: this is not compositional
        with FakeTensorMode.push() as mode:
            fake_args = [mode.from_tensor(a) for a in args]
            return super().run(*fake_args)

def partition_cudagraphs(gm, inputs):
    FakeTensorProp(gm).propagate(*inputs)
    supported_ops = CudaGraphsSupport()
    # TODO: single node partition is probably wrong due to the pessimization
    # from copying in and out the data
    partitioner = CapabilityBasedPartitioner(gm, supported_ops, allows_single_node_partition=True)
    partitions = partitioner.propose_partitions()
    fused_graph = partitioner.fuse_partitions(partitions)
    return fused_graph
