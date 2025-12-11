# Owner(s): ["module: fx"]

from collections.abc import Mapping

import torch
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupport
from torch.testing._internal.common_utils import TestCase


class DummyDevOperatorSupport(OperatorSupport):
    def is_node_supported(
        self, submodules: Mapping[str, torch.nn.Module], node: torch.fx.Node
    ) -> bool:
        return True


class DummyPartitioner(CapabilityBasedPartitioner):
    def __init__(self, graph_module: torch.fx.GraphModule):
        super().__init__(
            graph_module,
            DummyDevOperatorSupport(),
            allows_single_node_partition=True,
        )


# original graph node order is: ['x', 'add', 'add_1', 'output']
class AddModule(torch.nn.Module):
    def forward(self, x):
        y = torch.add(x, x)
        z = torch.add(y, x)
        return z


class TwoBranchModule(torch.nn.Module):
    def forward(self, x):
        # Branch 1: two supported ops
        y = torch.add(x, 1)
        y = torch.add(y, 1)
        # Branch 2: two supported ops
        z = torch.mul(x, 2)
        z = torch.mul(z, 2)
        return y, z


class CallFunctionOnlySupport(OperatorSupport):
    """Only supports call_function nodes, not placeholders/outputs."""

    def is_node_supported(
        self, submodules: Mapping[str, torch.nn.Module], node: torch.fx.Node
    ) -> bool:
        return node.op == "call_function"


class TestHorizontalFusion(TestCase):
    def test_skip_horizontal_fusion(self):
        m = TwoBranchModule()
        traced_m = torch.fx.symbolic_trace(m)

        # With horizontal fusion (default): should get 1 partition
        # (add and mul get merged through unsupported x)
        partitioner = CapabilityBasedPartitioner(
            traced_m, CallFunctionOnlySupport(), allows_single_node_partition=True
        )
        partitions = partitioner.propose_partitions()
        self.assertEqual(len(partitions), 1)

        # Without horizontal fusion: should get 2 partitions
        # (add and mul stay separate since they don't share a supported edge)
        traced_m = torch.fx.symbolic_trace(m)
        partitioner = CapabilityBasedPartitioner(
            traced_m,
            CallFunctionOnlySupport(),
            allows_single_node_partition=True,
            skip_horizontal_fusion=True,
        )
        partitions = partitioner.propose_partitions()
        self.assertEqual(len(partitions), 2)


class TestPartitionerOrder(TestCase):
    # partitioner test to check graph node order remains the same with the original graph after partitioning
    def test_partitioner_graph_node_order(self):
        m = AddModule()
        traced_m = torch.fx.symbolic_trace(m)
        origin_node_order = [n.name for n in traced_m.graph.nodes]
        partitions = DummyPartitioner(traced_m).propose_partitions()
        partition_nodes = [list(partition.nodes) for partition in partitions]
        partition_node_order = [n.name for n in partition_nodes[0]]
        self.assertTrue(partition_node_order == origin_node_order)

    # partitioner test to check graph node order remains the same during multiple runs
    def test_partitioner_multiple_runs_order(self):
        m = AddModule()
        traced_m = torch.fx.symbolic_trace(m)
        partitions = DummyPartitioner(traced_m).propose_partitions()
        partition_nodes = [list(partition.nodes) for partition in partitions]
        node_order = [n.name for n in partition_nodes[0]]
        for _ in range(10):
            traced_m = torch.fx.symbolic_trace(m)
            new_partition = DummyPartitioner(traced_m).propose_partitions()
            new_partition_nodes = [list(partition.nodes) for partition in new_partition]
            new_node_order = [n.name for n in new_partition_nodes[0]]
            self.assertTrue(node_order == new_node_order)


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
