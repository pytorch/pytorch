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
    raise RuntimeError(
        "This test is not currently used and should be "
        "enabled in discover_tests.py if required."
    )
