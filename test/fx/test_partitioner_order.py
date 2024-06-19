# Owner(s): ["module: fx"]

import unittest

from typing import Mapping

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


class AddModule(torch.nn.Module):
    def forward(self, x):
        y = torch.add(x, x)
        z = torch.add(y, x)
        return z


class TestPartitionerOrder(TestCase):
    # partitoner test to check graph node order
    def test_partitioner_order(self):
        m = AddModule()
        traced_m = torch.fx.symbolic_trace(m)
        partions = DummyPartitioner(traced_m).propose_partitions()
        partion_nodes = [list(partition.nodes) for partition in partions]
        node_order = [n.name for n in partion_nodes[0]]
        for _ in range(10):
            traced_m = torch.fx.symbolic_trace(m)
            new_partion = DummyPartitioner(traced_m).propose_partitions()
            new_partion_nodes = [list(partition.nodes) for partition in new_partion]
            new_node_order = [n.name for n in new_partion_nodes[0]]
            self.assertTrue(node_order == new_node_order)


if __name__ == "__main__":
    unittest.main()
