# Owner(s): ["module: fx"]

import operator
from unittest.mock import patch

import torch
from torch.fx import Graph, GraphModule, symbolic_trace
from torch.fx.passes.infra.partitioner import (
    _DependencyViewer,
    CapabilityBasedPartitioner,
)
from torch.fx.passes.operator_support import chain, create_op_support, OpSupports
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


def _make_merge_regression_graph():
    # -1 denotes the input; other indices refer to preceding addition nodes.
    parents = [
        (-1, -1),
        (0, 0),
        (1, -1),
        (-1, 1),
        (-1, 0),
        (-1, 4),
        (0, 5),
        (6, 1),
        (1, 7),
        (1, 4),
        (6, 0),
        (5, 0),
        (4, 4),
        (2, -1),
        (9, 8),
        (4, 1),
        (6, -1),
        (12, 2),
        (15, 13),
        (4, 3),
        (19, 1),
    ]
    graph = Graph()
    x = graph.placeholder("x")
    nodes = []
    coefficients = []
    for predecessors in parents:
        args = tuple(x if i == -1 else nodes[i] for i in predecessors)
        nodes.append(graph.call_function(operator.add, args))
        coefficients.append(sum(1 if i == -1 else coefficients[i] for i in predecessors))
    graph.output(tuple(nodes))
    return GraphModule(torch.nn.Module(), graph), coefficients


@instantiate_parametrized_tests
class TestCapabilityPartitionerRegressions(TestCase):
    def test_dependency_viewer_for_subclasses(self):
        class LegacyPartitioner(CapabilityBasedPartitioner):
            def downstreams_of(self, node):
                return self.dependency_viewer.downstreams_of(node)

        gm = symbolic_trace(lambda x: (x + 1) * 2)
        x, add, mul, output = gm.graph.nodes
        support = create_op_support(lambda submodules, node: node.op == "call_function")
        partitioner = LegacyPartitioner(gm, support)
        downstreams = partitioner.downstreams_of(x)
        self.assertEqual(downstreams, {add, mul, output})
        self.assertEqual(partitioner.downstreams_of(add), {mul, output})
        self.assertEqual(partitioner.downstreams_of(output), set())
        partitions = partitioner.propose_partitions()
        self.assertEqual([set(partition.nodes) for partition in partitions], [{add, mul}])
        self.assertIs(partitioner.downstreams_of(x), downstreams)
        self.assertEqual(downstreams, {add, mul, output})

        replacement = _DependencyViewer(gm)
        partitioner.dependency_viewer = replacement
        self.assertIs(partitioner.dependency_viewer, replacement)

    def test_partitioning_does_not_construct_legacy_dependencies(self):
        gm = symbolic_trace(lambda x: (x + 1) * 2)
        support = create_op_support(lambda submodules, node: node.op == "call_function")
        with patch(
            "torch.fx.passes.infra.partitioner._DependencyViewer",
            side_effect=AssertionError("Legacy dependencies should remain lazy"),
        ):
            partitioner = CapabilityBasedPartitioner(gm, support)
            partitions = partitioner.propose_partitions()
        self.assertEqual(len(partitions), 1)

    @parametrize("retrace", [False, True])
    @parametrize("skip_horizontal_fusion", [False, True])
    def test_reachability_after_partition_representative_changes(
        self, retrace, skip_horizontal_fusion
    ):
        gm, coefficients = _make_merge_regression_graph()
        if retrace:
            gm = symbolic_trace(gm)
        gm.graph.lint()
        x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        expected = tuple(x * coefficient for coefficient in coefficients)
        self.assertEqual(gm(x), expected)
        support = chain(
            create_op_support(lambda submodules, node: node.op == "call_function"),
            OpSupports.decline_if_node_in_names(
                {
                    "add",
                    "add_5",
                    "add_8",
                    "add_9",
                    "add_11",
                    "add_12",
                    "add_16",
                    "add_18",
                }
            ),
        )
        partitioner = CapabilityBasedPartitioner(
            gm,
            support,
            allows_single_node_partition=True,
            skip_horizontal_fusion=skip_horizontal_fusion,
        )
        partitions = partitioner.propose_partitions()
        # A stale reference to partition 2 misses its later representative 11.
        # With vertical fusion only, the old implementation creates the cycle
        # P11 -> P9 -> add_12 -> P11 and fuse_partitions raises an AssertionError.
        fused = partitioner.fuse_partitions(partitions)
        fused.graph.lint()
        self.assertEqual(fused(x), expected)


if __name__ == "__main__":
    run_tests()
