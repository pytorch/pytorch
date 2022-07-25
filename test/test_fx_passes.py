# Owner(s): ["oncall: fx"]

import operator
import logging

from torch import sigmoid

import torch
from torch.fx._symbolic_trace import symbolic_trace

from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupport
from torch.fx.passes.utils.fuser_utils import fuse_by_partitions
from torch.fx.passes.utils.matcher_utils import SubgraphMatcher

from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from torch.testing._internal.jit_utils import JitTestCase

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.linear2 = torch.nn.Linear(4, 4)
        self.param = torch.nn.Parameter(torch.rand(4, 4))

    def forward(self, a, b, c):
        add = a + b

        linear_1 = self.linear(add)

        add_1 = add + c
        add_2 = add_1 + self.param
        add_3 = add_1 + linear_1
        add_4 = add_2 + add_3

        linear_2 = self.linear2(add_4)

        add_5 = linear_2 + add_4
        add_6 = add_5 + a
        relu = add_6.relu()

        return add_4, add_6, relu

class TestPartitionFunctions:
    @staticmethod
    def forward1(a, b, c):
        add = a + b
        add_1 = add + b
        add_2 = add_1 + c
        relu_1 = add_2.relu()
        add_3 = add_1 + add_2
        add_4 = add_1 + relu_1 + add_3
        relu_2 = add_4.relu()
        add_5 = relu_2 + add_4
        add_6 = add_5 + add_4
        return add_4, add_6

    @staticmethod
    def forward2(a, b, _):
        add = a + b
        add_1 = add + b
        relu_1 = add_1.relu()  # blocked by this
        add_3 = add_1 + relu_1
        add_4 = add_1 + add_3
        return add_4, add_1

    @staticmethod
    def forward3(a, b, c):
        add = a + b
        add_1 = a + c
        add_2 = b + c
        return add, add_1, add_2

    @staticmethod
    def forward4(a, b, c):
        add = a + b
        add_1 = a + c
        add_2 = b + c
        return torch.where(add > 0, add_1, add_2)

    @staticmethod
    def forward5(a, b, c):
        # add should be fused right branch, as left branch is not supported
        add = a + 1
        # left branch
        relu = add.relu()
        # right branch
        add_1 = add + 2
        return relu, add_1

    @staticmethod
    def forward6(a, b, c):
        # add should have its own partition, as neither branchs are supported
        add = a + 1
        # left branch
        relu = add.relu()
        # right branch
        relu_1 = add.relu()
        return relu, relu_1

    @staticmethod
    def forward7(a, b, c):
        # both branches are supported, all adds should be fused together
        add = a + 1
        # left branch
        add_1 = add + 2
        # right branch is larger
        add_2 = add + 1
        add_3 = add_2 + 1
        return add_3, add_1

    @staticmethod
    def forward8(a, b, c):
        # both branches are in the same partition, add should join the same partition
        add = a + 1
        # left branch
        add_1 = add + 2
        # right branch
        add_2 = add + 1
        # left and right branch merges
        add_3 = add_2 + add_1

        return add_3

    @staticmethod
    def forward9(a, b, c):
        add = a + 1
        # branch 1
        add_1 = add + 1
        # branch 2
        add_2 = add + 1
        # branch_3
        add_3 = add + 1
        out = torch.stack([add_1, add_2, add_3])
        return out

    @staticmethod
    def forward10(a, b, c):
        add = a + 1
        # branch 1
        add_1 = add + 1
        # branch 2
        add_2 = add + 1
        # branch 3: depends on branch 2
        add_3 = add + add_2
        out = torch.stack([add_1, add_2, add_3])
        return out

    @staticmethod
    def forward11(a, b, c):
        add = a + 1
        # branch 1
        add_1 = add.relu()
        # branch 2 depends on branch 1
        add_2 = add + add_1
        # branch 3
        add_3 = add.relu()
        out = torch.stack([add_1, add_2, add_3])
        return out

# A mock OperatorSupport class, where only operator.add is supported
class MockOperatorSupport(OperatorSupport):
    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
        return node.op == "call_function" and node.target in {operator.add}


@instantiate_parametrized_tests
class TestFXGraphPasses(JitTestCase):

    @parametrize("fn, expected_partition", [
        (TestPartitionFunctions.forward1, [["add_7", "add_6"], ["add_5", "add_4", "add_3"], ["add_2", "add_1", "add"]]),
        (TestPartitionFunctions.forward2, [["add_3", "add_2"], ["add_1", "add"]]),

        # 2 branches cases
        (TestPartitionFunctions.forward5, [["add_1", "add"]]),
        (TestPartitionFunctions.forward6, [["add"]]),
        (TestPartitionFunctions.forward7, [["add_3", "add_2", "add", "add_1"]]),
        (TestPartitionFunctions.forward8, [["add_3", "add_2", "add", "add_1"]]),

        # 3 branch cases
        (TestPartitionFunctions.forward9, [['add_3', 'add_2', 'add_1', 'add']]),
        (TestPartitionFunctions.forward10, [['add_3', 'add_2', 'add', 'add_1']]),
        (TestPartitionFunctions.forward11, [['add_1'], ['add']]),
    ])
    def test_partitioner(self, fn, expected_partition):
        traced = symbolic_trace(fn)

        supported_ops = MockOperatorSupport()
        partitioner = CapabilityBasedPartitioner(traced, supported_ops, allows_single_node_partition=True)
        partitions = partitioner.propose_partitions()

        partitions_name = [[node.name for node in partition.nodes] for partition in partitions]
        assert len(partitions_name) == len(expected_partition)
        for i in range(len(partitions_name)):
            assert set(partitions_name[i]) == set(expected_partition[i])

        fused_graph = partitioner.fuse_partitions(partitions)

        a, b, c = torch.rand(4), torch.rand(4), torch.rand(4)

        expected = fn(a, b, c)
        result = fused_graph(a, b, c)
        torch.testing.assert_close(expected, result)


    @parametrize("fn, expected_partition", [
        # horizontal fusion without a common downstream node, not supported yet
        (TestPartitionFunctions.forward3, [["add_2", "add_1", "add"]]),
        # horizontal fusion with a common downstream node, not supported yet
        (TestPartitionFunctions.forward4, [["add_2", "add_1", "add"]]),
    ])
    def test_partitioner_xfail(self, fn, expected_partition):
        traced = symbolic_trace(fn)

        supported_ops = MockOperatorSupport()
        partitioner = CapabilityBasedPartitioner(traced, supported_ops, allows_single_node_partition=True)
        partitions = partitioner.propose_partitions()

        partitions_name = [[node.name for node in partition.nodes] for partition in partitions]
        with self.assertRaises(Exception):
            assert len(partitions_name) == len(expected_partition)

    @parametrize("partition", [
        [['add', 'add_1'], ['add_5', 'add_6']],
        [['add', 'add_1', 'add_2']],  # vertical fusion
        [['add_2', 'add_3']],         # horizontal fusion
        [['add_3', 'add_4']],
        [['add_6', 'add_5']],     # arbitray node order
        [['add_4', 'add_1', 'add_3', 'add_2']],           # arbitray node order
        [['add_5', 'add_6'], ['add_1', 'add_2', 'add_3', 'add_4']],  # arbitray partition order
        [['add_5', 'linear2']],   # includes call_function + call_module node
        [['add_6', 'relu']],   # includes call_function + call_module node
        [['param', 'add_2']],   # includes get_attr + call_module nodes
        [['param', 'add_1', 'linear']],   # includes get_attr + call_function + call_module nodes
        [["add", "linear", "add_1", "param", "add_2", "add_3", "add_4", "linear2", "add_5", "add_6", "relu"]],  # full graph
    ])
    def test_fuser_util(self, partition):
        m = TestModule()
        gm = symbolic_trace(m)

        nodes_by_name = {node.name : node for node in gm.graph.nodes}

        partitions = []
        for node_names in partition:
            partitions.append([nodes_by_name[name] for name in node_names])

        fused_graph = fuse_by_partitions(gm, partitions)

        a, b, c = torch.rand(4), torch.rand(4), torch.rand(4)

        expected = m(a, b, c)
        result = fused_graph(a, b, c)

        torch.testing.assert_close(expected, result)

    @parametrize("partition", [
        [['add', 'add_1'], ['add_1', 'add_5', 'add_6']],  # add_1 exists in multiple partitions
        [['add', 'add_1', 'add_3']],    # invalid partition: circular dependency
        [['add_4', 'add_5']],    # invalid partition: circular dependency
        [['relu', 'add_5']],    # invalid partition: circular dependency
    ])
    def test_fuser_util_xfail(self, partition):
        m = TestModule()
        gm = symbolic_trace(m)

        nodes_by_name = {node.name : node for node in gm.graph.nodes}

        partitions = []
        for node_names in partition:
            partitions.append([nodes_by_name[name] for name in node_names])

        with self.assertRaises(Exception):
            fuse_by_partitions(gm, partitions)


class TestCase:
    def forward(x):
        val = torch.neg(x)
        return torch.add(val, val)

    def pattern(a):
        return torch.neg(a)

    expects = [
        # match_output, match_placeholder, num_matches
        (False, False, 1),
        (True, False, 0),
        (False, True, 1),
        (True, True, 0)
    ]

class TestCase1:
    def forward(x):
        val = torch.neg(x) + torch.relu(x)
        return torch.add(val, val)

    def pattern(a):
        return torch.neg(a) + torch.relu(a)

    expects = [
        # match_output, match_placeholder, num_matches
        (False, False, 1),
        (True, False, 0),
        (False, True, 1),
        (True, True, 0)
    ]

class TestCase2:
    def forward(x, w1, w2):
        m1 = torch.cat([w1, w2]).sum()
        m2 = torch.cat([w2, w1]).sum()
        m3 = torch.cat([m1, m2]).sum()
        return x + torch.max(m1) + torch.max(m2) + m3

    def pattern(a, b):
        return torch.cat([a, b]).sum()

    expects = [
        # match_output, match_placeholder, num_matches
        (False, False, 3),
        (True, False, 0),
        (False, True, 2),
        (True, True, 0)
    ]

class TestCase3:
    def forward(x, y):
        return torch.mm(x, y)

    def pattern(x, y):
        return torch.mm(x, y)

    expects = [
        # match_output, match_placeholder, num_matches
        (False, False, 1),
        (True, False, 1),
        (False, True, 1),
        (True, True, 1)
    ]

class TestCase4:
    def forward(x, y):
            val = torch.neg(y) + torch.relu(x)
            return torch.add(val, val)

    def pattern(x):
        return torch.relu(x)

    expects = [
        # match_output, match_placeholder, num_matches
        (False, False, 1),
        (True, False, 0),
        (False, True, 1),
        (True, True, 0)
    ]

class TestCase5:
    def forward(x):
        a = torch.neg(x)
        return torch.add(a, a)

    def pattern(x):
        a = torch.neg(x)
        return torch.add(a, a)

    expects = [
        # match_output, match_placeholder, num_matches
        (False, False, 1),
        (True, False, 1),
        (False, True, 1),
        (True, True, 1)
    ]

class DiamondShapePatternTestCast:
    def forward(x):
        a = torch.neg(x)

        a = a.relu()
        left = a.sigmoid()
        right = a.relu()
        out = left + right

        return out

    def pattern(a):
        a = a.relu()
        left = a.sigmoid()
        right = a.relu()
        out = left + right
        return out

    expects = [
        # match_output, match_placeholder, num_matches
        (False, False, 1),
        (True, False, 1),
        (False, True, 0),
        (True, True, 0)
    ]

class TestCase7:
    def forward(x, w1, w2, b1, b2):
        m0 = torch.cat([w1, w2])
        m1 = torch.cat([w1, w2])
        m2 = torch.cat([x, b2])
        t0 = torch.addmm(b1, m1, m2.t())
        t1 = torch.sum(w1, 1)
        t2 = torch.addmm(b1, m1, m2.t())
        return torch.sum(t1), torch.sum(t2)

    def pattern(x, w1, w2, b1, b2):
        m1 = torch.cat([w1, w2])
        m2 = torch.cat([x, b2])
        return torch.addmm(b1, m1, m2.t())

    expects = [
        # match_output, match_placeholder, num_matches
        (False, False, 2),
        (True, False, 0),
        (False, True, 2),
        (True, True, 0)
    ]

class TestCase8:
    def forward(x):
        x = torch.sigmoid(x)
        x = torch.sigmoid(x)
        return torch.sigmoid(x)

    def pattern(x):
        return torch.sigmoid(torch.sigmoid(x))

    expects = [
        # match_output, match_placeholder, num_matches
        (False, False, 2),
        (True, False, 1),
        (False, True, 1),
        (True, True, 0)
    ]

class JerryZhangTestCase:
    def forward(x):
        x += 3
        x = x.dequantize()
        x = torch.sigmoid(x)
        x = x.to(torch.float16)
        return x

    def pattern(x):
        x = x.dequantize()
        x = torch.sigmoid(x)
        x = x.to(torch.float16)
        return x

    expects = [
        # match_output, match_placeholder, num_matches
        (False, False, 1),
        (True, False, 1),
        (False, True, 0),
        (True, True, 0)
    ]

class MultipleOutputsWithDependency:
    def forward(x):
        y = x.relu()
        z = y.sigmoid()
        return z, y

    def pattern(a):
        b = a.relu()
        c = b.sigmoid()
        return b, c     # outputs have data dependency

    expects = [
        # match_output, match_placeholder, num_matches
        (False, False, 1),
        (True, False, 0),
        (False, True, 1),
        (True, True, 0)
    ]

class MultipleOutputsWithoutDependency:

    def forward(x):
        x = x + 1

        # target subgraph to match
        x = x.relu()
        z = x.sum()
        y = x.sigmoid()

        out = y.sigmoid() + z.sum()
        return out

    def pattern(a):
        a = a.relu()
        b = a.sigmoid()
        c = a.sum()
        return b, c

    expects = [
        # match_output, match_placeholder, num_matches
        (False, False, 1),
        (True, False, 0),
        (False, True, 0),
        (True, True, 0)
    ]

class MultipleOutputsMultipleOverlappingMatches:

    def forward(x):
        x = x + 1

        # target subgraph to match
        x = x.relu()
        z = x.sum()
        z1 = x.sum()
        y = x.sigmoid()
        y1 = x.sigmoid()

        return z + z1 + y + y1

    def pattern(a):
        a = a.relu()
        b = a.sigmoid()
        c = a.sum()
        return b, c

    expects = [
        # match_output, match_placeholder, num_matches
        (False, False, 4),
    ]

class MultipleOutputsMultipleNonOverlappingMatches:

    def forward(x):
        x = x + 1

        # target subgraph to match
        x = x.relu()
        z = x.sum()
        y = x.sigmoid()

        x = x.relu()
        z1 = x.sum()
        y1 = x.sigmoid()

        x = x.relu()
        z2 = x.sum()
        y2 = x.sigmoid()

        return z + z1 + y + y1 + z2 + y2

    def pattern(a):
        a = a.relu()
        b = a.sigmoid()
        c = a.sum()
        return b, c

    expects = [
        # match_output, match_placeholder, num_matches
        (False, False, 3),
    ]

class MultipleOutputsIdenticalAnchor:

    def forward(x):
        x = x + 1

        # target subgraph to match
        x = x.relu()
        y = x.sigmoid()
        y1 = x.sigmoid()

        return  y, y1

    def pattern(a):
        a = a.relu()
        b = a.sigmoid()
        b1 = a.sigmoid()
        return b, b1

    expects = [
        # match_output, match_placeholder, num_matches
        # (False, False, 2),  # FIXME: currently still matches to 2, should fix to 1
        (True, False, 1),
        (False, True, 0),
    ]

@instantiate_parametrized_tests
class TestFXMatcherUtils(JitTestCase):

    @parametrize("test_case", [
        TestCase,
        TestCase1,
        TestCase2,
        TestCase3,
        TestCase4,
        TestCase5,
        DiamondShapePatternTestCast,
        TestCase7,
        TestCase8,
        JerryZhangTestCase,
        MultipleOutputsWithDependency,
        MultipleOutputsWithoutDependency,
        MultipleOutputsMultipleOverlappingMatches,
        MultipleOutputsMultipleNonOverlappingMatches,
        MultipleOutputsIdenticalAnchor,
    ])
    def test_pattern_matcher(self, test_case):

        traced = symbolic_trace(test_case.forward)
        pattern_traced = symbolic_trace(test_case.pattern)

        for match_output, match_placeholder, num_matches in test_case.expects:

            matcher = SubgraphMatcher(pattern_traced.graph,
                                    match_output=match_output,
                                    match_placeholder=match_placeholder)
            matches = matcher.match(traced.graph)

            assert len(matches) == num_matches

            for match in matches:
                for node in pattern_traced.graph.nodes:
                    if not match_placeholder and node.op == "placeholder":
                        continue
                    if not match_output and node.op == "output":
                        continue
                    assert node in match.nodes_map


if __name__ == "__main__":
    run_tests()
