# Owner(s): ["module: fx.passes"]

from dataclasses import dataclass
import operator
import logging

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

    @staticmethod
    def forward12(a, b, c):
        b0 = a + 1.0
        c0 = a + 1.5
        x0 = b0.relu()
        x1 = c0.relu()
        b1 = b0 + x1
        c1 = c0 + 1.2
        # c2 has dependency on x0 & b0, when we merge {c0, c1, c2}
        # this dependency should be updated to the fusion group and reflected
        # on the decision to not fuse b0 & b1, which forms a cyclic dependency in
        # the new graph
        c2 = x0 + c0
        return b1, c2

    @staticmethod
    def forward13(a, b, c):
        a0, a1, a2, a3 = a.split(1, 0)
        b1 = a0 + b
        c1 = a1 + c
        return b1 + c1

    @staticmethod
    def forward14(a, b, c):
        a0, a1 = torch.ops.aten.std_mean(a)
        out = a0 + 1.0
        return out

    @staticmethod
    def forward15(a, b, c):
        a0 = torch.ops.aten.view(a, [2, 2])
        a1 = torch.ops.aten.permute(a0, [1, 0])
        a2 = a1 + 1.0
        a3 = torch.ops.aten.permute(a2, [1, 0])
        a4 = a3 + 1.0
        a5 = torch.ops.aten.permute(a4, [1, 0])
        return torch.ops.aten.permute(a5, [1, 0])

    @staticmethod
    def forward16(a, b, c):
        a0 = a - 1.0
        a1 = torch.ops.aten.view(a0, [2, 2])
        a2 = torch.ops.aten.permute(a1, [1, 0])
        a3 = a2 + 1.0
        a4 = torch.ops.aten.permute(a3, [1, 0])
        a5 = a4 + 1.0
        a6 = torch.ops.aten.permute(a5, [1, 0])
        a7 = torch.ops.aten.permute(a6, [1, 0])
        return a7 - 1.0

# A mock OperatorSupport class, where only operator.add is supported
class MockOperatorSupport(OperatorSupport):
    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
        return (node.op == "call_function" and
                node.target in {operator.add, operator.getitem, torch.ops.aten.view, torch.ops.aten.permute, torch.ops.aten.std_mean})

@instantiate_parametrized_tests
class TestFXGraphPasses(JitTestCase):

    @parametrize("fn, expected_partition, bookend_non_compute_pass", [
        (TestPartitionFunctions.forward1, [["add_7", "add_6"], ["add_5", "add_4", "add_3"], ["add_2", "add_1", "add"]], False),
        (TestPartitionFunctions.forward2, [["add_3", "add_2"], ["add_1", "add"]], False),

        # 1 horizontal fusion with common producer
        (TestPartitionFunctions.forward3, [["add_2", "add_1", "add"]], False),
        (TestPartitionFunctions.forward4, [["add_2", "add_1", "add"]], False),

        # 2 branches cases
        (TestPartitionFunctions.forward5, [["add_1", "add"]], False),
        (TestPartitionFunctions.forward6, [["add"]], False),
        (TestPartitionFunctions.forward7, [["add_3", "add_2", "add", "add_1"]], False),
        (TestPartitionFunctions.forward8, [["add_3", "add_2", "add", "add_1"]], False),

        # 3 branch cases
        (TestPartitionFunctions.forward9, [['add_3', 'add_2', 'add_1', 'add']], False),
        (TestPartitionFunctions.forward10, [['add_3', 'add_2', 'add', 'add_1']], False),
        (TestPartitionFunctions.forward11, [['add_1'], ['add']], False),

        # 4 not necessarily the only partition, just to verify that there's no cyclic dependency after partition
        (TestPartitionFunctions.forward12, [["add_2"], ["add_3", "add_4", "add_1"], ["add"]], False),

        # 5 getitem special case
        (TestPartitionFunctions.forward13, [["add_2", "add_1", "add"]], False),
        (TestPartitionFunctions.forward14, [["add", "std_mean", "getitem", "getitem_1"]], False),

        # 6 bookend non_compute pass
        (TestPartitionFunctions.forward15, [["permute_1", "add_1", "add"]], True),
        (TestPartitionFunctions.forward15, [['add_1', 'add', 'permute_1', 'view', 'permute_2', 'permute_3', 'permute']], False),
        (TestPartitionFunctions.forward16, [["permute_1", "add_1", "add"]], True),
        (TestPartitionFunctions.forward16, [['add_1', 'add', 'permute_1', 'view', 'permute_2', 'permute_3', 'permute']], False),
    ])
    def test_partitioner(self, fn, expected_partition, bookend_non_compute_pass):
        traced = symbolic_trace(fn)

        non_compute_ops = []
        if bookend_non_compute_pass:
            non_compute_ops = ["torch.ops.aten.view", "torch.ops.aten.permute"]

        supported_ops = MockOperatorSupport()
        partitioner = CapabilityBasedPartitioner(traced,
                                                 supported_ops,
                                                 allows_single_node_partition=True,
                                                 non_compute_ops=non_compute_ops)
        partitions = partitioner.propose_partitions()
        if bookend_non_compute_pass:
            partitioner.remove_bookend_non_compute_ops(partitions)

        partitions_name = [[node.name for node in partition.nodes] for partition in partitions]
        assert len(partitions_name) == len(expected_partition)
        for i in range(len(partitions_name)):
            assert set(partitions_name[i]) == set(expected_partition[i])

        fused_graph = partitioner.fuse_partitions(partitions)

        a, b, c = torch.rand(4), torch.rand(4), torch.rand(4)

        expected = fn(a, b, c)
        result = fused_graph(a, b, c)
        torch.testing.assert_close(expected, result)

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

@dataclass
class TestCase:
    match_output: bool
    match_placeholder: bool
    num_matches: int
    remove_overlapping_matches: bool = True

class SingleNodePattern:
    @staticmethod
    def forward(x):
        val = torch.neg(x)
        return torch.add(val, val)

    @staticmethod
    def pattern(a):
        return torch.neg(a)

    test_cases = [
        # match_output, match_placeholder, num_matches
        TestCase(False, False, 1),
        TestCase(True, False, 0),
        TestCase(False, True, 1),
        TestCase(True, True, 0)
    ]
class SimplePattern:
    @staticmethod
    def forward(x, w1, w2):
        m1 = torch.cat([w1, w2]).sum()
        m2 = torch.cat([w2, w1]).sum()
        m3 = torch.cat([m1, m2]).sum()
        return x + torch.max(m1) + torch.max(m2) + m3

    @staticmethod
    def pattern(a, b):
        return torch.cat([a, b]).sum()

    test_cases = [
        # match_output, match_placeholder, num_matches
        TestCase(False, False, 3),
        TestCase(True, False, 0),
        TestCase(False, True, 2),
        TestCase(True, True, 0)
    ]

class SimpleFullGraphMatching:
    @staticmethod
    def forward(x):
        a = torch.neg(x)
        return torch.add(a, a)

    @staticmethod
    def pattern(x):
        a = torch.neg(x)
        return torch.add(a, a)

    test_cases = [
        # match_output, match_placeholder, num_matches
        TestCase(False, False, 1),
        TestCase(True, False, 1),
        TestCase(False, True, 1),
        TestCase(True, True, 1)
    ]

class DiamondShapePatternTestCase:
    @staticmethod
    def forward(x):
        a = torch.neg(x)

        a = a.relu()
        left = a.sigmoid()
        right = a.relu()
        out = left + right

        return out

    @staticmethod
    def pattern(a):
        a = a.relu()
        left = a.sigmoid()
        right = a.relu()
        out = left + right
        return out

    test_cases = [
        # match_output, match_placeholder, num_matches
        TestCase(False, False, 1),
        TestCase(True, False, 1),
        TestCase(False, True, 0),
        TestCase(True, True, 0)
    ]

class NonFullyContainedMatches:
    @staticmethod
    def forward(x, w1, w2, b1, b2):
        # fully contained matched subgraph
        m1 = torch.cat([w1, w2])
        m2 = torch.cat([x, b2])
        t0 = torch.addmm(b1, m1, m2.t())
        t0_sum = torch.sum(t0)   # use of t0 is not leaking

        # leaking matched subgraph, m3 is leaked
        m3 = torch.cat([w1, w2])
        m4 = torch.cat([x, b2])
        t1 = torch.addmm(b1, m3, m4.t())
        m3_sum = torch.sum(m3)

        return t0_sum, m3_sum

    @staticmethod
    def pattern(x, w1, w2, b1, b2):
        m1 = torch.cat([w1, w2])
        m2 = torch.cat([x, b2])
        return torch.addmm(b1, m1, m2.t())

    test_cases = [
        # match_output, match_placeholder, num_matches
        TestCase(False, False, 1),

        TestCase(True, False, 0),

        TestCase(False, True, 1),     # leaked used of placeholder is not leaking
    ]

class ChainRepeatedPattern:
    @staticmethod
    def forward(x):
        x = torch.sigmoid(x)
        x = torch.sigmoid(x)
        x = torch.sigmoid(x)
        return torch.sigmoid(x)

    @staticmethod
    def pattern(x):
        return torch.sigmoid(torch.sigmoid(x))

    test_cases = [
        # match_output, match_placeholder, num_matches
        TestCase(False, False, 3, remove_overlapping_matches=False),
        TestCase(False, False, 2, remove_overlapping_matches=True),
        TestCase(True, False, 1),
        TestCase(False, True, 1),
        TestCase(True, True, 0)
    ]

class QuantizationModel:
    @staticmethod
    def forward(x):
        x += 3
        x = x.dequantize()
        x = torch.sigmoid(x)
        x = x.to(torch.float16)
        return x

    @staticmethod
    def pattern(x):
        x = x.dequantize()
        x = torch.sigmoid(x)
        x = x.to(torch.float16)
        return x

    test_cases = [
        # match_output, match_placeholder, num_matches
        TestCase(False, False, 1),
        TestCase(True, False, 1),
        TestCase(False, True, 0),
        TestCase(True, True, 0)
    ]

class MultipleOutputsWithDependency:
    @staticmethod
    def forward(x):
        y = x.relu()
        z = y.sigmoid()
        return z, y

    @staticmethod
    def pattern(a):
        b = a.relu()
        c = b.sigmoid()
        return b, c     # outputs have data dependency

    test_cases = [
        # match_output, match_placeholder, num_matches
        TestCase(False, False, 1),
        TestCase(True, False, 0),
        TestCase(False, True, 1),
        TestCase(True, True, 0)
    ]

class MultipleOutputsWithoutDependency:
    @staticmethod
    def forward(x):
        x = x + 1

        # target subgraph to match
        x = x.relu()
        z = x.sum()
        y = x.sigmoid()

        out = y.sigmoid() + z.sum()
        return out

    @staticmethod
    def pattern(a):
        a = a.relu()
        b = a.sigmoid()
        c = a.sum()
        return b, c

    test_cases = [
        # match_output, match_placeholder, num_matches
        TestCase(False, False, 1),
        TestCase(True, False, 0),
        TestCase(False, True, 0),
        TestCase(True, True, 0)
    ]

class MultipleOutputsMultipleOverlappingMatches:
    @staticmethod
    def forward(x):
        x = x + 1

        # target subgraph to match
        x = x.relu()
        z = x.sum()
        z1 = x.sum()
        y = x.sigmoid()
        y1 = x.sigmoid()

        return z + z1 + y + y1

    @staticmethod
    def pattern(a):
        a = a.relu()
        b = a.sigmoid()
        c = a.sum()
        return a, b, c

    test_cases = [
        # match_output, match_placeholder, num_matches
        TestCase(False, False, 4, remove_overlapping_matches=False),
        TestCase(False, False, 1, remove_overlapping_matches=True),
    ]

class MultipleOutputsMultipleNonOverlappingMatches:
    @staticmethod
    def forward(x):
        x = x + 1

        # target subgraph to match
        x = x.relu()
        z = x.sum()
        y = x.sigmoid()

        x = x.relu()
        z1 = x.sum()
        y1 = x.sigmoid()

        return z + z1 + y + y1

    @staticmethod
    def pattern(a):
        a = a.relu()
        b = a.sigmoid()
        c = a.sum()
        return b, c

    test_cases = [
        # match_output, match_placeholder, num_matches
        TestCase(False, False, 1),
    ]

class MultipleOutputsIdenticalAnchor:
    @staticmethod
    def forward(x):
        x = x + 1

        # target subgraph to match
        x = x.relu()
        y = x.sigmoid()
        y1 = x.sigmoid()

        return y, y1

    @staticmethod
    def pattern(a):
        a = a.relu()
        b = a.sigmoid()
        b1 = a.sigmoid()
        return b, b1

    test_cases = [
        # match_output, match_placeholder, num_matches
        # (False, False, 2),  # FIXME: currently still matches to 2, should fix to 1
        TestCase(True, False, 1),
        TestCase(False, True, 0),
    ]


class MultipleOutputsHorizontalPattern:
    @staticmethod
    def forward(x):
        x = x + 1

        # target subgraph to match
        y1 = x.relu()
        y2 = x.sigmoid()

        return y1, y2

    @staticmethod
    def pattern(a):
        b1 = a.relu()
        b2 = a.sigmoid()

        return b1, b2

    test_cases = [
        # match_output, match_placeholder, num_matches
        TestCase(False, False, 1),
        TestCase(True, False, 1),
        TestCase(False, True, 0),
        TestCase(True, True, 0)
    ]

class MultiOutputWithWithInvalidMatches:
    @staticmethod
    def forward(x):
        res0 = torch.nn.functional.linear(x, torch.rand(3, 3))
        res1 = torch.sigmoid(res0)
        res2 = res0 * res1
        res3 = torch.sum(res2, dim=1)
        return res3

    @staticmethod
    def pattern(a, b, c):
        lin_res = torch.nn.functional.linear(a, b)
        mul_res = lin_res * c
        return lin_res, mul_res

    test_cases = [
        # match_output, match_placeholder, num_matches
        TestCase(False, False, 0),
        TestCase(True, False, 0),
        TestCase(False, True, 0),
    ]

class QuantizationFp8Pattern:
    @classmethod
    def setup(cls):
        cls.quantization = torch.library.Library("fp8_quantization", "DEF")
        cls.quantization.define("quantize_per_tensor_affine_fp8(Tensor self, int dtype, float scale) -> Tensor")
        cls.quantization.define("dequantize_per_tensor_affine_fp8(Tensor self, int dtype, float scale) -> Tensor")

    @classmethod
    def tearDown(cls):
        del cls.quantization

    @staticmethod
    def forward(self, arg0_1, arg1_1):
        qt = torch.ops.fp8_quantization
        _scale_0 = self._scale_0
        quantize_per_tensor_affine_fp8 = qt.quantize_per_tensor_affine_fp8(arg0_1, 0, _scale_0)
        dequantize_per_tensor_affine_fp8 = qt.dequantize_per_tensor_affine_fp8(quantize_per_tensor_affine_fp8, 0, _scale_0)
        _scale_1 = self._scale_0
        quantize_per_tensor_affine_fp8_1 = qt.quantize_per_tensor_affine_fp8(arg1_1, 0, _scale_1)
        dequantize_per_tensor_affine_fp8_1 = qt.dequantize_per_tensor_affine_fp8(quantize_per_tensor_affine_fp8_1, 0, _scale_1)
        add = torch.ops.aten.add.Tensor(dequantize_per_tensor_affine_fp8, dequantize_per_tensor_affine_fp8_1)
        _scale_2 = self._scale_0
        quantize_per_tensor_affine_fp8_2 = qt.quantize_per_tensor_affine_fp8(add, 0, _scale_2)
        dequantize_per_tensor_affine_fp8_2 = qt.dequantize_per_tensor_affine_fp8(quantize_per_tensor_affine_fp8_2, 0, _scale_2)
        return dequantize_per_tensor_affine_fp8_2

    @staticmethod
    def pattern(a, a_dtype, a_scale, b, b_dtype, b_scale, out_scale):
        qt = torch.ops.fp8_quantization
        a = qt.dequantize_per_tensor_affine_fp8(a, a_dtype, a_scale)
        b = qt.dequantize_per_tensor_affine_fp8(b, b_dtype, b_scale)
        output = torch.ops.aten.add.Tensor(a, b)

        qt.dequantize_per_tensor_affine_fp8

        output = qt.quantize_per_tensor_affine_fp8(output, a_dtype, out_scale)
        return output

    test_cases = [
        # match_output, match_placeholder, num_matches
        TestCase(False, False, 1),
    ]

class NoAnchorFound:
    # This test case is for pattern where no matching anchor is found in the target graph
    # `anchor` is the starting point of the pattern matching, it's usually the boundary returning nodes
    @staticmethod
    def forward(x):
        x = x + 1
        return x

    @staticmethod
    def pattern(a):
        b1 = a.relu()
        return b1

    test_cases = [
        # match_output, match_placeholder, num_matches
        TestCase(False, False, 0),
        TestCase(True, False, 0),
        TestCase(False, True, 0),
        TestCase(True, True, 0)
    ]

@instantiate_parametrized_tests
class TestFXMatcherUtils(JitTestCase):

    @parametrize("test_model", [
        SingleNodePattern,
        SimplePattern,
        SimpleFullGraphMatching,
        DiamondShapePatternTestCase,
        NonFullyContainedMatches,
        ChainRepeatedPattern,
        QuantizationModel,
        MultipleOutputsWithDependency,
        MultipleOutputsWithoutDependency,
        MultipleOutputsMultipleOverlappingMatches,
        MultipleOutputsMultipleNonOverlappingMatches,
        MultipleOutputsIdenticalAnchor,
        MultipleOutputsHorizontalPattern,
        MultiOutputWithWithInvalidMatches,
        QuantizationFp8Pattern,
        NoAnchorFound,
    ])
    def test_subgraph_matcher(self, test_model):

        setup = getattr(test_model, "setup", None)
        if callable(setup):
            setup()

        traced = symbolic_trace(test_model.forward)
        pattern_traced = symbolic_trace(test_model.pattern)

        for test_case in test_model.test_cases:

            matcher = SubgraphMatcher(pattern_traced.graph,
                                      match_output=test_case.match_output,
                                      match_placeholder=test_case.match_placeholder,
                                      remove_overlapping_matches=test_case.remove_overlapping_matches)
            matches = matcher.match(traced.graph)

            assert len(matches) == test_case.num_matches

            for match in matches:
                for node in pattern_traced.graph.nodes:
                    if not test_case.match_placeholder and node.op == "placeholder":
                        continue
                    if not test_case.match_output and node.op == "output":
                        continue
                    assert node in match.nodes_map

        tearDown = getattr(test_model, "tearDown", None)
        if callable(setup):
            tearDown()


if __name__ == "__main__":
    run_tests()
