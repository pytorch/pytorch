# Owner(s): ["module: fx.passes"]
# ruff: noqa: F841

from dataclasses import dataclass
import itertools
import operator
import logging
import random
import sys
from unittest.mock import patch

import torch
from torch._C import _PartitionBitset
from torch.fx import Graph, GraphModule
from torch.fx._symbolic_trace import symbolic_trace

from torch.fx.passes.infra._partition_graph import _PartitionGraph
from torch.fx.passes.infra.partitioner import _DependencyViewer, CapabilityBasedPartitioner
from torch.fx.passes.operator_support import (
    chain,
    create_op_support,
    OperatorSupport,
    OperatorSupportBase,
    OpSupports,
)
from torch.fx.passes.utils.fuser_utils import fuse_by_partitions, topo_sort
from torch.fx.passes.utils.matcher_utils import SubgraphMatcher

from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase as TorchTestCase,
)
from torch.testing._internal.jit_utils import JitTestCase

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class TestModule(torch.nn.Module):
    def __init__(self) -> None:
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

class TestDeepModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, a, b, c):
        o = a + b
        o = o + 1.0

        # testing to avoid DFS uses in passes. Since Python has max recursion depth.
        for _ in range(sys.getrecursionlimit() + 1):
            o = o - c

        return o


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
        # add should have its own partition, as neither branches are supported
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

    @staticmethod
    def forward17(a, b, c, d, e, f):
        a0 = a + b
        a1 = c + d
        a2 = e + f
        return a0, a1, a2

    @staticmethod
    def forward18(a, b, c):
        a0, a1 = torch.ops.aten.var_mean(a)
        return a0

    @staticmethod
    def forward19(a, b, c):
        add = a + b
        add_1 = add + c
        add_2 = add_1 + 1
        return add_2

    @staticmethod
    def forward20(a, b, c):
        relu = a.relu()
        add = relu + b
        add_1 = relu + c
        return add, add_1

    @staticmethod
    def forward21(a, b, c):
        add = a + b
        add_1 = c + c
        relu = add_1.relu()
        add_2 = add + relu
        add_3 = add + add_1
        return add_2, add_3

@torch.fx.wrap
def _nested_tuple_producer(x):
    """Returns a nested tuple structure: (x+1, (x+2, x+3))"""
    return (x + 1, (x + 2, x + 3))


class TestNestedGetitemFunctions:
    """Test functions for nested getitem chain handling in partitioner."""

    @staticmethod
    def forward_nested_getitem_cross_partition(a, b, c):
        result = _nested_tuple_producer(a)
        outer_0 = result[0]
        inner_tuple = result[1]
        inner_0 = inner_tuple[0]
        # UNSUPPORTED operation (relu) - this creates a partition boundary
        # the getitem chain should still be reassigned to producer's partition
        blocked = outer_0.relu()
        # downstream use that crosses partition boundary
        return inner_0 + blocked + b


# A mock OperatorSupport class, where only operator.add is supported
class MockOperatorSupport(OperatorSupport):
    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
        return (node.op == "call_function" and
                node.target in {operator.add, operator.getitem,
                                torch.ops.aten.view,
                                torch.ops.aten.permute,
                                torch.ops.aten.std_mean,
                                _nested_tuple_producer})

class _Support(OperatorSupportBase):
    def is_node_supported(self, submodules, node):
        return node.meta.get("supported", False)


def _make_graph(parents):
    graph = Graph()
    x = graph.placeholder("x")
    nodes = []
    for predecessors in parents:
        args = tuple(nodes[index] for index in predecessors)
        node = graph.call_function(torch.add, (args[0], args[-1]) if args else (x, x))
        node.meta["supported"] = True
        nodes.append(node)
    graph.output(tuple(nodes))
    return GraphModule(torch.nn.Module(), graph), nodes


def _edges(nodes, owner, keep=None, drop=None):
    mapping = {node: keep if pid == drop else pid for node, pid in owner.items()}
    edges = {pid: set() for pid in mapping.values()}
    for node in nodes:
        for user in node.users:
            if mapping[node] != mapping[user]:
                edges[mapping[node]].add(mapping[user])
    return edges


def _cyclic(edges):
    degrees = dict.fromkeys(edges, 0)
    for targets in edges.values():
        for target in targets:
            degrees[target] += 1
    ready = [pid for pid, count in degrees.items() if count == 0]
    visited = 0
    while ready:
        pid = ready.pop()
        visited += 1
        for target in edges[pid]:
            degrees[target] -= 1
            if degrees[target] == 0:
                ready.append(target)
    return visited != len(edges)


class _PartitionDebugState:
    def __init__(self, initial_owner: dict[torch.fx.Node, int]) -> None:
        self.owner = dict(initial_owner)
        self.members: dict[int, set[torch.fx.Node]] = {}
        for node, pid in self.owner.items():
            self.members.setdefault(pid, set()).add(node)

    def contract(self, keep: int, drop: int) -> None:
        for node in self.members[drop]:
            self.owner[node] = keep
        self.members[keep].update(self.members.pop(drop))


@instantiate_parametrized_tests
class TestPartitionBitset(TorchTestCase):
    def make_bits(self, width, indices):
        bits = _PartitionBitset(width)
        for index in indices:
            bits[index] = True
        return bits

    @parametrize("width", [0, 1, 63, 64, 65, 127, 128, 129])
    def test_fill_and_tail(self, width):
        empty = _PartitionBitset(width)
        full = _PartitionBitset(width, fill=True)
        self.assertEqual(len(empty), width)
        self.assertEqual(list(empty.set_bits()), [])
        self.assertEqual(list(full.set_bits()), list(range(width)))
        self.assertEqual(~empty, full)
        self.assertEqual(~full, empty)
        self.assertEqual(list((~empty).set_bits()), list(range(width)))
        self.assertFalse(empty.intersects(full))
        self.assertEqual(full.intersects(full), width > 0)

    def test_indexing_and_copy(self):
        bits = self.make_bits(130, [0, 63, 64, 65, 127, 128, 129])
        copied = bits.copy()
        copied[64] = False
        copied[1] = True
        self.assertTrue(bits[64])
        self.assertFalse(bits[1])
        bits[129] = False
        self.assertTrue(copied[129])
        self.assertEqual(list(bits.set_bits()), [0, 63, 64, 65, 127, 128])
        self.assertEqual(list(copied.set_bits()), [0, 1, 63, 65, 127, 128, 129])

    @parametrize("width", [0, 1, 63, 64, 65, 127, 128, 129, 1000])
    def test_operations_against_sets(self, width):
        rng = random.Random(20261019 + width)
        for _ in range(10):
            left, right, mask = [
                {i for i in range(width) if rng.random() < 0.4} for _ in range(3)
            ]
            a, b, active = [
                self.make_bits(width, indices) for indices in (left, right, mask)
            ]
            self.assertEqual(set((a & b).set_bits()), left & right)
            self.assertEqual(set((a | b).set_bits()), left | right)
            self.assertEqual(set(a.difference(b).set_bits()), left - right)
            self.assertEqual(set((~a).set_bits()), set(range(width)) - left)
            self.assertEqual(a.intersects(b), bool(left & right))
            self.assertEqual(a.intersects(b, active), bool(left & right & mask))
            self.assertEqual(set(a.set_bits()), left)
            self.assertEqual(set(b.set_bits()), right)
            result = a.copy()
            alias = result
            result |= b
            self.assertIs(result, alias)
            self.assertEqual(set(result.set_bits()), left | right)
            result |= result
            self.assertEqual(set(result.set_bits()), left | right)
            self.assertEqual(a, a.copy())
            self.assertNotEqual(a, _PartitionBitset(width + 1))

    def test_mask_excludes_inactive_bit(self):
        a = self.make_bits(65, [64])
        b = a.copy()
        alive = _PartitionBitset(65, fill=True)
        self.assertTrue(a.intersects(b, alive))
        alive[64] = False
        self.assertFalse(a.intersects(b, alive))

    def test_iterator_keeps_temporary_alive(self):
        iterator = (
            _PartitionBitset(129, fill=True) & ~_PartitionBitset(129)
        ).set_bits()
        self.assertIs(iter(iterator), iterator)
        self.assertEqual(next(iterator), 0)
        self.assertEqual(list(iterator), list(range(1, 129)))
        for _ in range(2):
            with self.assertRaises(StopIteration):
                next(iterator)
        bits = self.make_bits(130, [0, 64, 129])
        first, second = bits.set_bits(), bits.set_bits()
        del bits
        self.assertEqual(next(first), 0)
        self.assertEqual(list(second), [0, 64, 129])
        self.assertEqual(list(first), [64, 129])

    def test_invalid_sizes_and_indices(self):
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            _PartitionBitset(-1)
        for width in (0, 1, 65):
            bits = _PartitionBitset(width)
            for index in (-1, width, width + 64):
                with self.assertRaisesRegex(IndexError, "out of range"):
                    bits[index]
                with self.assertRaisesRegex(IndexError, "out of range"):
                    bits[index] = True

    def test_intersects_arguments(self):
        a = self.make_bits(65, [64])
        b = a.copy()
        active = _PartitionBitset(65)
        self.assertTrue(a.intersects(other=b))
        self.assertTrue(a.intersects(b, mask=None))
        self.assertFalse(a.intersects(other=b, mask=active))
        self.assertFalse(a.intersects(mask=active, other=b))
        self.assertFalse(a.intersects(b, mask=active))
        for args, kwargs in [
            ((), {}),
            ((b, active, active), {}),
            ((b,), {"other": b}),
            ((b, active), {"mask": active}),
            ((b,), {"unexpected": active}),
            ((None,), {}),
            ((b,), {"mask": object()}),
        ]:
            with self.assertRaises(TypeError):
                a.intersects(*args, **kwargs)

    def test_foreign_operands(self):
        bits = _PartitionBitset(65)
        for other in (None, 1, object()):
            for operation in (operator.and_, operator.or_, operator.ior):
                with self.assertRaises(TypeError):
                    operation(bits, other)
                with self.assertRaises(TypeError):
                    operation(other, bits)
            with self.assertRaises(TypeError):
                bits.difference(other)
            self.assertFalse(bits == other)
            self.assertTrue(bits != other)
        with self.assertRaises(TypeError):
            hash(bits)
        with self.assertRaises(TypeError):
            operator.lt(bits, bits)

    def test_iterator_releases_owner(self):
        bits = self.make_bits(65, [0, 64])
        references = sys.getrefcount(bits)
        iterator = bits.set_bits()
        self.assertEqual(sys.getrefcount(bits), references + 1)
        self.assertEqual(list(iterator), [0, 64])
        self.assertIsNone(next(iterator, None))
        self.assertIsNone(next(iterator, None))
        del iterator
        self.assertEqual(sys.getrefcount(bits), references)
        with self.assertRaises(TypeError):
            type(bits.set_bits())()

    def test_argument_conversion_errors(self):
        class BadIndex:
            def __index__(self):
                raise RuntimeError("bad index")

        class BadBool:
            def __bool__(self):
                raise RuntimeError("bad bool")

        bits = _PartitionBitset(size=65, fill=True)
        with self.assertRaisesRegex(RuntimeError, "bad bool"):
            _PartitionBitset(65, fill=BadBool())
        with self.assertRaisesRegex(RuntimeError, "bad index"):
            _PartitionBitset(BadIndex())
        with self.assertRaises(OverflowError):
            _PartitionBitset(1 << 100)
        with self.assertRaisesRegex(RuntimeError, "bad index"):
            bits[BadIndex()]
        with self.assertRaisesRegex(RuntimeError, "bad bool"):
            bits[0] = BadBool()
        self.assertTrue(bits[0])
        with self.assertRaises(TypeError):
            del bits[0]
        with self.assertRaises(TypeError):
            bits[0.5]
        with self.assertRaises(IndexError):
            bits[1 << 100]

    @parametrize("widths", [(0, 1), (1, 2), (64, 65), (65, 66)])
    def test_mismatched_widths(self, widths):
        a, b = [_PartitionBitset(width, fill=True) for width in widths]
        for operation in (operator.and_, operator.or_, operator.ior):
            with self.assertRaisesRegex(ValueError, "same size"):
                operation(a, b)
        with self.assertRaisesRegex(ValueError, "same size"):
            a.difference(b)
        with self.assertRaisesRegex(ValueError, "same size"):
            a.intersects(b)
        with self.assertRaisesRegex(ValueError, "same size"):
            a.intersects(a, b)


class TestPartitionGraph(TorchTestCase):
    def assert_closure(self, helper, nodes, state):
        edges = _edges(nodes, state.owner)
        self.assertFalse(_cyclic(edges))
        self.assertEqual(set(helper.alive.set_bits()), set(state.members))
        self.assertEqual(set(helper.ancestors), set(state.members))
        self.assertEqual(set(helper.descendants), set(state.members))
        closure = {}
        for source in edges:
            seen = set()
            stack = list(edges[source])
            while stack:
                target = stack.pop()
                if target not in seen:
                    seen.add(target)
                    stack.extend(edges[target])
            closure[source] = seen
            self.assertEqual(
                set((helper.descendants[source] & helper.alive).set_bits()), seen
            )
        for target in edges:
            expected = {
                source for source, reached in closure.items() if target in reached
            }
            self.assertEqual(
                set((helper.ancestors[target] & helper.alive).set_bits()), expected
            )

    def test_direct_edge_and_external_path(self):
        gm, nodes = _make_graph([[], [0], [1]])
        helper = _PartitionGraph(gm.graph.nodes)
        state = _PartitionDebugState(helper.initial_owner)
        a, b, c = [state.owner[node] for node in nodes]
        self.assertFalse(helper.would_create_cycle(a, b))
        self.assertTrue(helper.would_create_cycle(a, c))
        helper.contract(a, b)
        state.contract(a, b)
        self.assertFalse(helper.alive[b])
        self.assertFalse(helper.would_create_cycle(a, c))
        helper.contract(a, c)
        state.contract(a, c)
        self.assert_closure(helper, tuple(gm.graph.nodes), state)

    def test_horizontal_merge_creates_new_paths(self):
        gm, nodes = _make_graph([[], [0], [], [2]])
        helper = _PartitionGraph(gm.graph.nodes)
        state = _PartitionDebugState(helper.initial_owner)
        a, b, c, d = [state.owner[node] for node in nodes]
        self.assertFalse(helper.would_create_cycle(b, c))
        helper.contract(b, c)
        state.contract(b, c)
        self.assertTrue(helper.would_create_cycle(a, d))
        self.assert_closure(helper, tuple(gm.graph.nodes), state)

    def test_contractions_across_words(self):
        gm, nodes = _make_graph([[]] + [[i - 1] for i in range(1, 130)])
        helper = _PartitionGraph(gm.graph.nodes)
        state = _PartitionDebugState(helper.initial_owner)
        keep = state.owner[nodes[0]]
        last = state.owner[nodes[-1]]
        self.assertTrue(helper.would_create_cycle(keep, last))
        for node in nodes[1:]:
            drop = state.owner[node]
            self.assertFalse(helper.would_create_cycle(keep, drop))
            helper.contract(keep, drop)
            state.contract(keep, drop)
            self.assertFalse(helper.alive[drop])
        self.assertEqual(state.members[keep], set(nodes))
        self.assert_closure(helper, tuple(gm.graph.nodes), state)

    def test_random_contractions_against_kahn(self):
        rng = random.Random(20261010)
        for _ in range(100):
            size = rng.randrange(2, 13)
            parents = [
                rng.sample(range(i), min(i, rng.randrange(3))) for i in range(size)
            ]
            gm, _ = _make_graph(parents)
            nodes = tuple(gm.graph.nodes)
            helper = _PartitionGraph(nodes)
            state = _PartitionDebugState(helper.initial_owner)
            self.assert_closure(helper, nodes, state)
            while len(state.members) > 1:
                pairs = list(itertools.combinations(state.members, 2))
                rng.shuffle(pairs)
                accepted = None
                for a, b in pairs:
                    expected = _cyclic(_edges(nodes, state.owner, a, b))
                    self.assertEqual(helper.would_create_cycle(a, b), expected)
                    if not expected and accepted is None:
                        accepted = (a, b)
                if accepted is None:
                    break
                helper.contract(*accepted)
                state.contract(*accepted)
                self.assert_closure(helper, nodes, state)

    def test_unsupported_node_blocks_merge(self):
        gm, nodes = _make_graph([[], [0], [1]])
        nodes[1].meta["supported"] = False
        partitions = CapabilityBasedPartitioner(
            gm, _Support(), allows_single_node_partition=True
        ).propose_partitions()
        self.assertEqual(
            {frozenset(p.nodes) for p in partitions},
            {frozenset([nodes[0]]), frozenset([nodes[2]])},
        )

    def test_random_partition_graph_is_acyclic(self):
        rng = random.Random(20261010)
        for _ in range(100):
            graph = Graph()
            nodes = [graph.placeholder("x")]
            for i in range(rng.randrange(5, 35)):
                node = graph.call_function(
                    operator.add, (rng.choice(nodes), rng.choice(nodes))
                )
                node.meta["supported"] = rng.random() < 0.7
                nodes.append(node)
            graph.output(tuple(nodes[1:]))
            gm = GraphModule(torch.nn.Module(), graph)
            for skip_horizontal in (True, False):
                for single in (True, False):
                    partitions = CapabilityBasedPartitioner(
                        gm,
                        _Support(),
                        allows_single_node_partition=single,
                        skip_horizontal_fusion=skip_horizontal,
                    ).propose_partitions()
                    owner = {node: i for i, node in enumerate(graph.nodes)}
                    for i, partition in enumerate(partitions):
                        for node in partition.nodes:
                            owner[node] = -i - 1
                    self.assertFalse(_cyclic(_edges(tuple(graph.nodes), owner)))

    def test_nested_getitem_postprocessing(self):
        graph = Graph()
        x = graph.placeholder("x")
        producer = graph.call_function(torch.max, (x, 0))
        first = graph.call_function(operator.getitem, (producer, 0))
        second = graph.call_function(operator.getitem, (first, 0))
        graph.output(second)
        producer.meta["supported"] = True
        gm = GraphModule(torch.nn.Module(), graph)
        partitions = CapabilityBasedPartitioner(
            gm, _Support(), allows_single_node_partition=True
        ).propose_partitions()
        self.assertEqual(len(partitions), 1)
        self.assertEqual(set(partitions[0].nodes), {producer, first, second})


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
        coefficients.append(
            sum(1 if i == -1 else coefficients[i] for i in predecessors)
        )
    graph.output(tuple(nodes))
    return GraphModule(torch.nn.Module(), graph), coefficients


@instantiate_parametrized_tests
class TestCapabilityPartitionerRegressions(TorchTestCase):
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
        self.assertEqual(
            [set(partition.nodes) for partition in partitions], [{add, mul}]
        )
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



@instantiate_parametrized_tests
class TestFXGraphPasses(JitTestCase):
    hw_classification = HardwareClassification.GENERIC

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
        (TestPartitionFunctions.forward12, [["add_2", "add_3", "add_4"], ["add", "add_1"]], False),

        # 5 getitem special case
        (TestPartitionFunctions.forward13, [["add_2", "add_1", "add"]], False),
        (TestPartitionFunctions.forward14, [["add", "std_mean", "getitem", "getitem_1"]], False),

        # 6 bookend non_compute pass
        (TestPartitionFunctions.forward15, [["permute_1", "add_1", "add"]], True),
        (TestPartitionFunctions.forward15, [['add_1', 'add', 'permute_1', 'view', 'permute_2', 'permute_3', 'permute']], False),
        (TestPartitionFunctions.forward16, [["permute_1", "add_1", "add"]], True),
        (TestPartitionFunctions.forward16, [['add_1', 'add', 'permute_1', 'view', 'permute_2', 'permute_3', 'permute']], False),
        # should be empty partition, not a partition with empty nodes
        (TestPartitionFunctions.forward18, [], False),
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
        if len(partitions_name) != len(expected_partition):
            raise AssertionError(f"partition count mismatch: {len(partitions_name)} != {len(expected_partition)}")
        for i in range(len(partitions_name)):
            if set(partitions_name[i]) != set(expected_partition[i]):
                raise AssertionError(f"partition {i} mismatch: {set(partitions_name[i])} != {set(expected_partition[i])}")

        fused_graph = partitioner.fuse_partitions(partitions)

        a, b, c = torch.rand(4), torch.rand(4), torch.rand(4)

        expected = fn(a, b, c)
        result = fused_graph(a, b, c)
        torch.testing.assert_close(expected, result)

    @parametrize("fn, expected_partition", [
        (TestPartitionFunctions.forward17, [['add', 'add_1', 'add_2']]),
    ])
    def test_partitioner_independent_output(self, fn, expected_partition):
        traced = symbolic_trace(fn)

        supported_ops = MockOperatorSupport()
        partitioner = CapabilityBasedPartitioner(traced,
                                                 supported_ops,
                                                 allows_single_node_partition=True)
        partitions = partitioner.propose_partitions()
        partitions_name = [[node.name for node in partition.nodes] for partition in partitions]
        if len(partitions_name) != len(expected_partition):
            raise AssertionError(f"partition count mismatch: {len(partitions_name)} != {len(expected_partition)}")
        for i in range(len(partitions_name)):
            if set(partitions_name[i]) != set(expected_partition[i]):
                raise AssertionError(f"partition {i} mismatch: {set(partitions_name[i])} != {set(expected_partition[i])}")

        fused_graph = partitioner.fuse_partitions(partitions)

        a, b, c, d, e, f = torch.rand(4), torch.rand(4), torch.rand(4), torch.rand(4), torch.rand(4), torch.rand(4)

        expected = fn(a, b, c, d, e, f)
        result = fused_graph(a, b, c, d, e, f)
        torch.testing.assert_close(expected, result)

    @parametrize("fn, expected_partition", [
        # Independent supported outputs are horizontal fusion and stay separate.
        (TestPartitionFunctions.forward17, [["add"], ["add_1"], ["add_2"]]),
        # Data-dependent chains still fuse vertically.
        (TestPartitionFunctions.forward19, [["add", "add_1", "add_2"]]),
        # Consumers of an unsupported producer are not fused horizontally.
        (TestPartitionFunctions.forward20, [["add"], ["add_1"]]),
        # Reachability through unsupported nodes still prevents cyclic partitions.
        (
            TestPartitionFunctions.forward12,
            [["add"], ["add_1", "add_3", "add_4"], ["add_2"]],
        ),
        # Do not leave two vertical partitions with cross-partition cycles.
        (TestPartitionFunctions.forward21, [["add_2"], ["add", "add_1", "add_3"]]),
        # Tuple producer/getitem chains still fuse with their data-dependent user.
        (TestPartitionFunctions.forward14, [["std_mean", "getitem", "getitem_1", "add"]]),
    ])
    def test_partitioner_skip_horizontal_fusion(self, fn, expected_partition):
        traced = symbolic_trace(fn)

        supported_ops = MockOperatorSupport()
        partitioner = CapabilityBasedPartitioner(
            traced,
            supported_ops,
            allows_single_node_partition=True,
            skip_horizontal_fusion=True,
        )
        partitions = partitioner.propose_partitions()
        partitions_name = [
            frozenset(node.name for node in partition.nodes)
            for partition in partitions
        ]
        expected = [frozenset(partition) for partition in expected_partition]
        self.assertEqual(set(partitions_name), set(expected))

        fused_graph = partitioner.fuse_partitions(partitions)

        args = [torch.rand(4) for _ in range(fn.__code__.co_argcount)]
        expected_result = fn(*args)
        result = fused_graph(*args)
        torch.testing.assert_close(expected_result, result)

    @parametrize("partition", [
        [['add', 'add_1'], ['add_5', 'add_6']],
        [['add', 'add_1', 'add_2']],  # vertical fusion
        [['add_2', 'add_3']],         # horizontal fusion
        [['add_3', 'add_4']],
        [['add_6', 'add_5']],     # arbitrary node order
        [['add_4', 'add_1', 'add_3', 'add_2']],           # arbitrary node order
        [['add_5', 'add_6'], ['add_1', 'add_2', 'add_3', 'add_4']],  # arbitrary partition order
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
            partitions.append(dict.fromkeys([nodes_by_name[name] for name in node_names]))

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
            partitions.append(dict.fromkeys([nodes_by_name[name] for name in node_names]))

        with self.assertRaises(Exception):
            fuse_by_partitions(gm, partitions)

    def test_fuser_pass_deep_model(self):
        m = TestDeepModule()
        traced = symbolic_trace(m)

        supported_ops = MockOperatorSupport()
        partitioner = CapabilityBasedPartitioner(traced,
                                                 supported_ops,
                                                 allows_single_node_partition=True)
        partitions = partitioner.propose_partitions()

    def test_topo_sort_stability(self):
        """Test that topo_sort preserves relative order of independent nodes."""
        m = TestModule()
        gm = symbolic_trace(m)

        nodes_by_name = {node.name: node for node in gm.graph.nodes}

        # add_2 and linear depend on add_1 but are independent of each other.
        # add_3 depends on both add_2 and linear (via add_1 -> linear -> add_3
        # and add_1 -> add_2 -> add_3 paths in TestModule).
        # Provide them in a specific order and verify it is preserved.
        node_list = [
            nodes_by_name["add_1"],
            nodes_by_name["add_2"],
            nodes_by_name["linear"],
            nodes_by_name["add_3"],
        ]
        sorted_nodes = topo_sort(node_list)
        sorted_names = [n.name for n in sorted_nodes]

        # add_1 must come first (dependency of add_2, linear, add_3)
        self.assertEqual(sorted_names[0], "add_1")
        # add_2 and linear are independent; stable sort preserves input order
        self.assertEqual(sorted_names[1], "add_2")
        self.assertEqual(sorted_names[2], "linear")
        # add_3 depends on both, so it comes last
        self.assertEqual(sorted_names[3], "add_3")

        # Now reverse the order of the independent nodes and verify stability
        # is with respect to the new input order.
        node_list_reversed = [
            nodes_by_name["add_1"],
            nodes_by_name["linear"],
            nodes_by_name["add_2"],
            nodes_by_name["add_3"],
        ]
        sorted_nodes_reversed = topo_sort(node_list_reversed)
        sorted_names_reversed = [n.name for n in sorted_nodes_reversed]

        self.assertEqual(sorted_names_reversed[0], "add_1")
        # Now linear should come before add_2 since that's the input order
        self.assertEqual(sorted_names_reversed[1], "linear")
        self.assertEqual(sorted_names_reversed[2], "add_2")
        self.assertEqual(sorted_names_reversed[3], "add_3")

    def test_fuse_preserves_original_placeholder_input_order(self):
        class M(torch.nn.Module):
            def forward(self, input_ids, attention_mask):
                mask = attention_mask + 1
                ids = input_ids * 2
                return mask + ids

        gm = symbolic_trace(M())
        partition = {
            node: None
            for node in gm.graph.nodes
            if node.op not in ("placeholder", "output")
        }

        fused_graph = fuse_by_partitions(gm, [partition])
        fused_submodule = fused_graph.get_submodule("fused_0")
        submodule_inputs = [
            node.name for node in fused_submodule.graph.nodes if node.op == "placeholder"
        ]
        self.assertEqual(submodule_inputs, ["input_ids", "attention_mask"])

        fused_node = next(
            node for node in fused_graph.graph.nodes if node.op == "call_module"
        )
        self.assertEqual([node.name for node in fused_node.args], submodule_inputs)

        input_ids, attention_mask = torch.rand(4), torch.rand(4)
        torch.testing.assert_close(
            M()(input_ids, attention_mask),
            fused_graph(input_ids, attention_mask),
        )

    def test_fuse_preserves_original_placeholder_input_order_with_internal_deps(self):
        class M(torch.nn.Module):
            def forward(self, x, y, z):
                a = x + z
                return a * y

        gm = symbolic_trace(M())
        partition = {
            node: None
            for node in gm.graph.nodes
            if node.op not in ("placeholder", "output")
        }

        fused_graph = fuse_by_partitions(gm, [partition])
        fused_submodule = fused_graph.get_submodule("fused_0")
        submodule_inputs = [
            node.name for node in fused_submodule.graph.nodes if node.op == "placeholder"
        ]
        self.assertEqual(submodule_inputs, ["x", "y", "z"])

        fused_node = next(
            node for node in fused_graph.graph.nodes if node.op == "call_module"
        )
        self.assertEqual([node.name for node in fused_node.args], submodule_inputs)

        x, y, z = torch.rand(4), torch.rand(4), torch.rand(4)
        torch.testing.assert_close(M()(x, y, z), fused_graph(x, y, z))

    def test_fuse_preserves_intermediate_input_encounter_order(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                a = x + 1
                b = y + 1
                c = b * a
                return c + 1

        gm = symbolic_trace(M())
        nodes_by_name = {node.name: node for node in gm.graph.nodes}
        partition = {
            nodes_by_name["mul"]: None,
            nodes_by_name["add_2"]: None,
        }

        fused_graph = fuse_by_partitions(gm, [partition])
        fused_submodule = fused_graph.get_submodule("fused_0")
        submodule_inputs = [
            node.name for node in fused_submodule.graph.nodes if node.op == "placeholder"
        ]
        self.assertEqual(submodule_inputs, ["add_1", "add"])

        fused_node = next(
            node for node in fused_graph.graph.nodes if node.op == "call_module"
        )
        self.assertEqual([node.name for node in fused_node.args], submodule_inputs)

        x, y = torch.rand(4), torch.rand(4)
        torch.testing.assert_close(M()(x, y), fused_graph(x, y))

    def test_fuse_preserves_mixed_input_encounter_order(self):
        class M(torch.nn.Module):
            def forward(self, a, x, b):
                y = torch.mm(a, x)
                z = y + b
                a = z - a
                y = torch.mm(a, x)
                return y + b

        gm = symbolic_trace(M())
        nodes_by_name = {node.name: node for node in gm.graph.nodes}
        partition = {
            nodes_by_name["mm_1"]: None,
            nodes_by_name["add_1"]: None,
        }

        fused_graph = fuse_by_partitions(gm, [partition])
        fused_submodule = fused_graph.get_submodule("fused_0")
        submodule_inputs = [
            node.name for node in fused_submodule.graph.nodes if node.op == "placeholder"
        ]
        self.assertEqual(submodule_inputs, ["sub", "x", "b"])

        fused_node = next(
            node for node in fused_graph.graph.nodes if node.op == "call_module"
        )
        self.assertEqual([node.name for node in fused_node.args], submodule_inputs)

        a, x, b = torch.rand(4, 4), torch.rand(4, 4), torch.rand(4, 4)
        torch.testing.assert_close(M()(a, x, b), fused_graph(a, x, b))

    def test_partitioner_nested_getitem_chains(self):
        """Test that nested getitem chains are properly reassigned to producer's partition.

        this tests the fix for handling patterns like:
            producer_node = call(...)  # returns ((a, b), c)
            getitem_1 = producer_node[0]  # gets (a, b)
            getitem_2 = getitem_1[0]  # gets a

        without the iterative fix, getitem_2 would be assigned to getitem_1's original
        partition instead of the producer's partition.
        """
        traced = symbolic_trace(TestNestedGetitemFunctions.forward_nested_getitem_cross_partition)

        supported_ops = MockOperatorSupport()
        partitioner = CapabilityBasedPartitioner(
            traced,
            supported_ops,
            allows_single_node_partition=True
        )
        partitions = partitioner.propose_partitions()

        node_to_partition: dict[torch.fx.Node, int] = {}
        for idx, partition in enumerate(partitions):
            for node in partition.nodes:
                node_to_partition[node] = idx

        producer_node = None
        getitem_nodes = []
        for node in traced.graph.nodes:
            if node.op == "call_function":
                if node.target == _nested_tuple_producer:
                    producer_node = node
                elif node.target == operator.getitem:
                    getitem_nodes.append(node)

        self.assertIsNotNone(producer_node, "Should find the nested tuple producer node")
        self.assertGreaterEqual(len(getitem_nodes), 3, "Should find at least 3 getitem nodes")

        # all getitems that derive from the producer (including nested ones)
        # should be in the same partition as the producer
        if producer_node in node_to_partition:
            producer_partition = node_to_partition[producer_node]

            for getitem_node in getitem_nodes:
                current = getitem_node.args[0]
                derives_from_producer = False
                while hasattr(current, 'op'):
                    if current == producer_node:
                        derives_from_producer = True
                        break
                    if current.op == "call_function" and current.target == operator.getitem:
                        current = current.args[0]
                    else:
                        break

                if derives_from_producer:
                    getitem_partition = node_to_partition.get(getitem_node)
                    self.assertEqual(
                        getitem_partition,
                        producer_partition,
                        lambda msg: f"{msg}\nGetitem node '{getitem_node.name}' (nested in chain from producer) "
                        f"should be in same partition as producer. "
                        f"Got partition {getitem_partition}, expected {producer_partition}"
                    )

        fused_graph = partitioner.fuse_partitions(partitions)
        a, b, c = torch.rand(4), torch.rand(4), torch.rand(4)
        expected = TestNestedGetitemFunctions.forward_nested_getitem_cross_partition(a, b, c)
        result = fused_graph(a, b, c)
        torch.testing.assert_close(expected, result)

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
        cls.quantization = torch.library.Library("fp8_quantization", "DEF")  # noqa: SCOPED_LIBRARY
        cls.quantization.define("quantize_per_tensor_affine_fp8(Tensor self, int dtype, float scale) -> Tensor")
        cls.quantization.define("dequantize_per_tensor_affine_fp8(Tensor self, int dtype, float scale) -> Tensor")

    @classmethod
    def tearDown(cls):
        cls.quantization._destroy()

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
    hw_classification = HardwareClassification.GENERIC

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

            if len(matches) != test_case.num_matches:
                raise AssertionError(f"match count mismatch: {len(matches)} != {test_case.num_matches}")

            for match in matches:
                for node in pattern_traced.graph.nodes:
                    if not test_case.match_placeholder and node.op == "placeholder":
                        continue
                    if not test_case.match_output and node.op == "output":
                        continue
                    if node not in match.nodes_map:
                        raise AssertionError(f"node {node} not in match.nodes_map")

        tearDown = getattr(test_model, "tearDown", None)
        if callable(setup):
            tearDown()


if __name__ == "__main__":
    run_tests()
