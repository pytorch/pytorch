# Owner(s): ["module: fx"]

import itertools
import operator
import random

import torch
from torch.fx import Graph, GraphModule
from torch.fx.passes.infra._partition_graph import _PartitionGraph
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupportBase
from torch.testing._internal.common_utils import run_tests, TestCase


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


class TestPartitionGraph(TestCase):
    def assert_closure(self, helper, nodes):
        edges = _edges(nodes, helper.owner)
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
            self.assertEqual(set((helper.desc[source] & helper.alive).search(1)), seen)
        for target in edges:
            expected = {source for source, reached in closure.items() if target in reached}
            self.assertEqual(set((helper.anc[target] & helper.alive).search(1)), expected)
        helper.assert_acyclic()

    def test_direct_edge_and_external_path(self):
        gm, nodes = _make_graph([[], [0], [1]])
        helper = _PartitionGraph(gm.graph.nodes)
        a, b, c = [helper.owner[node] for node in nodes]
        self.assertFalse(helper.would_create_cycle(a, b))
        self.assertTrue(helper.would_create_cycle(a, c))
        helper.contract(a, b)
        self.assertFalse(helper.alive[b])
        self.assertFalse(helper.would_create_cycle(a, c))
        helper.contract(a, c)
        self.assert_closure(helper, tuple(gm.graph.nodes))

    def test_horizontal_merge_creates_new_paths(self):
        gm, nodes = _make_graph([[], [0], [], [2]])
        helper = _PartitionGraph(gm.graph.nodes)
        a, b, c, d = [helper.owner[node] for node in nodes]
        self.assertFalse(helper.would_create_cycle(b, c))
        helper.contract(b, c)
        self.assertTrue(helper.would_create_cycle(a, d))
        self.assert_closure(helper, tuple(gm.graph.nodes))

    def test_random_contractions_against_kahn(self):
        rng = random.Random(20261010)
        for _ in range(100):
            size = rng.randrange(2, 13)
            parents = [rng.sample(range(i), min(i, rng.randrange(3))) for i in range(size)]
            gm, _ = _make_graph(parents)
            nodes = tuple(gm.graph.nodes)
            helper = _PartitionGraph(nodes)
            runtime = _PartitionGraph(nodes, debug=False)
            while len(helper.members) > 1:
                pairs = list(itertools.combinations(helper.members, 2))
                rng.shuffle(pairs)
                accepted = None
                for a, b in pairs:
                    expected = _cyclic(_edges(nodes, helper.owner, a, b))
                    self.assertEqual(helper.would_create_cycle(a, b), expected)
                    self.assertEqual(runtime.would_create_cycle(a, b), expected)
                    if not expected and accepted is None:
                        accepted = (a, b)
                if accepted is None:
                    break
                helper.contract(*accepted)
                runtime.contract(*accepted)
                self.assert_closure(helper, nodes)
                self.assertEqual(runtime.alive, helper.alive)
                self.assertEqual(runtime.anc, helper.anc)
                self.assertEqual(runtime.desc, helper.desc)

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


if __name__ == "__main__":
    run_tests()
