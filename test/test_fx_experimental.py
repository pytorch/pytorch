import torch
import unittest
from typing import Dict
from torch.fx.symbolic_trace import symbolic_trace
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node
from torch.fx.experimental import GraphManipulation
from torch.fx.experimental.Partitioner import Partitioner
from torch.fx.experimental.rewriter import RewritingTracer
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.jit_utils import JitTestCase
from torch.fx.experimental.subgraph_creation_example import split_module
from torch.fx.experimental.partitioner_utils import (
    NodeLatency,
    get_partition_to_latency_mapping,
    get_latency_of_partitioned_graph,
    Device,
    PartitionerConfig
)
from typing import Union, Callable

try:
    from torchvision.models import resnet18
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")


def symbolic_trace_with_rewrite(root: Union[torch.nn.Module, Callable]) -> GraphModule:
    return GraphModule(
        root if isinstance(root, torch.nn.Module) else torch.nn.Module(),
        RewritingTracer().trace(root),
    )


class TestFXExperimental(JitTestCase):
    def test_serialize_graph(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
                self.e = torch.rand(4)
                self.conv = torch.nn.Conv2d(3, 3, 2, bias=False)

            def forward(self, a, b, c):
                add_1 = a + b
                conv1 = self.conv(c)
                linear = self.linear(add_1 + conv1)
                add_2 = linear + self.e
                return add_2

        m = TestModule()
        traced = symbolic_trace(m)
        a = torch.rand(4)
        b = torch.rand(4)
        c = torch.rand(3, 3, 2, 2)
        GraphManipulation.get_size_of_all_nodes(traced, [a, b, c])

        partitioner = Partitioner()
        devices = [Device("dev_0", 5000, 0), Device("dev_1", 125, 1)]
        partitioner_config = PartitionerConfig(devices, is_sparse_nn=True)
        ret = partitioner.partition_graph(traced, m, partitioner_config)
        module_with_submodules = ret.module_with_submodules
        # Fix for now to add type/shape to output
        for node in traced.graph.nodes:
            if node.op == "output":
                node.shape = a.shape
                node.dtype = a.dtype
        for mod in module_with_submodules.modules():
            if isinstance(mod, GraphModule):
                for node in mod.graph.nodes:
                    node.shape = a.shape
                    node.dtype = a.dtype
        for node in module_with_submodules.graph.nodes:
            node.shape = a.shape
            node.dtype = a.dtype

        agm1 = GraphManipulation.AcceleratedGraphModule(traced)
        agm2 = GraphManipulation.AcceleratedGraphModule(module_with_submodules)
        assert len(agm1.weights) == 4
        assert len(agm2.weights) == 4
        assert len(agm1.serialized_graph["nodes"]) == 10
        assert len(agm1.serialized_graph["weights"]) == 4
        assert len(agm1.serialized_graph["modules"]) == 0
        assert len(agm2.serialized_graph["nodes"]) == 6
        assert len(agm2.serialized_graph["weights"]) == 4
        assert len(agm2.serialized_graph["modules"]) == 1
        assert agm1.serialized_graph["weights"]["linear.weight"]["shape"] == "[4, 4]"
        assert (
            agm1.serialized_graph["weights"]["linear.weight"]["dtype"]
            == "torch.float32"
        )
        assert (
            agm1.serialized_graph["weights"]["linear.weight"]["is_quantized"] is False
        )
        assert agm1.serialized_graph["nodes"][0]["shape"] == "[4]"
        assert agm1.serialized_graph["nodes"][0]["dtype"] == "torch.float32"
        assert agm1.serialized_graph["nodes"][0]["target"] == "a"
        assert agm1.serialized_graph["nodes"][0]["op_code"] == "placeholder"
        assert agm1.serialized_graph["nodes"][0]["name"] == "a"
        assert agm1.serialized_graph["nodes"][6]["args"][0]["name"] == "add_2"
        assert agm1.serialized_graph["nodes"][6]["args"][0]["is_node"] is True

        # Test quantization info serialization.
        x = torch.tensor([[-1.0, 0.0], [1.0, 2.0]])
        q_tensor = torch.quantize_per_tensor(x, 1, 0, torch.qint32)
        q_tensor_channel = torch.quantize_per_channel(
            x, torch.tensor([0.1, 0.01]), torch.tensor([10, 0]), 0, torch.quint8
        )
        result = GraphManipulation.serialize_tensor_quantization(q_tensor)
        result2 = GraphManipulation.serialize_tensor_quantization(q_tensor_channel)
        assert result["q_scheme"] == "torch.per_tensor_affine"
        assert result["q_scale"] == 1.0
        assert result2["q_scheme"] == "torch.per_channel_affine"
        assert len(result2["q_per_channel_scales"]) == 2

    def test_find_single_partition(self):
        class TestModule(torch.nn.Module):
            def forward(self, a, b):
                return a + b

        m = TestModule()
        traced = symbolic_trace(m)
        a = torch.rand(1)
        b = torch.rand(1)
        GraphManipulation.get_size_of_all_nodes(traced, [a, b])
        partitioner = Partitioner()
        devices = [
            Device("dev_0", 125, 0),
            Device("dev_1", 125, 1),
            Device("dev_2", 125, 2),
        ]
        partitioner_config = PartitionerConfig(devices)
        ret = partitioner.partition_graph(traced, m, partitioner_config)
        module_with_submodules = ret.module_with_submodules
        dag = ret.dag
        self.assertEqual(traced(a, b), module_with_submodules(a, b))
        assert dag.nodes[0].logical_device_ids == [0]

    def test_size_based_partition(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, a, b):
                add_1 = a + b
                linear = self.linear(add_1)
                e = torch.rand(4)
                add_2 = linear + e
                return add_2

        m = TestModule()
        traced = symbolic_trace(m)
        a = torch.rand(4)
        b = torch.rand(4)
        GraphManipulation.get_size_of_all_nodes(traced, [a, b])
        partitioner = Partitioner()
        devices = [
            Device("dev_0", 125, 0),
            Device("dev_1", 125, 1),
            Device("dev_2", 125, 2),
        ]
        partitioner_config = PartitionerConfig(devices)
        ret = partitioner.partition_graph(traced, m, partitioner_config)
        module_with_submodules = ret.module_with_submodules
        dag = ret.dag
        self.assertEqual(traced(a, b), module_with_submodules(a, b))
        for i, node in enumerate(dag.nodes):
            assert node.logical_device_ids == [i]

    def test_partition_device_mapping(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, a):
                b = torch.rand(4)
                add_1 = a + b
                linear_1 = self.linear(add_1)
                add_2 = torch.rand(4) + a
                add_3 = add_2 + linear_1
                return add_3

        m = TestModule()
        traced = symbolic_trace(m)
        a = torch.rand(4)
        GraphManipulation.get_size_of_all_nodes(traced, [a])
        partitioner = Partitioner()
        devices = [Device("dev_0", 120, 0), Device("dev_1", 160, 1)]
        partitioner_config = PartitionerConfig(devices, is_sparse_nn=False)
        ret = partitioner.partition_graph(traced, m, partitioner_config)
        module_with_submodules = ret.module_with_submodules
        dag = ret.dag
        self.assertEqual(traced(a), module_with_submodules(a))
        for i, node in enumerate(dag.nodes):
            if i == 1:
                assert node.logical_device_ids == [1]
            else:
                assert node.logical_device_ids == [0]

    def test_sparse_nn_partition(self):
        class MyRecommendationModule(torch.nn.Module):
            def create_mlp(self, num_of_layers: int, input_size: int, output_size: int):
                layers = torch.nn.ModuleList()
                for _ in range(num_of_layers):
                    ll = torch.nn.Linear(input_size, output_size)
                    layers.append(ll)
                    layers.append(torch.nn.ReLU())
                return layers

            def __init__(self):
                super(MyRecommendationModule, self).__init__()
                layers = self.create_mlp(4, 4, 4)
                self.bottom_layers = torch.nn.Sequential(*layers)
                layers = self.create_mlp(3, 24, 24)
                self.top_layers = torch.nn.Sequential(*layers)
                self.embedding_layers = torch.nn.ModuleList()
                el = torch.nn.EmbeddingBag(500000, 4, mode="sum", sparse=True)
                self.embedding_layers.append(el)
                for i in range(3):
                    el = torch.nn.EmbeddingBag(1000000, 4, mode="sum", sparse=True)
                    self.embedding_layers.append(el)
                el = torch.nn.EmbeddingBag(500000, 4, mode="sum", sparse=True)
                self.embedding_layers.append(el)

            def forward(self, a, b, offset):
                x = self.bottom_layers(a)
                y = []
                c = []
                for i in range(len(self.embedding_layers)):
                    temp = torch.randint(10, (8,))
                    c.append(temp + b)
                for i in range(len(self.embedding_layers)):
                    if i % 2 == 0:
                        y.append(self.embedding_layers[i](c[i], offset))
                    else:
                        y.append(
                            self.embedding_layers[i](torch.randint(10, (8,)), offset)
                        )
                z = torch.cat([x] + y, dim=1)
                p = self.top_layers(z)
                return p

        m = MyRecommendationModule()
        a = torch.rand(2, 4)
        b = torch.randint(10, (8,))
        offset = torch.randint(1, (2,))
        traced = symbolic_trace(m)
        GraphManipulation.get_size_of_all_nodes(traced, [a, b, offset])
        devices = [
            Device("dev_0", 33000000, 0),
            Device("dev_1", 33000000, 1),
            Device("dev_2", 33000000, 2),
        ]
        partitioner_config = PartitionerConfig(devices, is_sparse_nn=True)
        partitioner = Partitioner()
        ret = partitioner.partition_graph(traced, m, partitioner_config)
        module_with_submodules = ret.module_with_submodules
        dag = ret.dag
        self.assertEqual(traced(a, b, offset), module_with_submodules(a, b, offset))
        assert len(module_with_submodules.graph.nodes) == 24

    def test_partition_latency(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super(TestModule, self).__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, a):
                add_1 = a + torch.rand(4)
                add_2 = add_1 + torch.rand(4)
                linear_1 = self.linear(add_1)
                add_3 = add_2 + linear_1
                add_4 = add_2 + add_3
                return add_4

        def get_node_to_latency_mapping(fx_module: GraphModule):
            """Given a fx module, generate node latency for each node
            based on the size of each node
            """
            node_to_latency_mapping: Dict[Node, NodeLatency] = {}
            for node in fx_module.graph.nodes:
                if node.op not in {"output", "placeholder", "get_attr"}:
                    if node.size_bytes.total_size == node.size_bytes.output_size:
                        node_to_latency_mapping[node] = NodeLatency(
                            node.size_bytes.total_size, 2.0 * node.size_bytes.total_size
                        )
                    else:
                        node_to_latency_mapping[node] = NodeLatency(
                            node.size_bytes.total_size, node.size_bytes.output_size
                        )
            return node_to_latency_mapping

        m = TestModule()
        traced = symbolic_trace(m)
        a = torch.rand(4)
        GraphManipulation.get_size_of_all_nodes(traced, [a])
        node_to_latency_mapping = get_node_to_latency_mapping(traced)
        devices = [Device("dev_0", 200, 0), Device("dev_1", 200, 1)]
        partitioner = Partitioner()
        partitioner_config = PartitionerConfig(devices, False)
        ret = partitioner.partition_graph(traced, m, partitioner_config)
        module_with_submodules = ret.module_with_submodules
        self.assertEqual(traced(a), module_with_submodules(a))
        partitions = partitioner.partitions
        partition_to_latency_mapping = get_partition_to_latency_mapping(
            partitions, node_to_latency_mapping
        )
        for p in partition_to_latency_mapping:
            if p.partition_id == 0:
                assert partition_to_latency_mapping[p] == (128.0, 80.0, 160.0)
            else:
                assert partition_to_latency_mapping[p] == (16.0, 32.0, 32.0)
        transfer_rate_bytes_per_sec = 0.5
        critical_path_latency_sec = get_latency_of_partitioned_graph(
            partitions, partition_to_latency_mapping, transfer_rate_bytes_per_sec
        )
        assert critical_path_latency_sec == 208.0

    def test_cost_aware_partition(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, a):
                add_1 = a + torch.rand(4)
                add_2 = add_1 + torch.rand(4)
                linear_1 = self.linear(add_1)
                add_3 = add_2 + torch.rand(4)
                add_4 = add_2 + linear_1
                add_5 = add_3 + add_4
                return add_5

        def get_node_to_latency_mapping(fx_module: GraphModule):
            node_to_latency_mapping: Dict[Node, Nodelatency] = {}
            for node in fx_module.graph.nodes:
                if node.op not in {'output', 'placeholder', 'get_attr'}:
                    if node.size_bytes.total_size == node.size_bytes.output_size:
                        node_to_latency_mapping[node] = NodeLatency(node.size_bytes.total_size, 1)
                    else:
                        node_to_latency_mapping[node] = NodeLatency(node.size_bytes.total_size, node.size_bytes.output_size)
            return node_to_latency_mapping

        m = MyModule()
        traced = symbolic_trace(m)
        a = torch.rand(4)
        GraphManipulation.get_size_of_all_nodes(traced, [a])
        devices = [
            Device('dev_0', 125, 0),
            Device('dev_1', 125, 1),
            Device('dev_2', 125, 2),
            Device('dev_3', 125, 3)
        ]
        node_to_latency_mapping = get_node_to_latency_mapping(traced)
        partitioner_config = PartitionerConfig(
            devices,
            is_sparse_nn=False,
            is_cost_aware=True,
            transfer_rate_bytes_per_sec=0.5,
            node_to_latency_mapping=node_to_latency_mapping
        )
        partitioner = Partitioner()
        ret = partitioner.partition_graph(traced, m, partitioner_config)
        module_with_submodules = ret.module_with_submodules
        dag = ret.dag
        self.assertEqual(traced(a), module_with_submodules(a))
        partitions = partitioner.partitions
        partition_to_latency_mapping = get_partition_to_latency_mapping(partitions, node_to_latency_mapping)
        critical_path_latency_sec = get_latency_of_partitioned_graph(
            partitions,
            partition_to_latency_mapping,
            partitioner_config.transfer_rate_bytes_per_sec
        )
        assert critical_path_latency_sec == 160.

    def test_call_to_assert_no_msg(self):
        class M(torch.nn.Module):
            def forward(self, a, b):
                assert a == b
                return a + b

        m = M()
        traced = symbolic_trace_with_rewrite(m)

        # Make sure the graph is well-formed
        traced.graph.lint(traced)

        # Check the IR to make sure there's a call_function node with target == "Assert"
        self.assertTrue(
            any(
                node.op == "call_function" and node.target == torch._assert
                for node in traced.graph.nodes
            )
        )

        # Ensure that the assert throws when it's supposed to and doesn't throw when it's not supposed to
        traced(3, 3)
        with self.assertRaisesRegex(AssertionError, ""):
            traced(3, 5)

        # Confirm that the output is correct
        self.assertEqual(traced(3, 3), m(3, 3))

    def test_call_to_assert_with_msg(self):
        class M(torch.nn.Module):
            def forward(self, a, b):
                assert a == b, "test message"
                return a + b

        m = M()
        traced = symbolic_trace_with_rewrite(m)

        # Make sure the graph is well-formed
        traced.graph.lint(traced)

        # Check the IR to make sure there's a call_function node with target == "Assert"
        self.assertTrue(
            any(
                node.op == "call_function" and node.target == torch._assert
                for node in traced.graph.nodes
            )
        )

        # Ensure that the assert throws when it's supposed to and doesn't throw when it's not supposed to
        traced(3, 3)
        with self.assertRaisesRegex(AssertionError, "test message"):
            traced(3, 5)

        # Confirm that the output is correct
        self.assertEqual(traced(3, 3), m(3, 3))

    def test_call_to_assert_with_empty_msg(self):
        class M(torch.nn.Module):
            def forward(self, a, b):
                assert a == b, ""
                return a + b

        m = M()
        traced = symbolic_trace_with_rewrite(m)

        # Make sure the graph is well-formed
        traced.graph.lint(traced)

        # Check the IR to make sure there's a call_function node with target == "Assert"
        self.assertTrue(
            any(
                node.op == "call_function" and node.target == torch._assert
                for node in traced.graph.nodes
            )
        )

        # Ensure that the assert throws when it's supposed to and doesn't throw when it's not supposed to
        traced(3, 3)
        with self.assertRaisesRegex(AssertionError, ""):
            traced(3, 5)

        # Confirm that the output is correct
        self.assertEqual(traced(3, 3), m(3, 3))

    def test_call_to_assert_with_multiline_message(self):
        class M(torch.nn.Module):
            def forward(self, a, b):
                error_msg = """
An error message with
terrible spacing
                """
                assert a == b, error_msg
                return a + b

        m = M()
        traced = symbolic_trace_with_rewrite(m)

        # Make sure the graph is well-formed
        traced.graph.lint(traced)

        # Check the IR to make sure there's a call_function node with target == "Assert"
        self.assertTrue(
            any(
                node.op == "call_function" and node.target == torch._assert
                for node in traced.graph.nodes
            )
        )

        # Ensure that the assert throws when it's supposed to and doesn't throw when it's not supposed to
        error_msg = """
An error message with
terrible spacing
    """
        traced(3, 3)
        with self.assertRaisesRegex(AssertionError, error_msg):
            traced(3, 5)

        # Confirm that the output is correct
        self.assertEqual(traced(3, 3), m(3, 3))

    def test_subgraph_creation(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(3, 4))
                self.linear = torch.nn.Linear(4, 5)

            def forward(self, x, y):
                z = self.linear(x + self.param).clamp(min=0.0, max=1.0)
                w = self.linear(y).clamp(min=0.0, max=1.0)
                return z + w

        # symbolically trace model
        my_module = MyModule()
        my_module_traced = symbolic_trace(my_module)

        # random mod partitioning
        partition_counter = 0
        NPARTITIONS = 3

        def mod_partition(node: Node):
            nonlocal partition_counter
            partition = partition_counter % NPARTITIONS
            partition_counter = (partition_counter + 1) % NPARTITIONS
            return partition

        # split module in module with submodules
        module_with_submodules = split_module(my_module_traced, my_module, mod_partition)

        x = torch.rand(3, 4)
        y = torch.rand(3, 4)

        orig_out = my_module_traced(x, y)
        submodules_out = module_with_submodules(x, y)

        self.assertEqual(orig_out, submodules_out)

    @skipIfNoTorchVision
    def test_subgraph_trivial_resnet(self):
        # Smoke test trivially splitting resnet into 1 partition works
        # There was an issue before causing submodule names to be aliased
        m = resnet18()
        traced = symbolic_trace(m)
        a = torch.rand(64, 3, 7, 7)
        module_with_submodules = split_module(traced, m, lambda node: 0)
        module_with_submodules(a)

    def test_traceable_function_with_nonstandard_name(self):
        def foo(x):
            return torch.relu(x)

        traced = symbolic_trace_with_rewrite(foo)


if __name__ == "__main__":
    run_tests()
