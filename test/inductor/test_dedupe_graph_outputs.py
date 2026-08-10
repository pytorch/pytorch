# Owner(s): ["module: inductor"]

from unittest import mock

import torch
from torch import fx
from torch._inductor import config
from torch._inductor.fx_passes.dedupe_graph_outputs import (
    _is_shareable_node,
    _structural_classes,
    dedupe_graph_outputs_pass,
    is_output_computation_sharing_supported,
)
from torch._inductor.test_case import run_tests, TestCase
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.passes.fake_tensor_prop import FakeTensorProp


def _propagate(gm: fx.GraphModule, *inputs: torch.Tensor) -> None:
    mode = FakeTensorMode()
    FakeTensorProp(gm, mode=mode).propagate(
        *(mode.from_tensor(value) for value in inputs)
    )


def _count(graph: fx.Graph, target) -> int:
    return sum(node.target is target for node in graph.nodes)


def _sin_graph(count: int = 8) -> fx.GraphModule:
    graph = fx.Graph()
    x = graph.placeholder("x")
    graph.output(
        tuple(
            graph.call_function(torch.ops.aten.sin.default, (x,)) for _ in range(count)
        )
    )
    gm = fx.GraphModule({}, graph)
    _propagate(gm, torch.randn(8))
    return gm


class TestDedupeGraphOutputs(TestCase):
    def test_shares_compute_and_preserves_storage(self):
        graph = fx.Graph()
        x = graph.placeholder("x")
        output_only = graph.call_function(torch.ops.aten.sin.default, (x,))
        canonical = graph.call_function(torch.ops.aten.sin.default, (x,))
        internal = graph.call_function(torch.ops.aten.add.Tensor, (canonical, 1))
        siblings = [
            graph.call_function(torch.ops.aten.sin.default, (x,)) for _ in range(6)
        ]
        graph.output((output_only, canonical, *siblings, internal))
        gm = fx.GraphModule({}, graph)
        _propagate(gm, torch.randn(8))

        dedupe_graph_outputs_pass(graph)
        gm.recompile()
        result = gm(torch.randn(8))

        self.assertEqual(len({value.data_ptr() for value in result[:8]}), 8)
        torch.testing.assert_close(result[-1], result[0] + 1)
        self.assertEqual(_count(graph, torch.ops.aten.sin.default), 1)
        self.assertEqual(_count(graph, torch.ops.aten.clone.default), 7)

    def test_branch_count_and_identity_bounds(self):
        for count in (7, 33):
            with self.subTest(count=count):
                gm = _sin_graph(count)
                dedupe_graph_outputs_pass(gm.graph)
                self.assertEqual(_count(gm.graph, torch.ops.aten.sin.default), count)
                self.assertEqual(_count(gm.graph, torch.ops.aten.clone.default), 0)

        graph = fx.Graph()
        x = graph.placeholder("x")
        value = graph.call_function(torch.ops.aten.sin.default, (x,))
        graph.output((value,) * 8)
        gm = fx.GraphModule({}, graph)
        _propagate(gm, torch.randn(8))
        dedupe_graph_outputs_pass(graph)
        self.assertEqual(_count(graph, torch.ops.aten.clone.default), 0)

    def test_effectful_graphs_fail_closed(self):
        graph = fx.Graph()
        x = graph.placeholder("x")
        replacement = graph.placeholder("replacement")
        before = [
            graph.call_function(torch.ops.aten.sin.default, (x,)) for _ in range(4)
        ]
        graph.call_function(torch.ops.aten.copy_.default, (x, replacement))
        after = [
            graph.call_function(torch.ops.aten.sin.default, (x,)) for _ in range(4)
        ]
        graph.output((*before, *after))
        gm = fx.GraphModule({}, graph)
        _propagate(gm, torch.zeros(8), torch.ones(8))
        dedupe_graph_outputs_pass(graph)
        self.assertEqual(_count(graph, torch.ops.aten.sin.default), 8)

        for target in (
            torch.ops.aten.rand_like.default,
            torch.ops.aten.empty_like.default,
        ):
            with self.subTest(target=target):
                graph = fx.Graph()
                x = graph.placeholder("x")
                graph.output(tuple(graph.call_function(target, (x,)) for _ in range(8)))
                gm = fx.GraphModule({}, graph)
                _propagate(gm, torch.randn(8))
                dedupe_graph_outputs_pass(graph)
                self.assertEqual(_count(graph, target), 8)
                self.assertEqual(_count(graph, torch.ops.aten.clone.default), 0)

    def test_collectives_fail_closed(self):
        import torch.distributed._functional_collectives

        graph = fx.Graph()
        x = graph.placeholder("x")
        target = torch.ops._c10d_functional.all_reduce.default
        graph.output(
            tuple(graph.call_function(target, (x, "sum", "0")) for _ in range(8))
        )
        gm = fx.GraphModule({}, graph)
        _propagate(gm, torch.randn(8))
        dedupe_graph_outputs_pass(graph)
        self.assertEqual(_count(graph, target), 8)

    def test_uninitialized_and_special_ops_are_not_shareable(self):
        graph = fx.Graph()
        scales = graph.placeholder("scales")
        zero_points = graph.placeholder("zero_points")
        qtensor = graph.placeholder("qtensor")
        nodes = [
            graph.call_function(
                torch.ops.aten._empty_affine_quantized.default,
                ((8,),),
                {"dtype": torch.quint8, "scale": 1.0, "zero_point": 0},
            ),
            graph.call_function(
                torch.ops.aten._empty_per_channel_affine_quantized.default,
                ((8,),),
                {
                    "scales": scales,
                    "zero_points": zero_points,
                    "axis": 0,
                    "dtype": torch.quint8,
                },
            ),
            graph.call_function(
                torch.ops.aten.empty_quantized.default, ((8,), qtensor)
            ),
            graph.call_function(
                torch.ops.aten._efficientzerotensor.default,
                ((8,),),
                {"dtype": torch.float32, "device": torch.device("cuda")},
            ),
        ]
        self.assertTrue(all(not _is_shareable_node(node) for node in nodes))

        for target, dtype in (
            (torch.ops.aten._conj.default, torch.complex64),
            (torch.ops.aten._neg_view.default, torch.float32),
        ):
            with self.subTest(target=target):
                graph = fx.Graph()
                x = graph.placeholder("x")
                graph.output(tuple(graph.call_function(target, (x,)) for _ in range(8)))
                gm = fx.GraphModule({}, graph)
                _propagate(gm, torch.randn(8, dtype=dtype))
                dedupe_graph_outputs_pass(graph)
                self.assertEqual(_count(graph, target), 8)

    def test_aliases_and_unsupported_storage_fail_closed(self):
        for inter_output_alias in (False, True):
            with self.subTest(inter_output_alias=inter_output_alias):
                graph = fx.Graph()
                x = graph.placeholder("x")
                source = (
                    graph.call_function(torch.ops.aten.sin.default, (x,))
                    if inter_output_alias
                    else x
                )
                outputs = [
                    graph.call_function(torch.ops.aten.view.default, (source, (2, 4)))
                    for _ in range(8)
                ]
                graph.output(tuple(outputs))
                gm = fx.GraphModule({}, graph)
                _propagate(gm, torch.randn(8))
                dedupe_graph_outputs_pass(graph)
                self.assertEqual(_count(graph, torch.ops.aten.clone.default), 0)

        mode = FakeTensorMode()
        sparse = mode.from_tensor(
            torch.sparse_coo_tensor(
                torch.tensor([[0], [1]]), torch.tensor([1.0]), (2, 2)
            )
        )
        graph = fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = sparse
        outputs = []
        for _ in range(8):
            node = graph.call_function(torch.ops.aten.clone.default, (x,))
            node.meta["val"] = sparse
            outputs.append(node)
        graph.output(tuple(outputs))
        dedupe_graph_outputs_pass(graph)
        self.assertEqual(_count(graph, torch.ops.aten.clone.default), 8)

    def test_unrelated_sparse_and_dense_strided_outputs(self):
        graph = fx.Graph()
        graph.placeholder("unused")
        x = graph.placeholder("x")
        outputs = []
        for _ in range(8):
            value = graph.call_function(torch.ops.aten.sin.default, (x,))
            outputs.append(
                graph.call_function(torch.ops.aten.permute.default, (value, (1, 0)))
            )
        graph.output(tuple(outputs))
        gm = fx.GraphModule({}, graph)

        mode = FakeTensorMode()
        sparse = mode.from_tensor(
            torch.sparse_coo_tensor(
                torch.tensor([[0], [1]]), torch.tensor([1.0]), (2, 2)
            )
        )
        dense = mode.from_tensor(torch.randn(3, 5))
        FakeTensorProp(gm, mode=mode).propagate(sparse, dense)
        dedupe_graph_outputs_pass(graph)
        gm.recompile()
        result = gm(
            torch.sparse_coo_tensor(
                torch.tensor([[0], [1]]), torch.tensor([1.0]), (2, 2)
            ),
            torch.randn(3, 5),
        )
        self.assertEqual(_count(graph, torch.ops.aten.sin.default), 1)
        self.assertEqual(len({value.data_ptr() for value in result}), 8)
        self.assertTrue(all(value.stride() == (1, 5) for value in result))

    def test_dynamic_outputs_fail_closed(self):
        gm = make_fx(
            lambda x: tuple(torch.sin(x) for _ in range(8)),
            tracing_mode="symbolic",
        )(torch.randn(5))
        dedupe_graph_outputs_pass(gm.graph)
        self.assertEqual(_count(gm.graph, torch.ops.aten.sin.default), 8)

    def test_structural_keys_cover_nested_constants_and_deep_graphs(self):
        graph = fx.Graph()
        x = graph.placeholder("x")
        outputs = []
        for _ in range(8):
            lhs = graph.call_function(torch.ops.aten.sin.default, (x,))
            rhs = graph.call_function(torch.ops.aten.cos.default, (x,))
            outputs.append(
                graph.call_function(torch.ops.aten.cat.default, ([lhs, rhs], 0))
            )
        graph.output(tuple(outputs))
        gm = fx.GraphModule({}, graph)
        _propagate(gm, torch.randn(8))
        dedupe_graph_outputs_pass(graph)
        self.assertEqual(_count(graph, torch.ops.aten.cat.default), 1)

        graph = fx.Graph()
        x = graph.placeholder("x")
        with_int = graph.call_function(torch.ops.aten.add.Tensor, (x, 1), {"alpha": 1})
        with_bool = graph.call_function(
            torch.ops.aten.add.Tensor, (x, 1), {"alpha": True}
        )
        classes = _structural_classes(graph)
        self.assertNotEqual(classes[with_int], classes[with_bool])

        graph = fx.Graph()
        x = graph.placeholder("x")
        deep_outputs = []
        for _ in range(8):
            value = x
            for _ in range(1100):
                value = graph.call_function(torch.ops.aten.sin.default, (value,))
            deep_outputs.append(value)
        graph.output(tuple(deep_outputs))
        classes = _structural_classes(graph)
        self.assertEqual(len({classes[node] for node in deep_outputs}), 1)

    def test_prims_and_view_branches_are_shared(self):
        graph = fx.Graph()
        x = graph.placeholder("x")
        outputs = []
        for _ in range(8):
            value = graph.call_function(
                torch.ops.prims.convert_element_type.default,
                (x, torch.bfloat16),
            )
            value = graph.call_function(torch.ops.aten.unsqueeze.default, (value, 0))
            value = graph.call_function(torch.ops.aten.view.default, (value, (1, 2, 4)))
            outputs.append(graph.call_function(torch.ops.aten.squeeze.dim, (value, 0)))
        graph.output(tuple(outputs))
        gm = fx.GraphModule({}, graph)
        _propagate(gm, torch.randn(2, 4))
        dedupe_graph_outputs_pass(graph)
        self.assertEqual(_count(graph, torch.ops.prims.convert_element_type.default), 1)
        self.assertEqual(_count(graph, torch.ops.aten.clone.default), 7)

    @config.patch("cuda_backend", "triton")
    def test_production_gate_uses_graph_device_and_sm(self):
        mode = FakeTensorMode()
        with mode:
            fake_cuda = torch.empty(8, device="cuda:3")

        graph = fx.Graph()
        output = graph.placeholder("output")
        output.meta["val"] = fake_cuda
        graph.output((output,))
        gm = fx.GraphModule({}, graph)

        for major, expected in ((10, True), (9, False)):
            with self.subTest(major=major):
                worker = mock.Mock()
                worker.get_device_properties.return_value = mock.Mock(major=major)
                device_interface = mock.Mock(Worker=worker)
                with mock.patch(
                    "torch._inductor.fx_passes.dedupe_graph_outputs."
                    "get_interface_for_device",
                    return_value=device_interface,
                ):
                    self.assertEqual(
                        is_output_computation_sharing_supported(gm), expected
                    )
                worker.get_device_properties.assert_called_once_with(
                    torch.device("cuda:3")
                )


if __name__ == "__main__":
    run_tests()
