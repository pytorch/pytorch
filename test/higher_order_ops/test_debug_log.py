# Owner(s): ["module: higher order operators"]
"""Tests for torch.utils.debug_log."""

import logging
import unittest
from contextlib import redirect_stdout
from functools import partial
from io import StringIO
from typing import Optional

import torch
import torch.distributed as dist
from functorch.compile import aot_function, make_boxed_func
from torch._dynamo.testing import AotEagerAndRecordGraphs, normalize_gm
from torch._higher_order_ops.effects import with_effects
from torch._higher_order_ops.invoke_leaf_function import invoke_leaf_function
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase
from torch.utils.debug_log import debug_log, debug_log_rank


if torch.distributed.is_available():
    from torch.distributed.tensor import DeviceMesh, DTensor, Replicate
    from torch.testing._internal.distributed.fake_pg import FakeStore


def _extract_graph(fx_g, _, graph_cell):
    graph_cell[0] = fx_g
    return make_boxed_func(fx_g)


def _has_invoke_leaf_function_node(gm):
    return any(
        n.op == "call_function"
        and (
            n.target is invoke_leaf_function
            or (
                n.target is with_effects
                and len(n.args) >= 2
                and n.args[1] is invoke_leaf_function
            )
        )
        for n in gm.graph.nodes
    )


def _run_eager(f, inputs):
    cloned = [x.clone().detach().requires_grad_(x.requires_grad) for x in inputs]
    buf = StringIO()
    with redirect_stdout(buf):
        out = f(*cloned)
        loss = out.sum()
        loss.backward()
    return out.detach(), [x.grad for x in cloned], buf.getvalue()


def _run_compile(f, inputs):
    torch._dynamo.reset()
    compiled_f = torch.compile(f, backend="aot_eager", fullgraph=True)
    cloned = [x.clone().detach().requires_grad_(x.requires_grad) for x in inputs]
    buf = StringIO()
    with redirect_stdout(buf):
        out = compiled_f(*cloned)
        loss = out.sum()
        loss.backward()
    return out.detach(), [x.grad for x in cloned], buf.getvalue()


def _run_aot_function(f, inputs):
    fw_cell: list[Optional[torch.fx.GraphModule]] = [None]
    bw_cell: list[Optional[torch.fx.GraphModule]] = [None]
    compiled_f = aot_function(
        f,
        fw_compiler=partial(_extract_graph, graph_cell=fw_cell),
        bw_compiler=partial(_extract_graph, graph_cell=bw_cell),
    )
    cloned = [x.clone().detach().requires_grad_(x.requires_grad) for x in inputs]
    buf = StringIO()
    with redirect_stdout(buf):
        out = compiled_f(*cloned)
        loss = out.sum()
        loss.backward()
    return (
        out.detach(),
        [x.grad for x in cloned],
        buf.getvalue(),
        fw_cell[0],
        bw_cell[0],
    )


class _LogCapture(logging.Handler):
    """Logging handler that captures formatted log records."""

    def __init__(self):
        super().__init__()
        self.records: list[str] = []

    def emit(self, record):
        self.records.append(self.format(record))


@skipIfTorchDynamo("debug_log tests manage their own compilation")
class TestDebugLog(TestCase):
    def test_eager_output_and_grads(self):
        def f(x):
            y = x * 2
            debug_log(y, "intermediate")
            return y

        x = torch.randn(3, 3, requires_grad=True)
        out, grads, _ = _run_eager(f, [x])

        expected_out = x * 2
        self.assertEqual(out, expected_out)
        # gradient of (x * 2).sum() w.r.t. x is 2
        self.assertEqual(grads[0], torch.full_like(x, 2.0))

    def test_compile_output_and_grads(self):
        def f(x):
            y = x * 2
            debug_log(y, "intermediate")
            return y

        x = torch.randn(3, 3, requires_grad=True)
        eager_out, eager_grads, _ = _run_eager(f, [x])
        compiled_out, compiled_grads, _ = _run_compile(f, [x])

        self.assertEqual(eager_out, compiled_out)
        self.assertEqual(eager_grads[0], compiled_grads[0])

    def test_aot_function_output_and_grads(self):
        def f(x):
            y = x * 2
            debug_log(y, "intermediate")
            return y

        x = torch.randn(3, 3, requires_grad=True)
        eager_out, eager_grads, _ = _run_eager(f, [x])
        aot_out, aot_grads, _, _, _ = _run_aot_function(f, [x])

        self.assertEqual(eager_out, aot_out)
        self.assertEqual(eager_grads[0], aot_grads[0])

    def test_eager_prints(self):
        def f(x):
            y = x * 2
            debug_log(y, "my_tag")
            return y

        x = torch.randn(3, 3, requires_grad=True)
        _, _, output = _run_eager(f, [x])

        self.assertIn("[my_tag][fwd]", output)
        self.assertIn("[my_tag][bwd]", output)

    def test_aot_function_fw_bw_graphs_have_leaf_node(self):
        def f(x):
            y = x * 2
            debug_log(y, "intermediate")
            return y

        x = torch.randn(3, 3, requires_grad=True)
        out, grads, output, fw_graph, bw_graph = _run_aot_function(f, [x])

        self.assertIn("[intermediate][fwd]", output)
        self.assertIn("[intermediate][bwd]", output)

        self.assertTrue(_has_invoke_leaf_function_node(fw_graph))
        self.assertTrue(_has_invoke_leaf_function_node(bw_graph))
        self.assertExpectedInline(
            normalize_gm(fw_graph.print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[0]", primals_2: "f32[3, 3]"):
        mul: "f32[3, 3]" = torch.ops.aten.mul.Tensor(primals_2, 2);  primals_2 = None
        _tree_spec_constant0 = self._tree_spec_constant0
        _tree_spec_constant1 = self._tree_spec_constant1
        _tree_spec_constant2 = self._tree_spec_constant2
        with_effects = torch.ops.higher_order.with_effects(primals_1, torch.ops.higher_order.invoke_leaf_function, _tree_spec_constant0, _tree_spec_constant1, _tree_spec_constant2, mul, 'intermediate', requires_grad_indices = (0,));  primals_1 = _tree_spec_constant0 = _tree_spec_constant1 = _tree_spec_constant2 = None

        getitem: "f32[0]" = with_effects[0];  with_effects = None
        return (getitem, mul)
""",  # noqa: B950
        )
        self.assertExpectedInline(
            normalize_gm(bw_graph.print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, tangents_1: "f32[3, 3]", tangents_token: "f32[0]"):
        _tree_spec_constant3 = self._tree_spec_constant3
        _tree_spec_constant4 = self._tree_spec_constant4
        _tree_spec_constant2_1 = self._tree_spec_constant2
        with_effects_1 = torch.ops.higher_order.with_effects(tangents_token, torch.ops.higher_order.invoke_leaf_function, _tree_spec_constant3, _tree_spec_constant4, _tree_spec_constant2_1, tangents_1, 'intermediate', requires_grad_indices = ());  tangents_token = _tree_spec_constant3 = _tree_spec_constant4 = _tree_spec_constant2_1 = None
        getitem_2: "f32[0]" = with_effects_1[0];  with_effects_1 = None
        mul_1: "f32[3, 3]" = torch.ops.aten.mul.Tensor(tangents_1, 2);  tangents_1 = None
        return (mul_1, getitem_2)
""",  # noqa: B950
        )

    def test_compile_fw_bw_graphs_have_leaf_node(self):
        def f(x):
            y = x * 2
            debug_log(y, "intermediate")
            return y

        backend = AotEagerAndRecordGraphs()
        compiled_f = torch.compile(f, backend=backend, fullgraph=True)
        x = torch.randn(3, 3, requires_grad=True)
        buf = StringIO()
        with redirect_stdout(buf):
            out = compiled_f(x)
            out.sum().backward()
        output = buf.getvalue()

        self.assertIn("[intermediate][fwd]", output)
        self.assertIn("[intermediate][bwd]", output)

        self.assertEqual(len(backend.fw_graphs), 1)
        self.assertEqual(len(backend.bw_graphs), 1)
        self.assertTrue(_has_invoke_leaf_function_node(backend.fw_graphs[0]))
        self.assertTrue(_has_invoke_leaf_function_node(backend.bw_graphs[0]))
        self.assertExpectedInline(
            normalize_gm(backend.fw_graphs[0].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[0]", primals_2: "f32[3, 3]"):
        mul: "f32[3, 3]" = torch.ops.aten.mul.Tensor(primals_2, 2);  primals_2 = None

        _tree_spec_constant0 = self._tree_spec_constant0
        _tree_spec_constant1 = self._tree_spec_constant1
        _tree_spec_constant2 = self._tree_spec_constant2
        with_effects = torch.ops.higher_order.with_effects(primals_1, torch.ops.higher_order.invoke_leaf_function, _tree_spec_constant0, _tree_spec_constant1, _tree_spec_constant2, mul, 'intermediate', requires_grad_indices = (0,));  primals_1 = _tree_spec_constant0 = _tree_spec_constant1 = _tree_spec_constant2 = None

        getitem: "f32[0]" = with_effects[0];  with_effects = None
        return (getitem, mul)
""",  # noqa: B950
        )
        self.assertExpectedInline(
            normalize_gm(backend.bw_graphs[0].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, tangents_1: "f32[3, 3]", tangents_token: "f32[0]"):
        _tree_spec_constant3 = self._tree_spec_constant3
        _tree_spec_constant4 = self._tree_spec_constant4
        _tree_spec_constant2_1 = self._tree_spec_constant2
        with_effects_1 = torch.ops.higher_order.with_effects(tangents_token, torch.ops.higher_order.invoke_leaf_function, _tree_spec_constant3, _tree_spec_constant4, _tree_spec_constant2_1, tangents_1, 'intermediate', requires_grad_indices = ());  tangents_token = _tree_spec_constant3 = _tree_spec_constant4 = _tree_spec_constant2_1 = None
        getitem_2: "f32[0]" = with_effects_1[0];  with_effects_1 = None

        mul_1: "f32[3, 3]" = torch.ops.aten.mul.Tensor(tangents_1, 2);  tangents_1 = None
        return (mul_1, getitem_2)
""",  # noqa: B950
        )

    def test_does_not_affect_computation(self):
        """debug_log should not change tensor values or gradients."""
        x = torch.randn(4, 4, requires_grad=True)
        x2 = x.clone().detach().requires_grad_(True)

        # With debug_log
        buf = StringIO()
        with redirect_stdout(buf):
            y = x * 2
            debug_log(y, "test")
            (y.sum()).backward()

        # Without debug_log
        y2 = x2 * 2
        (y2.sum()).backward()

        self.assertEqual(x.grad, x2.grad)

    def test_norm_printed_correctly(self):
        """Verify the printed norm matches the actual tensor norm."""
        x = torch.tensor([3.0, 4.0], requires_grad=True)
        buf = StringIO()
        with redirect_stdout(buf):
            debug_log(x, "check")
        output = buf.getvalue()
        # norm of [3, 4] is 5.0
        self.assertIn("5.", output)


@unittest.skipIf(not torch.distributed.is_available(), "requires distributed")
@skipIfTorchDynamo("debug_log_rank tests manage their own compilation")
class TestDebugLogRank(TestCase):
    def _init_fake_pg(self, rank, world_size=2):
        store = FakeStore()
        dist.init_process_group(
            backend="fake", rank=rank, world_size=world_size, store=store
        )

    def _add_log_capture(self):
        capture = _LogCapture()
        logger = logging.getLogger("torch.utils.debug_log")
        logger.addHandler(capture)
        logger.setLevel(logging.INFO)
        self.addCleanup(logger.removeHandler, capture)
        return capture

    def tearDown(self):
        super().tearDown()
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_eager_output_and_grads_with_dtensor(self):
        self._init_fake_pg(rank=0)
        mesh = DeviceMesh("cpu", range(2))
        x = torch.randn(4, 4, requires_grad=True)
        dt = DTensor.from_local(
            x.clone().detach().requires_grad_(True), mesh, [Replicate()]
        )

        def f(t):
            y = t * 2
            debug_log_rank(y, "dt_test", ranks=0)
            return y

        out = f(dt.to_local())
        loss = out.sum()
        loss.backward()

        self.assertEqual(out, x * 2)

    def test_compile_output_and_grads_with_dtensor(self):
        self._init_fake_pg(rank=0)
        mesh = DeviceMesh("cpu", range(2))

        def f(x):
            y = x * 2
            debug_log_rank(y, "dt_test", ranks=0)
            return y

        x = torch.randn(4, 4, requires_grad=True)
        dt = DTensor.from_local(x, mesh, [Replicate()])
        local = dt.to_local().clone().detach().requires_grad_(True)

        eager_out = f(local)
        eager_out.sum().backward()
        eager_grad = local.grad.clone()

        torch._dynamo.reset()
        compiled_f = torch.compile(f, backend="aot_eager", fullgraph=True)
        local2 = dt.to_local().clone().detach().requires_grad_(True)
        compiled_out = compiled_f(local2)
        compiled_out.sum().backward()

        self.assertEqual(eager_out.detach(), compiled_out.detach())
        self.assertEqual(eager_grad, local2.grad)

    def test_aot_function_output_and_grads_with_dtensor(self):
        self._init_fake_pg(rank=0)
        mesh = DeviceMesh("cpu", range(2))

        def f(x):
            y = x * 2
            debug_log_rank(y, "dt_test", ranks=0)
            return y

        x = torch.randn(4, 4, requires_grad=True)
        dt = DTensor.from_local(x, mesh, [Replicate()])
        local = dt.to_local().clone().detach().requires_grad_(True)

        eager_out = f(local)
        eager_out.sum().backward()
        eager_grad = local.grad.clone()

        local2 = dt.to_local().clone().detach().requires_grad_(True)
        aot_f = aot_function(f, fw_compiler=lambda g, _: make_boxed_func(g))
        aot_out = aot_f(local2)
        aot_out.sum().backward()

        self.assertEqual(eager_out.detach(), aot_out.detach())
        self.assertEqual(eager_grad, local2.grad)

    def test_eager_logs_when_rank_matches(self):
        for rank in range(2):
            with self.subTest(rank=rank):
                self._init_fake_pg(rank)
                mesh = DeviceMesh("cpu", range(2))
                x = torch.randn(4, 4, requires_grad=True)
                dt = DTensor.from_local(x, mesh, [Replicate()])
                capture = self._add_log_capture()

                local = dt.to_local()
                debug_log_rank(local, "match", ranks=rank)
                local.sum().backward()

                output = "\n".join(capture.records)
                self.assertIn(f"[rank {rank}][match][fwd]", output)
                self.assertIn(f"[rank {rank}][match][bwd]", output)
                dist.destroy_process_group()

    def test_eager_no_log_when_rank_does_not_match(self):
        for rank in range(2):
            with self.subTest(rank=rank):
                self._init_fake_pg(rank)
                mesh = DeviceMesh("cpu", range(2))
                x = torch.randn(4, 4, requires_grad=True)
                dt = DTensor.from_local(x, mesh, [Replicate()])
                capture = self._add_log_capture()

                other_rank = 1 - rank
                local = dt.to_local()
                debug_log_rank(local, "nomatch", ranks=other_rank)
                local.sum().backward()

                self.assertEqual(len(capture.records), 0)
                dist.destroy_process_group()

    def test_compile_logs_when_rank_matches(self):
        self._init_fake_pg(rank=0)
        mesh = DeviceMesh("cpu", range(2))
        dt = DTensor.from_local(
            torch.randn(4, 4, requires_grad=True), mesh, [Replicate()]
        )
        capture = self._add_log_capture()

        def f(x):
            y = x * 2
            debug_log_rank(y, "compiled", ranks=0)
            return y

        torch._dynamo.reset()
        compiled_f = torch.compile(f, backend="aot_eager", fullgraph=True)
        local = dt.to_local().clone().detach().requires_grad_(True)
        out = compiled_f(local)
        out.sum().backward()

        output = "\n".join(capture.records)
        self.assertIn("[rank 0][compiled][fwd]", output)
        self.assertIn("[rank 0][compiled][bwd]", output)

    def test_compile_no_log_when_rank_does_not_match(self):
        self._init_fake_pg(rank=0)
        mesh = DeviceMesh("cpu", range(2))
        dt = DTensor.from_local(
            torch.randn(4, 4, requires_grad=True), mesh, [Replicate()]
        )
        capture = self._add_log_capture()

        def f(x):
            debug_log_rank(x, "compiled", ranks=1)
            return x

        torch._dynamo.reset()
        compiled_f = torch.compile(f, backend="aot_eager", fullgraph=True)
        local = dt.to_local().clone().detach().requires_grad_(True)
        out = compiled_f(local)
        out.sum().backward()

        self.assertEqual(len(capture.records), 0)

    def test_aot_function_logs_when_rank_matches(self):
        self._init_fake_pg(rank=0)
        mesh = DeviceMesh("cpu", range(2))
        dt = DTensor.from_local(
            torch.randn(4, 4, requires_grad=True), mesh, [Replicate()]
        )
        capture = self._add_log_capture()

        def f(x):
            y = x * 2
            debug_log_rank(y, "aot", ranks=0)
            return y

        local = dt.to_local().clone().detach().requires_grad_(True)
        aot_f = aot_function(f, fw_compiler=lambda g, _: make_boxed_func(g))
        out = aot_f(local)
        out.sum().backward()

        output = "\n".join(capture.records)
        self.assertIn("[rank 0][aot][fwd]", output)
        self.assertIn("[rank 0][aot][bwd]", output)

    def test_aot_function_log_before_compute(self):
        self._init_fake_pg(rank=0)
        mesh = DeviceMesh("cpu", range(2))
        dt = DTensor.from_local(
            torch.randn(4, 4, requires_grad=True), mesh, [Replicate()]
        )
        capture = self._add_log_capture()

        def f(x):
            debug_log_rank(x, "before_compute", ranks=0)
            y = x * 2
            return y

        local = dt.to_local().clone().detach().requires_grad_(True)
        aot_f = aot_function(f, fw_compiler=lambda g, _: make_boxed_func(g))
        out = aot_f(local)
        out.sum().backward()

        output = "\n".join(capture.records)
        self.assertIn("[rank 0][before_compute][fwd]", output)
        self.assertIn("[rank 0][before_compute][bwd]", output)

    def test_aot_function_no_log_when_rank_does_not_match(self):
        self._init_fake_pg(rank=0)
        mesh = DeviceMesh("cpu", range(2))
        dt = DTensor.from_local(
            torch.randn(4, 4, requires_grad=True), mesh, [Replicate()]
        )
        capture = self._add_log_capture()

        def f(x):
            debug_log_rank(x, "aot", ranks=1)
            return x

        local = dt.to_local().clone().detach().requires_grad_(True)
        aot_f = aot_function(f, fw_compiler=lambda g, _: make_boxed_func(g))
        out = aot_f(local)
        out.sum().backward()

        self.assertEqual(len(capture.records), 0)

    def test_no_rank_filter_logs_all_ranks(self):
        for rank in range(2):
            with self.subTest(rank=rank):
                self._init_fake_pg(rank)
                mesh = DeviceMesh("cpu", range(2))
                x = torch.randn(4, 4, requires_grad=True)
                dt = DTensor.from_local(x, mesh, [Replicate()])
                capture = self._add_log_capture()

                local = dt.to_local()
                debug_log_rank(local, "all_ranks")
                local.sum().backward()

                output = "\n".join(capture.records)
                self.assertIn(f"[rank {rank}][all_ranks][fwd]", output)
                self.assertIn(f"[rank {rank}][all_ranks][bwd]", output)
                dist.destroy_process_group()

    def test_aot_function_fw_bw_graphs_have_leaf_node(self):
        self._init_fake_pg(rank=0)
        mesh = DeviceMesh("cpu", range(2))
        dt = DTensor.from_local(
            torch.randn(4, 4, requires_grad=True), mesh, [Replicate()]
        )
        capture = self._add_log_capture()

        def f(x):
            y = x * 2
            debug_log_rank(y, "graph_check", ranks=0)
            return y

        fw_cell: list[Optional[torch.fx.GraphModule]] = [None]
        bw_cell: list[Optional[torch.fx.GraphModule]] = [None]
        local = dt.to_local().clone().detach().requires_grad_(True)
        aot_f = aot_function(
            f,
            fw_compiler=partial(_extract_graph, graph_cell=fw_cell),
            bw_compiler=partial(_extract_graph, graph_cell=bw_cell),
        )
        out = aot_f(local)
        out.sum().backward()

        output = "\n".join(capture.records)
        self.assertIn("[rank 0][graph_check][fwd]", output)
        self.assertIn("[rank 0][graph_check][bwd]", output)

        self.assertTrue(_has_invoke_leaf_function_node(fw_cell[0]))
        self.assertTrue(_has_invoke_leaf_function_node(bw_cell[0]))
        self.assertExpectedInline(
            normalize_gm(fw_cell[0].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[0]", primals_2: "f32[4, 4]"):
        mul: "f32[4, 4]" = torch.ops.aten.mul.Tensor(primals_2, 2);  primals_2 = None
        _tree_spec_constant0 = self._tree_spec_constant0
        _tree_spec_constant1 = self._tree_spec_constant1
        _tree_spec_constant2 = self._tree_spec_constant2
        with_effects = torch.ops.higher_order.with_effects(primals_1, torch.ops.higher_order.invoke_leaf_function, _tree_spec_constant0, _tree_spec_constant1, _tree_spec_constant2, mul, 'graph_check', 0, requires_grad_indices = (0,));  primals_1 = _tree_spec_constant0 = _tree_spec_constant1 = _tree_spec_constant2 = None

        getitem: "f32[0]" = with_effects[0];  with_effects = None
        return (getitem, mul)
""",  # noqa: B950
        )
        self.assertExpectedInline(
            normalize_gm(bw_cell[0].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, tangents_1: "f32[4, 4]", tangents_token: "f32[0]"):
        _tree_spec_constant3 = self._tree_spec_constant3
        _tree_spec_constant4 = self._tree_spec_constant4
        _tree_spec_constant2_1 = self._tree_spec_constant2
        with_effects_1 = torch.ops.higher_order.with_effects(tangents_token, torch.ops.higher_order.invoke_leaf_function, _tree_spec_constant3, _tree_spec_constant4, _tree_spec_constant2_1, tangents_1, 'graph_check', 0, requires_grad_indices = ());  tangents_token = _tree_spec_constant3 = _tree_spec_constant4 = _tree_spec_constant2_1 = None
        getitem_2: "f32[0]" = with_effects_1[0];  with_effects_1 = None
        mul_1: "f32[4, 4]" = torch.ops.aten.mul.Tensor(tangents_1, 2);  tangents_1 = None
        return (mul_1, getitem_2)
""",  # noqa: B950
        )

    def test_compile_fw_bw_graphs_have_leaf_node(self):
        self._init_fake_pg(rank=0)
        mesh = DeviceMesh("cpu", range(2))
        dt = DTensor.from_local(
            torch.randn(4, 4, requires_grad=True), mesh, [Replicate()]
        )
        capture = self._add_log_capture()

        def f(x):
            y = x * 2
            debug_log_rank(y, "graph_check", ranks=0)
            return y

        backend = AotEagerAndRecordGraphs()
        compiled_f = torch.compile(f, backend=backend, fullgraph=True)
        local = dt.to_local().clone().detach().requires_grad_(True)
        out = compiled_f(local)
        out.sum().backward()

        output = "\n".join(capture.records)
        self.assertIn("[rank 0][graph_check][fwd]", output)
        self.assertIn("[rank 0][graph_check][bwd]", output)

        self.assertEqual(len(backend.fw_graphs), 1)
        self.assertEqual(len(backend.bw_graphs), 1)
        self.assertTrue(_has_invoke_leaf_function_node(backend.fw_graphs[0]))
        self.assertTrue(_has_invoke_leaf_function_node(backend.bw_graphs[0]))
        self.assertExpectedInline(
            normalize_gm(backend.fw_graphs[0].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[0]", primals_2: "f32[4, 4]"):
        mul: "f32[4, 4]" = torch.ops.aten.mul.Tensor(primals_2, 2);  primals_2 = None

        _tree_spec_constant0 = self._tree_spec_constant0
        _tree_spec_constant1 = self._tree_spec_constant1
        _tree_spec_constant2 = self._tree_spec_constant2
        with_effects = torch.ops.higher_order.with_effects(primals_1, torch.ops.higher_order.invoke_leaf_function, _tree_spec_constant0, _tree_spec_constant1, _tree_spec_constant2, mul, 'graph_check', 0, requires_grad_indices = (0,));  primals_1 = _tree_spec_constant0 = _tree_spec_constant1 = _tree_spec_constant2 = None

        getitem: "f32[0]" = with_effects[0];  with_effects = None
        return (getitem, mul)
""",  # noqa: B950
        )
        self.assertExpectedInline(
            normalize_gm(backend.bw_graphs[0].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, tangents_1: "f32[4, 4]", tangents_token: "f32[0]"):
        _tree_spec_constant3 = self._tree_spec_constant3
        _tree_spec_constant4 = self._tree_spec_constant4
        _tree_spec_constant2_1 = self._tree_spec_constant2
        with_effects_1 = torch.ops.higher_order.with_effects(tangents_token, torch.ops.higher_order.invoke_leaf_function, _tree_spec_constant3, _tree_spec_constant4, _tree_spec_constant2_1, tangents_1, 'graph_check', 0, requires_grad_indices = ());  tangents_token = _tree_spec_constant3 = _tree_spec_constant4 = _tree_spec_constant2_1 = None
        getitem_2: "f32[0]" = with_effects_1[0];  with_effects_1 = None

        mul_1: "f32[4, 4]" = torch.ops.aten.mul.Tensor(tangents_1, 2);  tangents_1 = None
        return (mul_1, getitem_2)
""",  # noqa: B950
        )

    def test_does_not_affect_computation_with_dtensor(self):
        self._init_fake_pg(rank=0)
        mesh = DeviceMesh("cpu", range(2))
        x = torch.randn(4, 4)
        dt = DTensor.from_local(x, mesh, [Replicate()])
        local = dt.to_local()

        y = local * 2
        debug_log_rank(y, "passthrough", ranks=0)
        self.assertEqual(y, local * 2)

    # need to add Dtensor support to invoke_leaf_function HOP
    @unittest.expectedFailure
    def test_eager_dtensor_directly(self):
        """invoke_leaf_function HOP has no dispatch rule for DTensor subclass."""
        self._init_fake_pg(rank=0)
        mesh = DeviceMesh("cpu", range(2))
        x = torch.randn(4, 4, requires_grad=True)
        dt = DTensor.from_local(x, mesh, [Replicate()])
        capture = self._add_log_capture()

        y = dt * 2
        debug_log_rank(y, "dtensor_direct", ranks=0)
        y.sum().backward()

        output = "\n".join(capture.records)
        self.assertIn("[rank 0][dtensor_direct][fwd]", output)
        self.assertIn("[rank 0][dtensor_direct][bwd]", output)


if __name__ == "__main__":
    run_tests()
