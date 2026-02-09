"""Tests for compile_print and make_compile_print."""

from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Optional

import torch
from functorch.compile import aot_function, nop
from torch._higher_order_ops.compile_print_wrapper import (
    compile_print,
    make_compile_print,
)
from torch._higher_order_ops.invoke_leaf_function import invoke_leaf_function
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import run_tests, TestCase


def _extract_graph(fx_g, _, graph_cell):
    graph_cell[0] = fx_g
    return fx_g


@dataclass
class RunResult:
    fwd_count: int
    bwd_count: int
    output: torch.Tensor
    grads: list[Optional[torch.Tensor]] = field(default_factory=list)
    fw_graph: Optional[torch.fx.GraphModule] = None
    bw_graph: Optional[torch.fx.GraphModule] = None
    gm: Optional[torch.fx.GraphModule] = None


def _has_invoke_leaf_function_node(gm):
    return any(
        n.op == "call_function" and n.target is invoke_leaf_function
        for n in gm.graph.nodes
    )


# MakeFn signature: (cp) -> f where f(*inputs) -> scalar tensor
MakeFn = Callable[..., Callable[..., torch.Tensor]]


def run_eager(
    make_fn: MakeFn, inputs: list[torch.Tensor], backward: bool = True
) -> RunResult:
    fwd_rec: list[torch.Size] = []
    bwd_rec: list[torch.Size] = []
    cp = make_compile_print(
        fwd_f=lambda t: fwd_rec.append(t.shape),
        bwd_f=lambda t: bwd_rec.append(t.shape),
    )
    f = make_fn(cp)
    out = f(*inputs)
    if backward and out.requires_grad:
        out.backward()
    return RunResult(
        fwd_count=len(fwd_rec),
        bwd_count=len(bwd_rec),
        output=out.detach(),
        grads=[x.grad for x in inputs],
    )


def run_compile(
    make_fn: MakeFn, inputs: list[torch.Tensor], backward: bool = True
) -> RunResult:
    fwd_rec: list[torch.Size] = []
    bwd_rec: list[torch.Size] = []
    cp = make_compile_print(
        fwd_f=lambda t: fwd_rec.append(t.shape),
        bwd_f=lambda t: bwd_rec.append(t.shape),
    )
    f = make_fn(cp)
    compiled_f = torch.compile(f, backend="aot_eager", fullgraph=True)
    cloned = [x.clone().detach().requires_grad_(x.requires_grad) for x in inputs]
    out = compiled_f(*cloned)
    if backward and out.requires_grad:
        out.backward()
    return RunResult(
        fwd_count=len(fwd_rec),
        bwd_count=len(bwd_rec),
        output=out.detach(),
        grads=[x.grad for x in cloned],
    )


def run_make_fx(make_fn: MakeFn, inputs: list[torch.Tensor]) -> RunResult:
    fwd_rec: list[torch.Size] = []
    bwd_rec: list[torch.Size] = []
    cp = make_compile_print(
        fwd_f=lambda t: fwd_rec.append(t.shape),
        bwd_f=lambda t: bwd_rec.append(t.shape),
    )
    f = make_fn(cp)
    # Tracing may call fwd_f
    gm = make_fx(f)(*inputs)
    fwd_rec.clear()
    bwd_rec.clear()
    # Re-execute on fresh inputs
    fresh = [torch.randn_like(x) for x in inputs]
    out = gm(*fresh)
    return RunResult(
        fwd_count=len(fwd_rec),
        bwd_count=len(bwd_rec),
        output=out.detach(),
        gm=gm,
    )


def run_aot_function(
    make_fn: MakeFn, inputs: list[torch.Tensor], backward: bool = True
) -> RunResult:
    fwd_rec: list[torch.Size] = []
    bwd_rec: list[torch.Size] = []
    cp = make_compile_print(
        fwd_f=lambda t: fwd_rec.append(t.shape),
        bwd_f=lambda t: bwd_rec.append(t.shape),
    )
    f = make_fn(cp)
    fw_cell: list[Optional[torch.fx.GraphModule]] = [None]
    bw_cell: list[Optional[torch.fx.GraphModule]] = [None]
    compiled_f = aot_function(
        f,
        fw_compiler=partial(_extract_graph, graph_cell=fw_cell),
        bw_compiler=partial(_extract_graph, graph_cell=bw_cell),
    )
    cloned = [x.clone().detach().requires_grad_(x.requires_grad) for x in inputs]
    # First call triggers tracing; clear records after to only count execution calls
    fwd_rec.clear()
    bwd_rec.clear()
    out = compiled_f(*cloned)
    fwd_at_forward = len(fwd_rec)
    if backward and out.requires_grad:
        out.backward()
    return RunResult(
        fwd_count=fwd_at_forward,
        bwd_count=len(bwd_rec),
        output=out.detach(),
        grads=[x.grad for x in cloned],
        fw_graph=fw_cell[0],
        bw_graph=bw_cell[0],
    )


class TestCompilePrint(TestCase):
    def test_single_tensor(self):
        def make_fn(cp):
            def f(x):
                cp(x)
                return x.sum()

            return f

        inputs = [torch.randn(3, 3, requires_grad=True)]

        eager = run_eager(make_fn, inputs)
        self.assertEqual(eager.fwd_count, 1)
        self.assertEqual(eager.bwd_count, 1)

        compiled = run_compile(make_fn, inputs)
        self.assertTrue(compiled.fwd_count >= 1)
        self.assertTrue(compiled.bwd_count >= 1)
        self.assertEqual(eager.grads[0], compiled.grads[0])

        fx = run_make_fx(make_fn, inputs)
        self.assertTrue(_has_invoke_leaf_function_node(fx.gm))
        self.assertEqual(fx.fwd_count, 1)

        aot = run_aot_function(make_fn, inputs)
        self.assertTrue(aot.fwd_count >= 1)
        self.assertEqual(eager.grads[0], aot.grads[0])

    def test_multiple_tensors(self):
        def make_fn(cp):
            def f(x, y):
                cp(x, y)
                return (x + y).sum()

            return f

        inputs = [
            torch.randn(3, 3, requires_grad=True),
            torch.randn(3, 3, requires_grad=True),
        ]

        eager = run_eager(make_fn, inputs)
        self.assertEqual(eager.fwd_count, 2)
        self.assertEqual(eager.bwd_count, 2)

        compiled = run_compile(make_fn, inputs)
        self.assertTrue(compiled.fwd_count >= 2)
        self.assertTrue(compiled.bwd_count >= 2)
        self.assertEqual(eager.grads[0], compiled.grads[0])
        self.assertEqual(eager.grads[1], compiled.grads[1])

        fx = run_make_fx(make_fn, inputs)
        self.assertTrue(_has_invoke_leaf_function_node(fx.gm))
        self.assertEqual(fx.fwd_count, 2)

        aot = run_aot_function(make_fn, inputs)
        self.assertTrue(aot.fwd_count >= 2)
        self.assertEqual(eager.grads[0], aot.grads[0])
        self.assertEqual(eager.grads[1], aot.grads[1])

    def test_no_grad_tensor(self):
        def make_fn(cp):
            def f(x):
                cp(x)
                return x.sum()

            return f

        inputs = [torch.randn(3, 3)]

        eager = run_eager(make_fn, inputs, backward=False)
        self.assertEqual(eager.fwd_count, 1)
        self.assertEqual(eager.bwd_count, 0)

        compiled = run_compile(make_fn, inputs, backward=False)
        self.assertTrue(compiled.fwd_count >= 1)

        fx = run_make_fx(make_fn, inputs)
        self.assertTrue(_has_invoke_leaf_function_node(fx.gm))
        self.assertEqual(fx.fwd_count, 1)

        aot = run_aot_function(make_fn, inputs, backward=False)
        self.assertTrue(aot.fwd_count >= 1)

    def test_computation_uses_original_tensors(self):
        def make_fn(cp):
            def f(x, y):
                cp(x, y)
                return (x * y).sum()

            return f

        inputs = [
            torch.randn(3, 3, requires_grad=True),
            torch.randn(3, 3, requires_grad=True),
        ]

        eager = run_eager(make_fn, inputs)
        compiled = run_compile(make_fn, inputs)
        aot = run_aot_function(make_fn, inputs)

        self.assertEqual(eager.grads[0], compiled.grads[0])
        self.assertEqual(eager.grads[1], compiled.grads[1])
        self.assertEqual(eager.grads[0], aot.grads[0])
        self.assertEqual(eager.grads[1], aot.grads[1])

    def test_make_fx_graph_has_opaque_node(self):
        def make_fn(cp):
            def f(x):
                cp(x)
                return x.sum()

            return f

        inputs = [torch.randn(3, 3)]
        fx = run_make_fx(make_fn, inputs)

        self.assertTrue(_has_invoke_leaf_function_node(fx.gm))
        # Re-execution produces correct results
        x = torch.randn(3, 3)
        self.assertEqual(fx.gm(x), x.sum())

    def test_aot_function_fw_graph_has_opaque_node(self):
        def make_fn(cp):
            def f(x):
                cp(x)
                return x.sum()

            return f

        inputs = [torch.randn(3, 3, requires_grad=True)]
        aot = run_aot_function(make_fn, inputs)

        self.assertTrue(_has_invoke_leaf_function_node(aot.fw_graph))

    def test_gradients_unchanged(self):
        """cp should not affect gradient values."""

        def make_fn(cp):
            def f(x):
                cp(x)
                return x.sum()

            return f

        inputs = [torch.randn(3, 3, requires_grad=True)]

        eager = run_eager(make_fn, inputs)
        self.assertEqual(eager.grads[0], torch.ones(3, 3))

        compiled = run_compile(make_fn, inputs)
        self.assertEqual(compiled.grads[0], torch.ones(3, 3))

        aot = run_aot_function(make_fn, inputs)
        self.assertEqual(aot.grads[0], torch.ones(3, 3))


class TestCompilePrintEagerOnly(TestCase):
    def test_returns_none(self):
        cp = make_compile_print(fwd_f=lambda t: None, bwd_f=lambda t: None)
        result = cp(torch.randn(3, 3))
        self.assertIsNone(result)

    def test_convenience_compile_print(self):
        recorded = []
        result = compile_print(
            lambda t: recorded.append(t.shape),
            lambda t: None,
            torch.randn(2, 2),
        )
        self.assertIsNone(result)
        self.assertEqual(len(recorded), 1)


if __name__ == "__main__":
    run_tests()
