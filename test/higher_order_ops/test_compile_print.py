# Owner(s): ["module: higher order operators"]
"""Tests for compile_print and make_compile_print."""

from collections.abc import Callable
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from functools import partial
from io import StringIO
from typing import Optional

import torch
import torch.distributed as dist
from functorch.compile import aot_function
from torch._higher_order_ops.compile_print_wrapper import (
    compile_print,
    make_compile_print,
)
from torch._higher_order_ops.invoke_leaf_function import invoke_leaf_function
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase
from torch.testing._internal.distributed.fake_pg import FakeStore


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


@skipIfTorchDynamo("compile_print tests manage their own compilation")
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

    def test_print_tensor_repr(self):
        """print(tensor) calls __repr__ which dispatches torch.isfinite; verify it works."""

        def make_fn(cp):
            def f(x):
                cp(x)
                return x.sum()

            return f

        # Eager mode: fwd and bwd both call print(tensor)
        inputs = [torch.randn(3, 3, requires_grad=True)]
        cp = make_compile_print(
            fwd_f=lambda t: print("fwd", t),
            bwd_f=lambda t: print("bwd", t),
        )
        buf = StringIO()
        with redirect_stdout(buf):
            run_eager(lambda _cp: make_fn(cp), inputs)
        output = buf.getvalue()
        self.assertIn("fwd", output)
        self.assertIn("bwd", output)
        # print(tensor) includes "tensor(" in its repr
        self.assertGreaterEqual(output.count("tensor("), 2)

        # torch.compile mode
        inputs = [torch.randn(3, 3, requires_grad=True)]
        cp = make_compile_print(
            fwd_f=lambda t: print("fwd", t),
            bwd_f=lambda t: print("bwd", t),
        )
        f = make_fn(cp)
        compiled_f = torch.compile(f, backend="aot_eager", fullgraph=True)
        cloned = [x.clone().detach().requires_grad_(x.requires_grad) for x in inputs]
        buf = StringIO()
        with redirect_stdout(buf):
            out = compiled_f(*cloned)
            out.backward()
        output = buf.getvalue()
        self.assertIn("fwd", output)
        self.assertIn("bwd", output)
        self.assertGreaterEqual(output.count("tensor("), 2)

        # aot_function mode — backward hooks don't fire here because
        # only the leaf function body runs at runtime, not compile_print_impl
        # which registers the hooks. Forward print still exercises the fix.
        inputs = [torch.randn(3, 3, requires_grad=True)]
        cp = make_compile_print(
            fwd_f=lambda t: print("fwd", t),
            bwd_f=lambda t: print("bwd", t),
        )
        buf = StringIO()
        with redirect_stdout(buf):
            run_aot_function(lambda _cp: make_fn(cp), inputs)
        output = buf.getvalue()
        self.assertIn("fwd", output)
        self.assertIn("tensor(", output)

    def test_multiple_compile_prints(self):
        """Two separate make_compile_print instances in the same function."""

        def make_fn(cp1, cp2):
            def f(x, y):
                cp1(x)
                cp2(y)
                return (x + y).sum()

            return f

        inputs = [
            torch.randn(3, 3, requires_grad=True),
            torch.randn(3, 3, requires_grad=True),
        ]

        # Eager
        fwd1, bwd1 = [], []
        fwd2, bwd2 = [], []
        cp1 = make_compile_print(
            fwd_f=lambda t: fwd1.append(t.shape),
            bwd_f=lambda t: bwd1.append(t.shape),
        )
        cp2 = make_compile_print(
            fwd_f=lambda t: fwd2.append(t.shape),
            bwd_f=lambda t: bwd2.append(t.shape),
        )
        f = make_fn(cp1, cp2)
        cloned = [x.clone().detach().requires_grad_(True) for x in inputs]
        out = f(*cloned)
        out.backward()
        self.assertEqual(len(fwd1), 1)
        self.assertEqual(len(fwd2), 1)
        self.assertEqual(len(bwd1), 1)
        self.assertEqual(len(bwd2), 1)
        eager_grads = [x.grad for x in cloned]

        # Compile
        fwd1, bwd1, fwd2, bwd2 = [], [], [], []
        cp1 = make_compile_print(
            fwd_f=lambda t: fwd1.append(t.shape),
            bwd_f=lambda t: bwd1.append(t.shape),
        )
        cp2 = make_compile_print(
            fwd_f=lambda t: fwd2.append(t.shape),
            bwd_f=lambda t: bwd2.append(t.shape),
        )
        f = make_fn(cp1, cp2)
        compiled_f = torch.compile(f, backend="aot_eager", fullgraph=True)
        cloned = [x.clone().detach().requires_grad_(True) for x in inputs]
        out = compiled_f(*cloned)
        out.backward()
        self.assertTrue(len(fwd1) >= 1)
        self.assertTrue(len(fwd2) >= 1)
        self.assertTrue(len(bwd1) >= 1)
        self.assertTrue(len(bwd2) >= 1)
        self.assertEqual(eager_grads[0], cloned[0].grad)
        self.assertEqual(eager_grads[1], cloned[1].grad)

        # aot_function
        fwd1, bwd1, fwd2, bwd2 = [], [], [], []
        cp1 = make_compile_print(
            fwd_f=lambda t: fwd1.append(t.shape),
            bwd_f=lambda t: bwd1.append(t.shape),
        )
        cp2 = make_compile_print(
            fwd_f=lambda t: fwd2.append(t.shape),
            bwd_f=lambda t: bwd2.append(t.shape),
        )
        f = make_fn(cp1, cp2)
        compiled_f = aot_function(f, fw_compiler=lambda g, _: g)
        cloned = [x.clone().detach().requires_grad_(True) for x in inputs]
        fwd1.clear()
        fwd2.clear()
        bwd1.clear()
        bwd2.clear()
        out = compiled_f(*cloned)
        self.assertTrue(len(fwd1) >= 1)
        self.assertTrue(len(fwd2) >= 1)
        out.backward()
        self.assertEqual(eager_grads[0], cloned[0].grad)
        self.assertEqual(eager_grads[1], cloned[1].grad)

    def test_tag_printing(self):
        """tag kwarg prints [tag][fwd] and [tag][bwd] labels."""
        cp = make_compile_print(
            fwd_f=lambda t: None,
            bwd_f=lambda t: None,
        )

        # Eager
        x = torch.randn(3, 3, requires_grad=True)
        buf = StringIO()
        with redirect_stdout(buf):
            cp(x, tag="X")
            x.sum().backward()
        output = buf.getvalue()
        self.assertIn("[X][fwd]", output)
        self.assertIn("[X][bwd]", output)

        # Compile
        def f(x):
            cp(x, tag="X")
            return x.sum()

        compiled_f = torch.compile(f, backend="aot_eager", fullgraph=True)
        x = torch.randn(3, 3, requires_grad=True)
        buf = StringIO()
        with redirect_stdout(buf):
            out = compiled_f(x)
            out.backward()
        output = buf.getvalue()
        self.assertIn("[X][fwd]", output)
        self.assertIn("[X][bwd]", output)

    def test_multiple_tags(self):
        """Two cp calls with different tags in the same function."""
        cp = make_compile_print(
            fwd_f=lambda t: None,
            bwd_f=lambda t: None,
        )

        def f(x, y):
            cp(x, tag="A")
            cp(y, tag="B")
            return (x + y).sum()

        # Eager
        x = torch.randn(3, 3, requires_grad=True)
        y = torch.randn(3, 3, requires_grad=True)
        buf = StringIO()
        with redirect_stdout(buf):
            out = f(x, y)
            out.backward()
        output = buf.getvalue()
        self.assertIn("[A][fwd]", output)
        self.assertIn("[B][fwd]", output)
        self.assertIn("[A][bwd]", output)
        self.assertIn("[B][bwd]", output)

        # Compile
        compiled_f = torch.compile(f, backend="aot_eager", fullgraph=True)
        x = torch.randn(3, 3, requires_grad=True)
        y = torch.randn(3, 3, requires_grad=True)
        buf = StringIO()
        with redirect_stdout(buf):
            out = compiled_f(x, y)
            out.backward()
        output = buf.getvalue()
        self.assertIn("[A][fwd]", output)
        self.assertIn("[B][fwd]", output)
        self.assertIn("[A][bwd]", output)
        self.assertIn("[B][bwd]", output)


@skipIfTorchDynamo("compile_print tests manage their own compilation")
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


@skipIfTorchDynamo("compile_print tests manage their own compilation")
class TestCompilePrintDistributed(TestCase):
    """Tests for rank-aware tag printing using the fake distributed backend.

    Each test runs for both rank 0 and rank 1 via subTest, reinitializing
    the fake process group between iterations.
    """

    def _init_fake_pg(self, rank):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=rank, world_size=2, store=store)

    def tearDown(self):
        super().tearDown()
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_rank_prefix_in_tag(self):
        """Tag output includes [rank N] when distributed is initialized."""
        for rank in range(2):
            with self.subTest(rank=rank):
                self._init_fake_pg(rank)
                cp = make_compile_print(fwd_f=lambda t: None, bwd_f=lambda t: None)

                x = torch.randn(3, 3, requires_grad=True)
                buf = StringIO()
                with redirect_stdout(buf):
                    cp(x, tag="input")
                    x.sum().backward()
                output = buf.getvalue()
                self.assertIn(f"[rank {rank}][input][fwd]", output)
                self.assertIn(f"[rank {rank}][input][bwd]", output)
                dist.destroy_process_group()

    def test_ranks_filter_match(self):
        """ranks=current rank allows output."""
        for rank in range(2):
            with self.subTest(rank=rank):
                self._init_fake_pg(rank)
                fwd_rec: list[torch.Size] = []
                bwd_rec: list[torch.Size] = []
                cp = make_compile_print(
                    fwd_f=lambda t: fwd_rec.append(t.shape),
                    bwd_f=lambda t: bwd_rec.append(t.shape),
                )

                x = torch.randn(3, 3, requires_grad=True)
                buf = StringIO()
                with redirect_stdout(buf):
                    cp(x, tag="filtered", ranks=rank)
                    x.sum().backward()
                output = buf.getvalue()
                self.assertIn(f"[rank {rank}][filtered][fwd]", output)
                self.assertIn(f"[rank {rank}][filtered][bwd]", output)
                self.assertEqual(len(fwd_rec), 1)
                self.assertEqual(len(bwd_rec), 1)
                dist.destroy_process_group()

    def test_ranks_filter_no_match(self):
        """ranks for the other rank suppresses output."""
        for rank in range(2):
            with self.subTest(rank=rank):
                self._init_fake_pg(rank)
                other_rank = 1 - rank
                fwd_rec: list[torch.Size] = []
                bwd_rec: list[torch.Size] = []
                cp = make_compile_print(
                    fwd_f=lambda t: fwd_rec.append(t.shape),
                    bwd_f=lambda t: bwd_rec.append(t.shape),
                )

                x = torch.randn(3, 3, requires_grad=True)
                buf = StringIO()
                with redirect_stdout(buf):
                    cp(x, tag="filtered", ranks=other_rank)
                    x.sum().backward()
                output = buf.getvalue()
                self.assertEqual(output, "")
                self.assertEqual(len(fwd_rec), 0)
                self.assertEqual(len(bwd_rec), 0)
                dist.destroy_process_group()

    def test_ranks_filter_set(self):
        """ranks={0, 1} enables current rank."""
        for rank in range(2):
            with self.subTest(rank=rank):
                self._init_fake_pg(rank)
                fwd_rec: list[torch.Size] = []
                cp = make_compile_print(
                    fwd_f=lambda t: fwd_rec.append(t.shape),
                    bwd_f=lambda t: None,
                )

                x = torch.randn(3, 3)
                cp(x, tag="both", ranks={0, 1})
                self.assertEqual(len(fwd_rec), 1)
                dist.destroy_process_group()

    def test_no_ranks_kwarg_prints_all(self):
        """Without ranks kwarg, all ranks execute callbacks."""
        for rank in range(2):
            with self.subTest(rank=rank):
                self._init_fake_pg(rank)
                fwd_rec: list[torch.Size] = []
                cp = make_compile_print(
                    fwd_f=lambda t: fwd_rec.append(t.shape),
                    bwd_f=lambda t: None,
                )

                x = torch.randn(3, 3)
                cp(x, tag="all")
                self.assertEqual(len(fwd_rec), 1)
                dist.destroy_process_group()


if __name__ == "__main__":
    run_tests()
