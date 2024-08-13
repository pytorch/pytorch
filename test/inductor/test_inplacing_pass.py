# Owner(s): ["module: inductor"]

import torch
from functorch import make_fx
from torch._dynamo.utils import counters
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._inductor.fx_passes.reinplace import reinplace_inplaceable_ops_core
from torch._inductor.test_case import run_tests, TestCase as InductorTestCase
from torch.testing._internal.common_utils import IS_LINUX
from torch.testing._internal.inductor_utils import HAS_CUDA


aten = torch.ops.aten


const = torch.tensor(0.0)
device = "cuda"


def num_reinplacing_failures():
    return counters["inductor"]["possibly_missed_reinplacing_opportunities"]


@torch.library.custom_op("_reinplacing::sin", mutates_args={"out"})
def sin(x: torch.Tensor, out: torch.Tensor) -> None:
    out.copy_(x.sin())


@torch.library.custom_op("_reinplacing::sin_cos", mutates_args={"out_sin", "out_cos"})
def sin_cos(x: torch.Tensor, out_sin: torch.Tensor, out_cos: torch.Tensor) -> None:
    out_sin.copy_(x.sin())
    out_cos.copy_(x.cos())


class TestReinplacingPassCorrectness(InductorTestCase):
    def setUp(self):
        counters.clear()
        return super().setUp()

    def _test(self, f):
        nf = torch.compile(f)
        inp = (
            torch.randn(4, device=device),
            torch.ones(2, device=device, dtype=torch.int),
        )
        inp2 = (inp[0].clone(), inp[1].clone())
        self.assertEqual(f(*inp), nf(*inp2))
        self.assertEqual(inp, inp2)

    def test_dont_modify_live(self):
        def f(x, y):
            x = x.cos()
            x2 = x.index_put((y,), const)
            return x2, x

        self._test(f)

    def test_dont_modify_view_of_live(self):
        def f(x, y):
            x = x.cos()
            x2 = aten.alias(x)
            x2 = x2.index_put((y,), const)
            y = x2 + x.cos()
            return y

        self._test(f)

    def test_dont_modify_input(self):
        def f(x, y):
            return x.index_put((y,), const)

        self._test(f)

    def test_should_modify_inner(self):
        def f(x, y):
            x = x.cos()
            x = x.index_put((y,), const)
            return x

        self._test(f)

    def test_should_modify_input(self):
        def f(x, y):
            x = x.index_put_((y,), const)
            return x

        self._test(f)

    def test_counters(self):
        counters.clear()

        def f(x):
            out = torch.empty_like(x)
            _, new_out = auto_functionalized(sin._opoverload, x=x, out=out)
            y = out * new_out
            return new_out, y

        x = torch.randn(3, device=device)
        gm = make_fx(f, tracing_mode="fake")(x)
        reinplace_inplaceable_ops_core(gm.graph)

        # We shouldn't have been able to reinplace `out` because it was used after
        # auto_functionalized. Note that this usually doesn't happen in practice;
        # we're artificially creating this example to test the counter.
        # IF THIS NUMBER GOES TO ZERO, PLEASE FIND ANOTHER EXAMPLE
        self.assertEqual(num_reinplacing_failures(), 1)

    def test_multi_output_intermediate(self):
        for requires_grad in [False, True]:
            counters.clear()

            def f(x):
                out1 = torch.empty_like(x)
                out2 = torch.empty_like(x)
                sin_cos(x, out1, out2)
                return out1, out2, x**2

            x = torch.randn(3, device=device, requires_grad=requires_grad)
            res1, res2, _ = torch.compile(f)(x)
            self.assertEqual(res1, x.sin())
            self.assertEqual(res2, x.cos())
            self.assertEqual(num_reinplacing_failures(), 0)

    def test_multiple_mutations(self):
        counters.clear()

        def f(x, out):
            sin(x, out)
            sin(out, out)
            sin(out, out)
            return out

        x = torch.randn(3, device=device)
        out = torch.randn(3, device=device)
        result = torch.compile(f)(x, out)
        self.assertEqual(result, x.sin().sin().sin())
        self.assertEqual(result, out)
        self.assertEqual(num_reinplacing_failures(), 0)

    def test_multiple_intermediate(self):
        counters.clear()

        def f(x):
            out = torch.empty_like(x)
            sin(x, out)
            sin(out, out)
            sin(out, out)
            return out

        x = torch.randn(3, device=device)
        result = torch.compile(f)(x)
        self.assertEqual(result, x.sin().sin().sin())
        self.assertEqual(num_reinplacing_failures(), 0)

    def test_backward(self):
        @torch.library.custom_op("mylib::foo", mutates_args={})
        def foo(x: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(x)

        @foo.register_fake
        def _(x):
            return torch.empty_like(x)

        class MySin(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                out = foo(x)
                sin(x, out)
                ctx.save_for_backward(out, x)
                return out

            @staticmethod
            def backward(ctx, grad):
                saved, x = ctx.saved_tensors
                out = foo(x)
                out.diag().fill_(1)
                sin(saved, out)
                return out

        @torch.compile
        def f(x):
            return MySin.apply(x)

        x = torch.randn(3, 3, requires_grad=True, device=device)
        y = f(x)
        self.assertEqual(num_reinplacing_failures(), 0)


class TestMutationRegionId(InductorTestCase):
    def test_alias_info(self):
        def f(a):
            b = a.cos()
            c = a[0]
            d = c.sin()
            e = c.view(-1)
            return b, c, d, e

        x = torch.randn(3)
        gm = make_fx(f, tracing_mode="fake")(x)

        from torch._inductor.fx_utils import AliasInfo

        alias_info = AliasInfo(gm)
        a, cos, select, sin, view, output = gm.graph.nodes

        # Basic tests
        a_aliases = {ref() for ref in alias_info.find_aliases(a)}
        self.assertEqual(a_aliases, {a, select, view})
        cos_aliases = {ref() for ref in alias_info.find_aliases(cos)}
        self.assertEqual(cos_aliases, {cos})

        # Test incremental update
        with gm.graph.inserting_after(view):
            view2 = gm.graph.call_function(torch.ops.aten.view.default, args=(view, -1))
            view2.meta["val"] = view.meta["val"].view(-1)

        a_aliases = {ref() for ref in alias_info.find_aliases(a)}
        self.assertEqual(a_aliases, {a, select, view, view2})

    def test_mutations(self):
        def f(a):
            b = a.clone()
            c = a.clone()
            a.set_(b)
            d = a.clone()
            e = a.clone()
            a.set_(b)
            f = a.clone()
            g = a.clone()
            return g

        x = torch.randn(3)
        gm = make_fx(f, tracing_mode="fake")(x)
        from torch._inductor.pattern_matcher import compute_mutation_region_ids

        compute_mutation_region_ids(gm.graph)
        ids = [str(n.meta["mutation_region_id"]) for n in gm.graph.nodes]
        # Expect to see three barrier_ids and no reinplace_ids
        self.assertExpectedInline(
            "\n".join(ids),
            """\
MutationRegionId(barrier_id=0, reinplace_id=None)
MutationRegionId(barrier_id=0, reinplace_id=None)
MutationRegionId(barrier_id=0, reinplace_id=None)
MutationRegionId(barrier_id=1, reinplace_id=None)
MutationRegionId(barrier_id=1, reinplace_id=None)
MutationRegionId(barrier_id=1, reinplace_id=None)
MutationRegionId(barrier_id=2, reinplace_id=None)
MutationRegionId(barrier_id=2, reinplace_id=None)
MutationRegionId(barrier_id=2, reinplace_id=None)
MutationRegionId(barrier_id=2, reinplace_id=None)""",
        )

    def test_auto_functionalized(self):
        def f(a):
            b = a.clone()
            c = a.clone()
            d = b.view(-1)
            e = torch.ops.higher_order.auto_functionalized(sin._opoverload, x=c, out=b)
            g = a.clone()
            a.set_(c)
            h = a.clone()
            return h

        x = torch.randn(3)
        gm = make_fx(f, tracing_mode="fake")(x)
        from torch._inductor.pattern_matcher import compute_mutation_region_ids

        compute_mutation_region_ids(gm.graph)
        ids = [str(n.meta["mutation_region_id"]) for n in gm.graph.nodes]
        # b and d should have their own reinplace_id
        self.assertExpectedInline(
            "\n".join(ids),
            """\
MutationRegionId(barrier_id=0, reinplace_id=None)
MutationRegionId(barrier_id=0, reinplace_id=0)
MutationRegionId(barrier_id=0, reinplace_id=None)
MutationRegionId(barrier_id=0, reinplace_id=0)
MutationRegionId(barrier_id=0, reinplace_id=None)
MutationRegionId(barrier_id=0, reinplace_id=None)
MutationRegionId(barrier_id=0, reinplace_id=None)
MutationRegionId(barrier_id=0, reinplace_id=None)
MutationRegionId(barrier_id=1, reinplace_id=None)
MutationRegionId(barrier_id=1, reinplace_id=None)
MutationRegionId(barrier_id=1, reinplace_id=None)""",
        )

    def test_auto_functionalized_incremental(self):
        def f(a):
            b = a.clone()
            c = a.clone()
            d = b.view(-1)
            e = torch.ops.higher_order.auto_functionalized(sin._opoverload, x=c, out=b)
            g = a.clone()
            a.set_(g)
            h = a.clone()
            return h

        x = torch.randn(3)
        gm = make_fx(f, tracing_mode="fake")(x)
        from torch._inductor.pattern_matcher import (
            compute_mutation_region_ids,
            get_mutation_region_id,
        )

        compute_mutation_region_ids(gm.graph)
        ids = [str(n.meta["mutation_region_id"]) for n in gm.graph.nodes]
        self.assertExpectedInline(
            "\n".join(ids),
            """\
MutationRegionId(barrier_id=0, reinplace_id=None)
MutationRegionId(barrier_id=0, reinplace_id=0)
MutationRegionId(barrier_id=0, reinplace_id=None)
MutationRegionId(barrier_id=0, reinplace_id=0)
MutationRegionId(barrier_id=0, reinplace_id=None)
MutationRegionId(barrier_id=0, reinplace_id=None)
MutationRegionId(barrier_id=0, reinplace_id=None)
MutationRegionId(barrier_id=0, reinplace_id=None)
MutationRegionId(barrier_id=1, reinplace_id=None)
MutationRegionId(barrier_id=1, reinplace_id=None)
MutationRegionId(barrier_id=1, reinplace_id=None)""",
        )

        graph = gm.graph
        nodes = list(graph.nodes)
        a, b, c, d, *_ = nodes

        with gm.graph.inserting_after(d):
            af = gm.graph.call_function(
                torch.ops.higher_order.auto_functionalized,
                args=(sin._opoverload,),
                kwargs={"x": a, "out": c},
            )
            af.meta["val"] = None
            # Unrelated
            k = gm.graph.call_function(torch.ops.aten.clone.default, args=(a,))
            k.meta["val"] = a.meta["val"].clone()

        get_mutation_region_id(gm.graph, af)
        ids = [str(n.meta["mutation_region_id"]) for n in gm.graph.nodes]
        self.assertExpectedInline(
            "\n".join(ids),
            """\
MutationRegionId(barrier_id=0, reinplace_id=None)
MutationRegionId(barrier_id=0, reinplace_id=0)
MutationRegionId(barrier_id=0, reinplace_id=1)
MutationRegionId(barrier_id=0, reinplace_id=0)
MutationRegionId(barrier_id=0, reinplace_id=None)
MutationRegionId(barrier_id=0, reinplace_id=None)
MutationRegionId(barrier_id=0, reinplace_id=None)
MutationRegionId(barrier_id=0, reinplace_id=None)
MutationRegionId(barrier_id=0, reinplace_id=None)
MutationRegionId(barrier_id=0, reinplace_id=None)
MutationRegionId(barrier_id=1, reinplace_id=None)
MutationRegionId(barrier_id=1, reinplace_id=None)
MutationRegionId(barrier_id=1, reinplace_id=None)""",
        )


if __name__ == "__main__":
    if IS_LINUX and HAS_CUDA:
        run_tests(needs="filelock")
