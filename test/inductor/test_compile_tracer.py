# Owner(s): ["module: inductor"]
import torch
from torch._dynamo import mark_dynamic
from torch._dynamo.decorators import mark_unbacked
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


def _placeholder_shape(compiled):
    """The shape of the first placeholder in the make_fx-captured graph."""
    ph = next(n for n in compiled.gm.graph.nodes if n.op == "placeholder")
    return ph.meta["val"].shape


class TestCompileTracer(TestCase):
    def test_unknown_tracer_raises(self):
        def f(x):
            return x + 1

        with self.assertRaisesRegex(RuntimeError, "Unknown tracer"):
            torch.compile(f, tracer="bogus")

    def test_make_fx_returns_wrapper(self):
        def f(x):
            return x + 1

        compiled = torch.compile(f, tracer="make_fx")
        self.assertEqual(type(compiled).__name__, "_MakeFxTracerWrapper")

    def test_disable_returns_original(self):
        def f(x):
            return x + 1

        compiled = torch.compile(f, tracer="make_fx", disable=True)
        self.assertIs(compiled, f)

    def test_mark_unbacked_produces_unbacked_symbol(self):
        def f(x):
            return (x + 1).relu()

        x = torch.randn(8, 4)
        mark_unbacked(x, 0)
        compiled = torch.compile(f, tracer="make_fx")
        compiled(x)
        d0, d1 = _placeholder_shape(compiled)
        self.assertTrue(str(d0).startswith("u"), f"expected unbacked symbol, got {d0}")
        self.assertEqual(d1, 4)

    def test_mark_dynamic_produces_backed_symbol(self):
        def f(x):
            return (x + 1).relu()

        x = torch.randn(8, 4)
        mark_dynamic(x, 0)
        compiled = torch.compile(f, tracer="make_fx")
        compiled(x)
        d0, d1 = _placeholder_shape(compiled)
        self.assertTrue(str(d0).startswith("s"), f"expected backed symbol, got {d0}")
        self.assertEqual(d1, 4)

    def test_no_annotation_static_is_specialized(self):
        def f(x):
            return (x + 1).relu()

        x = torch.randn(8, 4)
        compiled = torch.compile(f, tracer="make_fx")
        compiled(x)
        self.assertEqual(tuple(_placeholder_shape(compiled)), (8, 4))

    def test_dynamic_true_all_dims_backed(self):
        def f(x):
            return (x + 1).relu()

        x = torch.randn(8, 4)
        compiled = torch.compile(f, tracer="make_fx", dynamic=True)
        compiled(x)
        self.assertTrue(
            all(str(d).startswith("s") for d in _placeholder_shape(compiled))
        )


class TestCompileTracerNumerics(TestCase):
    @parametrize("tracer", ["dynamo", "make_fx"])
    def test_function_positional(self, device, tracer):
        def f(x, y):
            return (x + y).relu() * 2

        x = torch.randn(8, 8, device=device)
        y = torch.randn(8, 8, device=device)
        compiled = torch.compile(f, tracer=tracer)
        self.assertEqual(compiled(x, y), f(x, y))

    def test_make_fx_kwargs(self, device):
        def f(x, y):
            return x * y + x

        x = torch.randn(4, 5, device=device)
        y = torch.randn(4, 5, device=device)
        compiled = torch.compile(f, tracer="make_fx")
        self.assertEqual(compiled(x, y=y), f(x, y=y))

    def test_make_fx_cached_across_calls(self, device):
        calls = 0

        def f(x):
            nonlocal calls
            calls += 1
            return x.sin() + x.cos()

        x = torch.randn(6, device=device)
        compiled = torch.compile(f, tracer="make_fx")
        first = compiled(x)
        second = compiled(x)
        # f runs once at trace time; subsequent calls hit the cached graph.
        self.assertEqual(calls, 1)
        self.assertEqual(first, f(x))
        self.assertEqual(second, f(x))

    def test_make_fx_decorator(self, device):
        @torch.compile(tracer="make_fx")
        def g(x):
            return x.tanh() + 1.0

        x = torch.randn(3, 3, device=device)
        self.assertEqual(g(x), x.tanh() + 1.0)

    def test_make_fx_dynamic_reuses_kernel_across_shapes(self, device):
        calls = 0

        def f(x, y):
            nonlocal calls
            calls += 1
            return (x + y).relu() * 2

        compiled = torch.compile(f, tracer="make_fx", dynamic=True)
        x = torch.randn(8, device=device)
        y = torch.randn(8, device=device)
        self.assertEqual(compiled(x, y), f(x, y))

        # A different size must reuse the single dynamic kernel; f is only
        # ever run once, at trace time.
        x2 = torch.randn(20, device=device)
        y2 = torch.randn(20, device=device)
        self.assertEqual(compiled(x2, y2), f(x2, y2))
        self.assertEqual(calls, 3)

    def test_make_fx_static_specializes(self, device):
        def f(x):
            return x.sin()

        compiled = torch.compile(f, tracer="make_fx", dynamic=False)
        x = torch.randn(8, device=device)
        self.assertEqual(compiled(x), f(x))

    def test_make_fx_mark_unbacked_runs_across_sizes(self, device):
        def f(x):
            return (x + 1).relu() * 2

        x = torch.randn(8, 4, device=device)
        mark_unbacked(x, 0)
        compiled = torch.compile(f, tracer="make_fx")
        self.assertEqual(compiled(x), f(x))

        # The unbacked kernel must serve other sizes, including small ones that
        # a specialized kernel would have baked in.
        for n in (16, 1, 3):
            xr = torch.randn(n, 4, device=device)
            self.assertEqual(compiled(xr), f(xr))

    def test_make_fx_nn_module(self, device):
        mod = torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 4),
        ).to(device)
        x = torch.randn(2, 8, device=device)
        compiled = torch.compile(mod, tracer="make_fx")
        self.assertEqual(compiled(x), mod(x))


instantiate_device_type_tests(TestCompileTracerNumerics, globals())


if __name__ == "__main__":
    run_tests()
