# Owner(s): ["module: dynamo"]

import torch
import torch._dynamo.test_case
from torch._dynamo.utils import counters
from torch._functorch import config as functorch_config
from torch._inductor import config as inductor_config
from torch.testing._internal.triton_utils import requires_cuda_and_triton
from torch.utils._debug_mode import DebugMode


class TestWrapInductorCompiledRegions(torch._dynamo.test_case.TestCase):
    """Tests for wrap_inductor_compiled_regions option"""

    @requires_cuda_and_triton
    def test_wrap_enabled_visible_in_debug_mode(self):
        """Test that compiled regions are wrapped when option is enabled"""

        @torch.compile(
            backend="inductor",
            options={"wrap_inductor_compiled_regions": True},
            fullgraph=True,
        )
        def fn(x, y):
            return torch.matmul(x, y)

        x = torch.randn(4, 4, device="cuda")
        y = torch.randn(4, 4, device="cuda")

        with DebugMode() as debug_mode:
            result = fn(x, y)

        debug_string = debug_mode.debug_string()

        # inductor_compiled_code HOP should be visible in DebugMode
        self.assertIn("inductor_compiled_code", debug_string)

        # Result should be correct
        expected = torch.matmul(x, y)
        self.assertEqual(result, expected)

    @requires_cuda_and_triton
    def test_wrap_disabled_not_visible_in_debug_mode(self):
        """Test that compiled regions are not wrapped when option is disabled"""

        @torch.compile(
            backend="inductor",
            options={"wrap_inductor_compiled_regions": False},
            fullgraph=True,
        )
        def fn(x, y):
            return torch.matmul(x, y)

        x = torch.randn(4, 4, device="cuda")
        y = torch.randn(4, 4, device="cuda")

        with DebugMode() as debug_mode:
            result = fn(x, y)

        debug_string = debug_mode.debug_string()

        # inductor_compiled_code HOP should NOT be visible
        self.assertNotIn("inductor_compiled_code", debug_string)

        # Result should still be correct
        expected = torch.matmul(x, y)
        self.assertEqual(result, expected)

    @requires_cuda_and_triton
    def test_wrap_default_disabled(self):
        """Test that wrapping is disabled by default"""

        @torch.compile(backend="inductor", fullgraph=True)
        def fn(x, y):
            return torch.matmul(x, y)

        x = torch.randn(4, 4, device="cuda")
        y = torch.randn(4, 4, device="cuda")

        with DebugMode() as debug_mode:
            result = fn(x, y)

        debug_string = debug_mode.debug_string()

        # inductor_compiled_code HOP should NOT be visible by default
        self.assertNotIn("inductor_compiled_code", debug_string)

        # Result should be correct
        expected = torch.matmul(x, y)
        self.assertEqual(result, expected)

    @requires_cuda_and_triton
    def test_wrap_with_backward(self):
        """Test that wrapping works correctly with backward pass"""

        @torch.compile(
            backend="inductor",
            options={"wrap_inductor_compiled_regions": True},
            fullgraph=True,
        )
        def fn(x, y):
            return torch.matmul(x, y)

        x = torch.randn(4, 4, device="cuda", requires_grad=True)
        y = torch.randn(4, 4, device="cuda", requires_grad=True)

        # Clone for eager comparison
        x_eager = x.detach().clone().requires_grad_(True)
        y_eager = y.detach().clone().requires_grad_(True)

        # Compiled forward and backward
        with DebugMode() as debug_mode:
            result = fn(x, y)
            loss = result.sum()
            loss.backward()

        debug_string = debug_mode.debug_string()

        # inductor_compiled_code HOP should be visible in forward
        self.assertIn("inductor_compiled_code", debug_string)

        # Eager forward and backward
        expected = torch.matmul(x_eager, y_eager)
        expected_loss = expected.sum()
        expected_loss.backward()

        # Check correctness
        self.assertEqual(result, expected)
        self.assertEqual(x.grad, x_eager.grad)
        self.assertEqual(y.grad, y_eager.grad)

    @requires_cuda_and_triton
    def test_wrap_with_multiple_ops(self):
        """Test wrapping with a function that has multiple operations"""

        @torch.compile(
            backend="inductor",
            options={"wrap_inductor_compiled_regions": True},
            fullgraph=True,
        )
        def fn(x, y):
            a = torch.matmul(x, y)
            b = torch.relu(a)
            c = b + x
            return c

        x = torch.randn(4, 4, device="cuda")
        y = torch.randn(4, 4, device="cuda")

        with DebugMode() as debug_mode:
            result = fn(x, y)

        debug_string = debug_mode.debug_string()

        # inductor_compiled_code HOP should be visible
        self.assertIn("inductor_compiled_code", debug_string)

        # Result should be correct
        a = torch.matmul(x, y)
        b = torch.relu(a)
        expected = b + x
        self.assertEqual(result, expected)

    @requires_cuda_and_triton
    def test_wrap_option_type_validation(self):
        """Test that wrap_inductor_compiled_regions validates type correctly"""

        # Should accept bool
        @torch.compile(
            backend="inductor",
            options={"wrap_inductor_compiled_regions": True},
        )
        def fn_true(x):
            return x + 1

        @torch.compile(
            backend="inductor",
            options={"wrap_inductor_compiled_regions": False},
        )
        def fn_false(x):
            return x + 1

        x = torch.randn(4, device="cuda")
        _ = fn_true(x)
        _ = fn_false(x)

        # Should reject non-bool
        with self.assertRaises(RuntimeError) as cm:

            @torch.compile(
                backend="inductor",
                options={"wrap_inductor_compiled_regions": "true"},
            )
            def fn_invalid(x):
                return x + 1

        self.assertIn("Unexpected type", str(cm.exception))

    @requires_cuda_and_triton
    def test_wrap_per_compilation(self):
        """Test that wrap option is per-compilation, not global"""

        @torch.compile(
            backend="inductor",
            options={"wrap_inductor_compiled_regions": True},
            fullgraph=True,
        )
        def fn_wrapped(x, y):
            return torch.matmul(x, y)

        @torch.compile(
            backend="inductor",
            options={"wrap_inductor_compiled_regions": False},
            fullgraph=True,
        )
        def fn_not_wrapped(x, y):
            return torch.matmul(x, y)

        x = torch.randn(4, 4, device="cuda")
        y = torch.randn(4, 4, device="cuda")

        # First function should be wrapped
        with DebugMode() as debug_mode1:
            _ = fn_wrapped(x, y)
        self.assertIn("inductor_compiled_code", debug_mode1.debug_string())

        # Second function should not be wrapped
        with DebugMode() as debug_mode2:
            _ = fn_not_wrapped(x, y)
        self.assertNotIn("inductor_compiled_code", debug_mode2.debug_string())

    @requires_cuda_and_triton
    @inductor_config.patch("fx_graph_cache", True)
    @inductor_config.patch("fx_graph_remote_cache", False)
    @functorch_config.patch({"enable_autograd_cache": True})
    def test_wrap_with_cache(self):
        """
        Test that wrap_inductor_compiled_regions works correctly with caching.
        Verify that the wrapper is properly applied when loading from cache by
        checking that DebugMode can see the inductor_compiled_code HOP on both
        cache miss and cache hit.
        """
        from torch._functorch._aot_autograd.autograd_cache import AOTAutogradCache

        def fn(x, y):
            return torch.matmul(x, y)

        x = torch.randn(4, 4, device="cuda")
        y = torch.randn(4, 4, device="cuda")

        # Clear all caches and counters
        counters.clear()
        torch._inductor.codecache.FxGraphCache.clear()
        AOTAutogradCache.clear()
        torch._dynamo.reset()
        torch._inductor.codecache.PyCodeCache.cache_clear(purge=True)

        compiled_fn = torch.compile(
            fn,
            backend="inductor",
            options={"wrap_inductor_compiled_regions": True},
            fullgraph=True,
        )

        # First call should miss the cache
        with DebugMode() as debug_mode1:
            result1 = compiled_fn(x, y)

        debug_string1 = debug_mode1.debug_string()

        # Verify wrapper is applied and invoked on cache miss
        # If DebugMode sees the HOP, it means the wrapper was actually invoked
        # (because DebugMode is registered with redirect_to_mode)
        self.assertIn(
            "inductor_compiled_code",
            debug_string1,
            "inductor_compiled_code HOP should be visible to DebugMode on cache miss",
        )

        # Verify cache miss
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)

        # Clear dynamo and codecache (but not FX or AOT autograd cache)
        torch._dynamo.reset()
        torch._inductor.codecache.PyCodeCache.cache_clear(purge=True)

        # Second call should hit the cache
        with DebugMode() as debug_mode2:
            result2 = compiled_fn(x, y)

        debug_string2 = debug_mode2.debug_string()

        # Verify wrapper is still applied and invoked after loading from cache
        # This proves that post_compile() properly wraps the cached callable
        self.assertIn(
            "inductor_compiled_code",
            debug_string2,
            "inductor_compiled_code HOP should be visible to DebugMode on cache hit, "
            "proving wrapper was properly applied in post_compile()",
        )

        # Verify cache hit
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)

        # Results should be correct and identical
        expected = torch.matmul(x, y)
        self.assertEqual(result1, expected)
        self.assertEqual(result2, expected)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
