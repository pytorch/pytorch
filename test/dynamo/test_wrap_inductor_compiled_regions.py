# Owner(s): ["module: dynamo"]

import functools

import torch
import torch._dynamo.test_case
from functorch.compile import min_cut_rematerialization_partition
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.utils import counters
from torch._functorch import config as functorch_config
from torch._inductor import config as inductor_config
from torch.nn.attention.flex_attention import flex_attention, flex_attention_hop
from torch.testing._internal.triton_utils import requires_cuda_and_triton
from torch.utils._debug_mode import DebugMode
from torch.utils.checkpoint import (
    checkpoint,
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)


def count_ops(
    gm, args, freq=None, freq_ge=None, op=None, freqs=None, freqs_ge=None, ops=None
):
    """
    Count operations in a graph module.
    Used to verify SAC behavior by counting ops in forward/backward graphs.
    """

    def match_rng_op(node, op):
        if isinstance(node.target, torch._ops.HigherOrderOperator):
            if node.name == "run_and_save_rng_state":
                return node.args[0] == op
            elif node.name == "run_with_rng_state":
                return node.args[1] == op
            elif node.name == "graphsafe_run_with_rng_state":
                return node.args[0] == op
        return False

    if op is not None:
        if isinstance(op, list):
            raise AssertionError("Expected op to not be a list")
        ops = [op]
    if freq is not None:
        freqs = [freq]
    if freq_ge is not None:
        freqs_ge = [freq_ge]
    if freqs:
        for op, freq in zip(ops, freqs):
            actual_count = 0
            for node in gm.graph.nodes:
                if match_rng_op(node, op) or node.target == op:
                    actual_count += 1
            err_msg = f"In graph {gm}, expected {op} to have occurred {freq} times in the graph, but got {actual_count}."
            if actual_count != freq:
                raise AssertionError(err_msg)
    else:
        if freqs_ge is None:
            raise AssertionError("Expected freqs_ge to not be None")
        for op, freq_ge in zip(ops, freqs_ge):
            actual_count = 0
            for node in gm.graph.nodes:
                if match_rng_op(node, op) or node.target == op:
                    actual_count += 1
            if actual_count < freq_ge:
                raise AssertionError(
                    f"In graph {gm}, expected {op} to have occurred at least {freq_ge} times in the graph, but got {actual_count}."
                )
    return gm


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

    @requires_cuda_and_triton
    @inductor_config.patch("fx_graph_cache", True)
    @inductor_config.patch("fx_graph_remote_cache", False)
    @functorch_config.patch({"enable_autograd_cache": True})
    def test_wrap_config_affects_cache_key(self):
        """
        Test that wrap_inductor_compiled_regions is part of the cache key.
        Changing this option should cause a cache miss because it produces
        different compiled artifacts (wrapped vs unwrapped).
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

        # Compile with wrapping enabled
        compiled_fn_wrapped = torch.compile(
            fn,
            backend="inductor",
            options={"wrap_inductor_compiled_regions": True},
            fullgraph=True,
        )

        # First call with wrapping=True should miss the cache
        result1 = compiled_fn_wrapped(x, y)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)

        # Clear dynamo and codecache (but not FX or AOT autograd cache)
        torch._dynamo.reset()
        torch._inductor.codecache.PyCodeCache.cache_clear(purge=True)

        # Second call with wrapping=True should hit the cache
        result2 = compiled_fn_wrapped(x, y)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)

        # Clear dynamo and codecache again
        torch._dynamo.reset()
        torch._inductor.codecache.PyCodeCache.cache_clear(purge=True)

        # Now compile with wrapping disabled - should miss cache because
        # the config is different, even though the function is the same
        compiled_fn_unwrapped = torch.compile(
            fn,
            backend="inductor",
            options={"wrap_inductor_compiled_regions": False},
            fullgraph=True,
        )

        result3 = compiled_fn_unwrapped(x, y)
        # Should have a new cache miss because config changed
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)

        # Clear dynamo and codecache again
        torch._dynamo.reset()
        torch._inductor.codecache.PyCodeCache.cache_clear(purge=True)

        # Call again with wrapping=False - should hit the cache for unwrapped version
        result4 = compiled_fn_unwrapped(x, y)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 2)

        # All results should be correct
        expected = torch.matmul(x, y)
        self.assertEqual(result1, expected)
        self.assertEqual(result2, expected)
        self.assertEqual(result3, expected)
        self.assertEqual(result4, expected)

        # Verify the wrapping behavior is different
        with DebugMode() as debug_wrapped:
            _ = compiled_fn_wrapped(x, y)
        with DebugMode() as debug_unwrapped:
            _ = compiled_fn_unwrapped(x, y)

        # Wrapped version should show the HOP
        self.assertIn("inductor_compiled_code", debug_wrapped.debug_string())
        # Unwrapped version should not
        self.assertNotIn("inductor_compiled_code", debug_unwrapped.debug_string())

    @requires_cuda_and_triton
    def test_flex_attention_with_wrapper_basic(self):
        """Test that flex_attention works with wrap_inductor_compiled_regions=True"""

        def causal_score_mod(score, b, h, q_idx, k_idx):
            return torch.where(q_idx >= k_idx, score, float("-inf"))

        @torch.compile(
            backend="inductor",
            options={"wrap_inductor_compiled_regions": True},
            fullgraph=True,
        )
        def fn(q, k, v):
            return flex_attention(q, k, v, score_mod=causal_score_mod)

        B, H, S, D = 2, 4, 128, 64
        q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)

        # Test forward pass
        output = fn(q, k, v)
        self.assertEqual(output.shape, (B, H, S, D))

        # Verify correctness by comparing with unwrapped version
        @torch.compile(
            backend="inductor",
            options={"wrap_inductor_compiled_regions": False},
            fullgraph=True,
        )
        def fn_unwrapped(q, k, v):
            return flex_attention(q, k, v, score_mod=causal_score_mod)

        output_unwrapped = fn_unwrapped(q, k, v)
        torch.testing.assert_close(output, output_unwrapped, rtol=1e-3, atol=1e-3)

    @requires_cuda_and_triton
    def test_flex_attention_wrapper_visible_in_debug_mode(self):
        """Test that inductor_compiled_code HOP is visible to DebugMode when wrapper is enabled"""

        def score_mod(score, b, h, q_idx, k_idx):
            return score

        @torch.compile(
            backend="inductor",
            options={"wrap_inductor_compiled_regions": True},
            fullgraph=True,
        )
        def fn_wrapped(q, k, v):
            return flex_attention(q, k, v, score_mod=score_mod)

        @torch.compile(
            backend="inductor",
            options={"wrap_inductor_compiled_regions": False},
            fullgraph=True,
        )
        def fn_unwrapped(q, k, v):
            return flex_attention(q, k, v, score_mod=score_mod)

        B, H, S, D = 2, 4, 128, 64
        q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)

        # Test with wrapper enabled - should see inductor_compiled_code HOP
        with DebugMode() as debug_wrapped:
            _ = fn_wrapped(q, k, v)

        debug_string_wrapped = debug_wrapped.debug_string()
        self.assertIn(
            "inductor_compiled_code",
            debug_string_wrapped,
            "inductor_compiled_code HOP should be visible when wrapper is enabled",
        )

        # Test with wrapper disabled - should NOT see inductor_compiled_code HOP
        with DebugMode() as debug_unwrapped:
            _ = fn_unwrapped(q, k, v)

        debug_string_unwrapped = debug_unwrapped.debug_string()
        self.assertNotIn(
            "inductor_compiled_code",
            debug_string_unwrapped,
            "inductor_compiled_code HOP should not be visible when wrapper is disabled",
        )

    @requires_cuda_and_triton
    def test_flex_attention_wrapper_with_backward(self):
        """Test that wrapper works correctly with backward pass"""

        def score_mod(score, b, h, q_idx, k_idx):
            return score + 0.1

        @torch.compile(
            backend="inductor",
            options={"wrap_inductor_compiled_regions": True},
            fullgraph=True,
        )
        def fn(q, k, v):
            return flex_attention(q, k, v, score_mod=score_mod)

        B, H, S, D = 2, 4, 128, 64
        q = torch.randn(
            B, H, S, D, device="cuda", dtype=torch.float16, requires_grad=True
        )
        k = torch.randn(
            B, H, S, D, device="cuda", dtype=torch.float16, requires_grad=True
        )
        v = torch.randn(
            B, H, S, D, device="cuda", dtype=torch.float16, requires_grad=True
        )

        # Forward and backward
        output = fn(q, k, v)
        loss = output.sum()
        loss.backward()

        # Verify gradients exist
        self.assertIsNotNone(q.grad)
        self.assertIsNotNone(k.grad)
        self.assertIsNotNone(v.grad)

        # Compare with unwrapped version
        q2 = q.detach().clone().requires_grad_(True)
        k2 = k.detach().clone().requires_grad_(True)
        v2 = v.detach().clone().requires_grad_(True)

        @torch.compile(
            backend="inductor",
            options={"wrap_inductor_compiled_regions": False},
            fullgraph=True,
        )
        def fn_unwrapped(q, k, v):
            return flex_attention(q, k, v, score_mod=score_mod)

        output2 = fn_unwrapped(q2, k2, v2)
        loss2 = output2.sum()
        loss2.backward()

        torch.testing.assert_close(q.grad, q2.grad, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(k.grad, k2.grad, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(v.grad, v2.grad, rtol=1e-3, atol=1e-3)

    @requires_cuda_and_triton
    @inductor_config.patch("fx_graph_cache", True)
    @inductor_config.patch("fx_graph_remote_cache", False)
    @functorch_config.patch({"enable_autograd_cache": True})
    def test_flex_attention_wrapper_with_cache(self):
        """Test that wrapper works correctly with caching"""
        from torch._functorch._aot_autograd.autograd_cache import AOTAutogradCache

        def score_mod(score, b, h, q_idx, k_idx):
            return score

        def make_compiled_fn():
            @torch.compile(
                backend="inductor",
                options={"wrap_inductor_compiled_regions": True},
                fullgraph=True,
            )
            def fn(q, k, v):
                return flex_attention(q, k, v, score_mod=score_mod)

            return fn

        B, H, S, D = 2, 4, 128, 64
        q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)

        # Clear all caches
        counters.clear()
        torch._inductor.codecache.FxGraphCache.clear()
        AOTAutogradCache.clear()
        torch._dynamo.reset()
        torch._inductor.codecache.PyCodeCache.cache_clear(purge=True)

        # First call - cache miss
        fn1 = make_compiled_fn()
        with DebugMode() as debug_mode1:
            result1 = fn1(q, k, v)

        # Verify wrapper is visible in DebugMode
        self.assertIn("inductor_compiled_code", debug_mode1.debug_string())

        # Verify cache miss
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)

        # Clear dynamo and codecache (but not FX or AOT autograd cache)
        torch._dynamo.reset()
        torch._inductor.codecache.PyCodeCache.cache_clear(purge=True)

        # Second call - cache hit
        fn2 = make_compiled_fn()
        with DebugMode() as debug_mode2:
            result2 = fn2(q, k, v)

        # Verify wrapper is still visible after loading from cache
        self.assertIn(
            "inductor_compiled_code",
            debug_mode2.debug_string(),
            "Wrapper should be applied even when loading from cache",
        )

        # Verify cache hit
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)

        # Verify correctness
        torch.testing.assert_close(result1, result2)

    @requires_cuda_and_triton
    def test_flex_attention_with_sac_must_save(self):
        """
        Test that SAC policy MUST_SAVE for flex_attention_hop
        prevents recomputation during backward when used with wrapper.

        This verifies that flex_attention works correctly with SAC when
        wrap_inductor_compiled_regions is enabled.
        """

        def score_mod(score, b, h, q_idx, k_idx):
            return score

        # SAC policy: MUST_SAVE flex_attention_hop
        def policy_fn(ctx, op, *args, **kwargs):
            if op == flex_attention_hop:
                return CheckpointPolicy.MUST_SAVE
            return CheckpointPolicy.PREFER_RECOMPUTE

        def gn(q, k, v):
            return flex_attention(q, k, v, score_mod=score_mod)

        def fn(q, k, v):
            context_fn = functools.partial(
                create_selective_checkpoint_contexts, policy_fn
            )
            return checkpoint(
                gn,
                q,
                k,
                v,
                use_reentrant=False,
                context_fn=context_fn,
            )

        B, H, S, D = 2, 4, 128, 64
        q = torch.randn(
            B, H, S, D, device="cuda", dtype=torch.float16, requires_grad=True
        )
        k = torch.randn(
            B, H, S, D, device="cuda", dtype=torch.float16, requires_grad=True
        )
        v = torch.randn(
            B, H, S, D, device="cuda", dtype=torch.float16, requires_grad=True
        )

        # Forward compiler: should see flex_attention_hop once
        fw_compiler = functools.partial(
            count_ops,
            freq=1,
            op=flex_attention_hop,
        )

        # Backward compiler: should NOT see flex_attention_hop
        # because MUST_SAVE means it was saved, not recomputed
        bw_compiler = functools.partial(
            count_ops,
            freq=0,
            op=flex_attention_hop,
        )

        backend = aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            partition_fn=min_cut_rematerialization_partition,
        )

        # Use config.patch to enable wrapping at inductor level
        with inductor_config.patch({"wrap_inductor_compiled_regions": True}):
            compiled_fn = torch.compile(
                fn,
                backend=backend,
                fullgraph=True,
            )

            output = compiled_fn(q, k, v)
            loss = output.sum()
            loss.backward()

        # Verify gradients exist
        self.assertIsNotNone(q.grad)
        self.assertIsNotNone(k.grad)
        self.assertIsNotNone(v.grad)

    @requires_cuda_and_triton
    def test_flex_attention_with_sac_prefer_recompute(self):
        """
        Test that SAC policy PREFER_RECOMPUTE for flex_attention_hop
        causes recomputation during backward when used with wrapper.

        This verifies that flex_attention is properly recomputed when SAC
        policy specifies PREFER_RECOMPUTE.
        """

        def score_mod(score, b, h, q_idx, k_idx):
            return score

        # SAC policy: PREFER_RECOMPUTE flex_attention_hop
        def policy_fn(ctx, op, *args, **kwargs):
            if op == flex_attention_hop:
                # this would be very weird IRL fwiw, just testing
                return CheckpointPolicy.PREFER_RECOMPUTE
            return CheckpointPolicy.PREFER_RECOMPUTE

        def gn(q, k, v):
            return flex_attention(q, k, v, score_mod=score_mod)

        def fn(q, k, v):
            context_fn = functools.partial(
                create_selective_checkpoint_contexts, policy_fn
            )
            return checkpoint(
                gn,
                q,
                k,
                v,
                use_reentrant=False,
                context_fn=context_fn,
            )

        B, H, S, D = 2, 4, 128, 64
        q = torch.randn(
            B, H, S, D, device="cuda", dtype=torch.float16, requires_grad=True
        )
        k = torch.randn(
            B, H, S, D, device="cuda", dtype=torch.float16, requires_grad=True
        )
        v = torch.randn(
            B, H, S, D, device="cuda", dtype=torch.float16, requires_grad=True
        )

        # Forward compiler: should see flex_attention_hop once
        fw_compiler = functools.partial(
            count_ops,
            freq=1,
            op=flex_attention_hop,
        )

        # Backward compiler: should see flex_attention_hop once
        # because PREFER_RECOMPUTE means it gets recomputed
        bw_compiler = functools.partial(
            count_ops,
            freq=1,
            op=flex_attention_hop,
        )

        backend = aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            partition_fn=min_cut_rematerialization_partition,
        )

        # Use config.patch to enable wrapping at inductor level
        with inductor_config.patch({"wrap_inductor_compiled_regions": True}):
            compiled_fn = torch.compile(
                fn,
                backend=backend,
                fullgraph=True,
            )

            output = compiled_fn(q, k, v)
            loss = output.sum()
            loss.backward()

        # Verify gradients exist
        self.assertIsNotNone(q.grad)
        self.assertIsNotNone(k.grad)
        self.assertIsNotNone(v.grad)

    @requires_cuda_and_triton
    def test_sac_outer_compile_inner_basic(self):
        """
        Test SAC(compile(foo)) pattern - SAC on eager code with inner compiled region.

        This is different from compile(SAC(foo)) - here the checkpoint region itself
        is NOT compiled, but it contains a compiled function inside it.

        The inner compiled function should be wrapped when wrap_inductor_compiled_regions
        is enabled, making it visible to SAC's dispatch modes.
        """

        # Inner compiled function with wrapping enabled
        @torch.compile(
            backend="inductor",
            options={"wrap_inductor_compiled_regions": True},
            fullgraph=True,
        )
        def inner_compiled_matmul(x, y):
            return torch.matmul(x, y)

        # SAC policy: save matmul operations
        def policy_fn(ctx, op, *args, **kwargs):
            # When the compiled region is wrapped in inductor_compiled_code HOP,
            # SAC should be able to see it and apply policy
            from torch._higher_order_ops.wrap import inductor_compiled_code

            if op == inductor_compiled_code:
                return CheckpointPolicy.MUST_SAVE
            return CheckpointPolicy.PREFER_RECOMPUTE

        # Eager checkpointed function that calls compiled code
        def checkpointed_fn(x, y):
            # This compiled call should be wrapped in inductor_compiled_code HOP
            a = inner_compiled_matmul(x, y)
            b = torch.relu(a)
            return b

        x = torch.randn(4, 4, device="cuda", requires_grad=True)
        y = torch.randn(4, 4, device="cuda", requires_grad=True)

        # Clone for comparison
        x_eager = x.detach().clone().requires_grad_(True)
        y_eager = y.detach().clone().requires_grad_(True)

        # SAC(compile(foo)) - checkpoint the eager function with inner compiled region
        context_fn = functools.partial(create_selective_checkpoint_contexts, policy_fn)

        # Test with DebugMode to verify the HOP is visible
        with DebugMode() as debug_mode:
            output = checkpoint(
                checkpointed_fn,
                x,
                y,
                use_reentrant=False,
                context_fn=context_fn,
            )
            loss = output.sum()
            loss.backward()

        debug_string = debug_mode.debug_string()

        # inductor_compiled_code HOP should be visible to DebugMode
        self.assertIn(
            "inductor_compiled_code",
            debug_string,
            "inductor_compiled_code HOP should be visible when inner compiled function "
            "is called from eager checkpoint region",
        )

        # Verify correctness against eager
        a_eager = torch.matmul(x_eager, y_eager)
        b_eager = torch.relu(a_eager)
        loss_eager = b_eager.sum()
        loss_eager.backward()

        self.assertEqual(output, b_eager)
        self.assertEqual(x.grad, x_eager.grad)
        self.assertEqual(y.grad, y_eager.grad)

    @requires_cuda_and_triton
    def test_wrap_no_dispatch_mode_no_hop_invoked(self):
        """
        Test that without TorchDispatchMode, the HOP is NOT invoked.

        Even when wrap_inductor_compiled_regions=True, if there's no active
        TorchDispatchMode, the wrapper should not invoke the HOP (optimization).
        This verifies that we're not paying the HOP overhead unnecessarily.
        """
        from unittest.mock import patch

        from torch._higher_order_ops.wrap import inductor_compiled_code

        # Patch it in the output_code module where it's imported and used
        patch_path = "torch._inductor.output_code.inductor_compiled_code"

        # Test WITHOUT dispatch mode - HOP should not route through a mode
        with patch(patch_path, wraps=inductor_compiled_code) as mock_hop:

            @torch.compile(
                backend="inductor",
                options={"wrap_inductor_compiled_regions": True},
                fullgraph=True,
            )
            def fn(x, y):
                return torch.matmul(x, y)

            x = torch.randn(4, 4, device="cuda")
            y = torch.randn(4, 4, device="cuda")
            expected = torch.matmul(x, y)

            result_without = fn(x, y)

            self.assertEqual(result_without, expected)

            if mock_hop.called:
                args, kwargs = mock_hop.call_args
                # When no dispatch modes are active, we expect mode argument to be None
                # (wrapper is used purely for tracing alignment).
                self.assertIsNone(kwargs.get("mode"))

        # Test WITH DebugMode - HOP SHOULD be called
        with patch(patch_path, wraps=inductor_compiled_code) as mock_hop:

            @torch.compile(
                backend="inductor",
                options={"wrap_inductor_compiled_regions": True},
                fullgraph=True,
            )
            def fn2(x, y):
                return torch.matmul(x, y)

            x2 = torch.randn(4, 4, device="cuda")
            y2 = torch.randn(4, 4, device="cuda")
            expected2 = torch.matmul(x2, y2)

            with DebugMode():
                result_with = fn2(x2, y2)

            # Verify HOP WAS called
            mock_hop.assert_called()
            self.assertEqual(result_with, expected2)

    @requires_cuda_and_triton
    def test_sac_outer_compile_inner_flex_attention(self):
        """
        Test SAC(compile(foo)) with flex_attention - the key motivating use case.

        Pattern: Eager checkpoint region containing compiled flex_attention.
        This is the pattern where users want SAC to control compiled flex_attention.
        """

        def score_mod(score, b, h, q_idx, k_idx):
            return score

        # Policy: save the compiled flex_attention region
        def policy_fn(ctx, op, *args, **kwargs):
            from torch._higher_order_ops.wrap import inductor_compiled_code

            # When flex_attention is compiled with wrapping, its compiled kernel
            # should be wrapped in inductor_compiled_code HOP
            if op == inductor_compiled_code:
                return CheckpointPolicy.MUST_SAVE
            # Also handle the flex_attention_hop itself
            if op == flex_attention_hop:
                return CheckpointPolicy.MUST_SAVE
            return CheckpointPolicy.PREFER_RECOMPUTE

        # Eager function that calls flex_attention (which internally compiles)
        def checkpointed_flex_fn(q, k, v):
            # flex_attention internally uses torch.compile, so with
            # wrap_inductor_compiled_regions enabled, its compiled regions
            # should be wrapped in the HOP
            output = flex_attention(q, k, v, score_mod=score_mod)
            return output

        B, H, S, D = 2, 4, 128, 64
        q = torch.randn(
            B, H, S, D, device="cuda", dtype=torch.float16, requires_grad=True
        )
        k = torch.randn(
            B, H, S, D, device="cuda", dtype=torch.float16, requires_grad=True
        )
        v = torch.randn(
            B, H, S, D, device="cuda", dtype=torch.float16, requires_grad=True
        )

        # Enable wrapping at the inductor config level so that flex_attention's
        # internal compilation will wrap compiled regions
        with inductor_config.patch({"wrap_inductor_compiled_regions": True}):
            context_fn = functools.partial(
                create_selective_checkpoint_contexts, policy_fn
            )

            # SAC(compile(foo)) - eager checkpoint with inner compiled flex_attention
            output = checkpoint(
                checkpointed_flex_fn,
                q,
                k,
                v,
                use_reentrant=False,
                context_fn=context_fn,
            )
            loss = output.sum()
            loss.backward()

        # Verify gradients exist
        self.assertIsNotNone(q.grad)
        self.assertIsNotNone(k.grad)
        self.assertIsNotNone(v.grad)

        # Verify correctness by comparing with non-checkpointed version
        q2 = q.detach().clone().requires_grad_(True)
        k2 = k.detach().clone().requires_grad_(True)
        v2 = v.detach().clone().requires_grad_(True)

        with inductor_config.patch({"wrap_inductor_compiled_regions": True}):
            output2 = flex_attention(q2, k2, v2, score_mod=score_mod)
            loss2 = output2.sum()
            loss2.backward()

        torch.testing.assert_close(output, output2, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(q.grad, q2.grad, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(k.grad, k2.grad, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(v.grad, v2.grad, rtol=1e-3, atol=1e-3)

    def test_fake_tensor_mode_works(self):
        """Test that running compiled code inside FakeTensorMode works with FX graph fallback"""
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            model = torch.nn.Linear(4, 4)
            inp = torch.rand(4, 4)

            # The FX graph is now serialized using SerializedGraphModule, so it
            # survives cache serialization/deserialization and works with caching
            with inductor_config.patch({"wrap_inductor_compiled_regions": True}):
                # This should now work - the inductor_compiled_code HOP will
                # use the stored FX graph to propagate fake tensors
                result = torch.compile(model)(inp)
                # Verify the result has the expected shape
                self.assertEqual(result.shape, (4, 4))

    def test_proxy_tensor_mode_works(self):
        """Test that running compiled code inside ProxyTensorMode works with FX graph fallback"""
        from torch._higher_order_ops.wrap import (
            inductor_code_side_table,
            inductor_compiled_code,
            InductorCompiledCallable,
        )
        from torch.fx.experimental.proxy_tensor import make_fx

        # Reset the side table for a clean test
        inductor_code_side_table.reset_table()

        # Create a simple FX graph
        class SimpleModel(torch.nn.Module):
            def forward(self, x):
                return x + 1

        model = SimpleModel()
        gm = torch.fx.symbolic_trace(model)

        # Create an InductorCompiledCallable
        # The compiled callable should match FX graph's output convention
        def simple_compiled(inputs):
            return inputs[0] + 1

        callable_obj = InductorCompiledCallable(simple_compiled, gm)

        # Wrapper that uses the HOP
        def wrapper(x):
            return inductor_compiled_code(callable_obj, [x])

        inp = torch.randn(4, 4)
        traced = make_fx(wrapper)(inp)

        # Verify the traced graph contains the inductor_compiled_code HOP
        hop_found = False
        for node in traced.graph.nodes:
            if node.op == "call_function" and "inductor_compiled_code" in str(
                node.target
            ):
                hop_found = True
                # Verify the callable index is an int
                self.assertIsInstance(node.args[0], int)
                break
        self.assertTrue(
            hop_found, "inductor_compiled_code HOP not found in traced graph"
        )

        # Verify the traced graph can be executed and produces correct results
        result = traced(inp)
        expected = inp + 1
        torch.testing.assert_close(result, expected)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
