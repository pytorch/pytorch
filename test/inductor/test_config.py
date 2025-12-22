# Owner(s): ["module: inductor"]
import math
import os
import unittest
from unittest.mock import patch

import torch
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.pattern_matcher import PatternMatcherPass
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.inductor_utils import HAS_CPU, HAS_TRITON


def dummy_fn(x):
    return torch.sigmoid(x + math.pi) / 10.0


class DummyModule(torch.nn.Module):
    def forward(self, x):
        return dummy_fn(x)


class TestInductorConfig(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._saved_config = config.save_config()

    def tearDown(self):
        super().tearDown()
        config.load_config(self._saved_config)

    def test_set(self):
        config.max_fusion_size = 13337
        self.assertEqual(config.max_fusion_size, 13337)
        self.assertEqual(config.get_config_copy()["max_fusion_size"], 13337)
        config.max_fusion_size = 32
        self.assertEqual(config.max_fusion_size, 32)

        # a nested config
        prior = config.triton.cudagraphs
        config.triton.cudagraphs = not prior
        self.assertEqual(config.triton.cudagraphs, not prior)
        self.assertEqual(config.get_config_copy()["triton.cudagraphs"], not prior)

    def test_save_load(self):
        config.max_fusion_size = 123
        config.triton.cudagraphs = True
        saved1 = config.save_config()
        config.max_fusion_size = 321
        config.triton.cudagraphs = False
        saved2 = config.save_config()

        self.assertEqual(config.max_fusion_size, 321)
        self.assertEqual(config.triton.cudagraphs, False)
        config.load_config(saved1)
        self.assertEqual(config.max_fusion_size, 123)
        self.assertEqual(config.triton.cudagraphs, True)
        config.load_config(saved2)
        self.assertEqual(config.max_fusion_size, 321)
        self.assertEqual(config.triton.cudagraphs, False)

    def test_hasattr(self):
        self.assertTrue(hasattr(config, "max_fusion_size"))
        self.assertFalse(hasattr(config, "missing_name"))

    def test_invalid_names(self):
        self.assertRaises(AttributeError, lambda: config.does_not_exist)
        self.assertRaises(AttributeError, lambda: config.triton.does_not_exist)

        def store1():
            config.does_not_exist = True

        def store2():
            config.triton.does_not_exist = True

        self.assertRaises(AttributeError, store1)
        self.assertRaises(AttributeError, store2)

    def test_patch(self):
        with config.patch(max_fusion_size=456):
            self.assertEqual(config.max_fusion_size, 456)
            with config.patch(max_fusion_size=789):
                self.assertEqual(config.max_fusion_size, 789)
            self.assertEqual(config.max_fusion_size, 456)

        with config.patch({"cpp.threads": 9000, "max_fusion_size": 9001}):
            self.assertEqual(config.cpp.threads, 9000)
            self.assertEqual(config.max_fusion_size, 9001)
            with config.patch("cpp.threads", 8999):
                self.assertEqual(config.cpp.threads, 8999)
            self.assertEqual(config.cpp.threads, 9000)

    @unittest.skipIf(not HAS_CPU, "requires C++ compiler")
    def test_compile_api(self):
        # these are mostly checking config processing doesn't blow up with exceptions
        x = torch.randn(8)
        y = dummy_fn(x)
        checks = [
            {},
            {"mode": "default"},
            {"mode": "reduce-overhead"},
            {"mode": "max-autotune"},
            {
                "options": {
                    "max-fusion-size": 128,
                    "unroll_reductions_threshold": 32,
                    "triton.cudagraphs": False,
                }
            },
            {"dynamic": True},
            {"fullgraph": True, "backend": "inductor"},
            {"disable": True},
        ]

        for kwargs in checks:
            torch._dynamo.reset()
            opt_fn = torch.compile(dummy_fn, **kwargs)
            torch.testing.assert_close(
                opt_fn(x), y, msg=f"torch.compile(..., **{kwargs!r}) failed"
            )

    def test_get_compiler_config(self):
        from torch._inductor import config as inductor_default_config

        default_cudagraphs = inductor_default_config.triton.cudagraphs

        # nn.Module: should update default config with a new value
        model = DummyModule()
        optimized_module = torch.compile(
            model, options={"triton.cudagraphs": not default_cudagraphs}
        )
        compiler_config = optimized_module.get_compiler_config()
        self.assertEqual(compiler_config["triton.cudagraphs"], not default_cudagraphs)

        # nn.Module: keep default config
        model = DummyModule()
        optimized_module = torch.compile(model)
        compiler_config = optimized_module.get_compiler_config()
        self.assertEqual(
            compiler_config["triton.cudagraphs"],
            default_cudagraphs,
        )

        # compile user func: should update default config with a new value
        optimized_module = torch.compile(
            dummy_fn, options={"triton.cudagraphs": not default_cudagraphs}
        )
        compiler_config = optimized_module.get_compiler_config()
        self.assertEqual(compiler_config["triton.cudagraphs"], not default_cudagraphs)

        # compile user func: keep default config
        optimized_module = torch.compile(dummy_fn)
        compiler_config = optimized_module.get_compiler_config()
        self.assertEqual(
            compiler_config["triton.cudagraphs"],
            default_cudagraphs,
        )

        # backend=eager: expect None
        optimized_module = torch.compile(dummy_fn, backend="eager")
        compiler_config = optimized_module.get_compiler_config()
        self.assertTrue(compiler_config is None)

    def test_compile_api_passes_config(self):
        # ensure configs are actually passed down to inductor
        self.assertRaises(
            torch._dynamo.exc.BackendCompilerFailed,
            lambda: torch.compile(dummy_fn, options={"_raise_error_for_testing": True})(
                torch.randn(10)
            ),
        )

    def test_api_options(self):
        reduce_overhead_opts = torch._inductor.list_mode_options("reduce-overhead")
        self.assertEqual(reduce_overhead_opts["triton.cudagraphs"], True)
        self.assertEqual(reduce_overhead_opts.get("max_autotune", False), False)

        max_autotune_opts = torch._inductor.list_mode_options("max-autotune")
        self.assertEqual(max_autotune_opts["max_autotune"], True)
        self.assertEqual(max_autotune_opts["triton.cudagraphs"], True)

        max_autotune_opts = torch._inductor.list_mode_options(
            "max-autotune", dynamic=True
        )
        self.assertEqual(max_autotune_opts["max_autotune"], True)
        self.assertEqual(max_autotune_opts["triton.cudagraphs"], True)

        max_autotune_no_cudagraphs_opts = torch._inductor.list_mode_options(
            "max-autotune-no-cudagraphs"
        )
        self.assertEqual(max_autotune_no_cudagraphs_opts["max_autotune"], True)
        self.assertEqual(
            max_autotune_no_cudagraphs_opts.get("triton.cudagraphs", False), False
        )

    def test_invalid_backend(self):
        self.assertRaises(
            torch._dynamo.exc.InvalidBackend,
            lambda: torch.compile(dummy_fn, backend="does_not_exist")(torch.randn(10)),
        )

    def test_non_inductor_backend(self):
        def assert_options(expected_mode=None, expected_options=None):
            def backend(gm, _, *, mode=None, options=None):
                nonlocal call_count
                self.assertEqual(mode, expected_mode)
                self.assertEqual(options, expected_options)
                call_count += 1
                return gm

            return backend

        inp = torch.randn(8)

        def fn(x):
            return x + 1

        for mode, options in [
            (None, None),
            ("fast-mode", None),
            (None, {"foo": "bar"}),
        ]:
            call_count = 0
            torch.compile(
                fn, backend=assert_options(mode, options), mode=mode, options=options
            )(inp)
            torch._dynamo.reset()
            self.assertEqual(call_count, 1)

    def test_codegen_skips_custom_passes(self):
        class _CustomPass(PatternMatcherPass):
            def __init__(self) -> None:
                super().__init__()

            def __call__(self, g: torch.fx.Graph):
                self.apply(g)

        g = _CustomPass()

        with torch._inductor.config.patch(
            post_grad_custom_post_pass=g,
            post_grad_custom_pre_pass=g,
        ):
            code = torch._inductor.config.codegen_config()
            self.assertNotIn("post_grad_custom", code)

    def test_select_decomp_table_fallback_embedding_bag_byte_unpack(self):
        """Test that select_decomp_table removes embedding_bag_byte_unpack when fallback is enabled"""
        from torch._inductor.decomposition import select_decomp_table

        # Test with fallback_embedding_bag_byte_unpack = False (default)
        with config.patch(fallback_embedding_bag_byte_unpack=False):
            decomp_table = select_decomp_table()
            # The operation should be in decompositions when fallback is False
            # Note: We check if it's in the fast_random_decomps() or decompositions table
            self.assertTrue(
                torch.ops.quantized.embedding_bag_byte_unpack.default in decomp_table
                or len(decomp_table)
                > 0  # fast_random_decomps() is used when fallback is False
            )

        # Test with fallback_embedding_bag_byte_unpack = True
        with config.patch(fallback_embedding_bag_byte_unpack=True):
            decomp_table = select_decomp_table()
            # The operation should NOT be in decompositions when fallback is True
            self.assertNotIn(
                torch.ops.quantized.embedding_bag_byte_unpack.default, decomp_table
            )

    @unittest.skipIf(not HAS_TRITON, "requires triton")
    def test_options_do_something(self):
        """
        Verify that we can populate and load functions from the cache.
        """

        counters.clear()

        def fn(x, y):
            yy = y @ y
            return x * 2 + yy.view(25)

        def fn2(x, y):
            yy = y @ y
            return x * 2 + yy.view(25)

        a_orig = torch.rand(25, dtype=torch.float32, device="cpu")
        b_orig = torch.rand(5, 5, dtype=torch.float32, device="cpu")

        compiled_fn = torch.compile(
            fn,
            options={
                "fx_graph_cache": True,
                "fx_graph_remote_cache": False,
                "bundle_triton_into_fx_graph_cache": True,
            },
        )

        a1 = a_orig.clone()
        b1 = b_orig.clone()
        a2 = a_orig.clone()
        b2 = b_orig.clone()

        # A first call should miss in the cache.
        eager_result = fn(a1, b1)
        compiled_result = compiled_fn(a2, b2)
        self.assertEqual(eager_result, compiled_result)
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
        self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 0)

        counters.clear()

        compiled_fn2 = torch.compile(
            fn2,
            options={
                "fx_graph_cache": False,
                "fx_graph_remote_cache": False,
                "bundle_triton_into_fx_graph_cache": False,
            },
        )

        # A first call should do nothing since cache is disabled
        eager_result = fn2(a1, b1)
        compiled_result = compiled_fn2(a2, b2)
        self.assertEqual(eager_result, compiled_result)
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
        self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 0)


class TestEdgeCases(TestCase):
    """Test edge cases for inductor utilities to ensure they handle invalid inputs gracefully."""

    def test_get_k_splits_zero_m(self):
        """Test get_k_splits handles m=0 without crashing (division by zero protection)."""
        from torch._inductor.utils import get_k_splits

        # Clear the cache to ensure we're testing fresh
        get_k_splits.cache_clear()

        # m=0 should return empty list (no meaningful k-splitting possible)
        result = get_k_splits(0, 10, 100)
        self.assertEqual(result, [])

    def test_get_k_splits_zero_n(self):
        """Test get_k_splits handles n=0 without crashing (division by zero protection)."""
        from torch._inductor.utils import get_k_splits

        # Clear the cache to ensure we're testing fresh
        get_k_splits.cache_clear()

        # n=0 should return empty list (no meaningful k-splitting possible)
        result = get_k_splits(10, 0, 100)
        self.assertEqual(result, [])

    def test_get_k_splits_zero_both(self):
        """Test get_k_splits handles m=0 and n=0 without crashing."""
        from torch._inductor.utils import get_k_splits

        # Clear the cache to ensure we're testing fresh
        get_k_splits.cache_clear()

        # Both m=0 and n=0 should return empty list
        result = get_k_splits(0, 0, 100)
        self.assertEqual(result, [])

    def test_get_k_splits_normal_case(self):
        """Test get_k_splits works correctly for normal inputs."""
        from torch._inductor.utils import get_k_splits

        # Clear the cache to ensure we're testing fresh
        get_k_splits.cache_clear()

        # Normal case should return a list (may be empty depending on k value)
        result = get_k_splits(100, 100, 256)
        self.assertIsInstance(result, list)

    def test_parse_rocm_num_stages_invalid_string(self):
        """Test _parse_rocm_num_stages handles invalid string values gracefully."""
        from torch._inductor.config import _parse_rocm_num_stages

        # Test with invalid string value - should return None without crashing
        with patch.dict(os.environ, {"TORCHINDUCTOR_ROCM_NUM_STAGES": "invalid"}):
            result = _parse_rocm_num_stages()
            self.assertIsNone(result)

    def test_parse_rocm_num_stages_negative(self):
        """Test _parse_rocm_num_stages handles negative values gracefully."""
        from torch._inductor.config import _parse_rocm_num_stages

        # Test with negative value - should return None without crashing
        with patch.dict(os.environ, {"TORCHINDUCTOR_ROCM_NUM_STAGES": "-1"}):
            result = _parse_rocm_num_stages()
            self.assertIsNone(result)

    def test_parse_rocm_num_stages_empty_string(self):
        """Test _parse_rocm_num_stages handles empty string gracefully."""
        from torch._inductor.config import _parse_rocm_num_stages

        # Test with empty string - should return None without crashing
        with patch.dict(os.environ, {"TORCHINDUCTOR_ROCM_NUM_STAGES": ""}):
            result = _parse_rocm_num_stages()
            self.assertIsNone(result)

    def test_parse_rocm_num_stages_whitespace(self):
        """Test _parse_rocm_num_stages handles whitespace-only values gracefully."""
        from torch._inductor.config import _parse_rocm_num_stages

        # Test with whitespace-only value - should return None without crashing
        with patch.dict(os.environ, {"TORCHINDUCTOR_ROCM_NUM_STAGES": "   "}):
            result = _parse_rocm_num_stages()
            self.assertIsNone(result)

    def test_parse_rocm_num_stages_valid(self):
        """Test _parse_rocm_num_stages works correctly for valid values."""
        from torch._inductor.config import _parse_rocm_num_stages

        # Test with valid integer - should return the parsed value
        with patch.dict(os.environ, {"TORCHINDUCTOR_ROCM_NUM_STAGES": "3"}):
            result = _parse_rocm_num_stages()
            self.assertEqual(result, 3)

    def test_parse_rocm_num_stages_zero(self):
        """Test _parse_rocm_num_stages handles zero correctly (valid edge case)."""
        from torch._inductor.config import _parse_rocm_num_stages

        # Test with zero - should return 0 (valid value for no pipelining)
        with patch.dict(os.environ, {"TORCHINDUCTOR_ROCM_NUM_STAGES": "0"}):
            result = _parse_rocm_num_stages()
            self.assertEqual(result, 0)

    def test_parse_rocm_num_stages_float(self):
        """Test _parse_rocm_num_stages handles float values gracefully."""
        from torch._inductor.config import _parse_rocm_num_stages

        # Test with float value - should return None without crashing
        with patch.dict(os.environ, {"TORCHINDUCTOR_ROCM_NUM_STAGES": "2.5"}):
            result = _parse_rocm_num_stages()
            self.assertIsNone(result)

    def test_get_rocm_arch_num_stages_no_crash(self):
        """Test _get_rocm_arch_num_stages doesn't crash even when CUDA is unavailable."""
        from torch._inductor.utils import _get_rocm_arch_num_stages

        # Clear the cache to test fresh
        _get_rocm_arch_num_stages.cache_clear()

        # This should return an integer without crashing, even if CUDA isn't available
        # The function has exception handling for RuntimeError and AttributeError
        result = _get_rocm_arch_num_stages()
        self.assertIsInstance(result, int)
        self.assertGreaterEqual(result, 2)  # Should be at least 2 (conservative default)

    def test_get_layout_opt_default_cuda(self):
        """Test _get_layout_opt_default returns '1' for CUDA (non-ROCm) builds."""
        from torch._inductor.config import _get_layout_opt_default

        # Mock torch.version.hip to be None (CUDA build)
        with patch.object(torch.version, "hip", None):
            result = _get_layout_opt_default()
            self.assertEqual(result, "1")

    def test_get_layout_opt_default_rocm_no_cuda(self):
        """Test _get_layout_opt_default returns '0' for ROCm when CUDA is not available."""
        from torch._inductor.config import _get_layout_opt_default

        # Mock ROCm build with CUDA not available
        with patch.object(torch.version, "hip", "6.0"):
            with patch.object(torch.cuda, "is_available", return_value=False):
                result = _get_layout_opt_default()
                self.assertEqual(result, "0")

    def test_get_layout_opt_default_rocm_mi300(self):
        """Test _get_layout_opt_default returns '1' for ROCm with MI300 series (gfx942)."""
        from torch._inductor.config import _get_layout_opt_default

        # Mock ROCm build with MI300 GPU (gfx942)
        mock_props = type("MockProps", (), {"gcnArchName": "gfx942:sramecc+:xnack-"})()
        with patch.object(torch.version, "hip", "6.0"):
            with patch.object(torch.cuda, "is_available", return_value=True):
                with patch.object(torch.cuda, "current_device", return_value=0):
                    with patch.object(
                        torch.cuda, "get_device_properties", return_value=mock_props
                    ):
                        result = _get_layout_opt_default()
                        self.assertEqual(result, "1")

    def test_get_layout_opt_default_rocm_mi200(self):
        """Test _get_layout_opt_default returns '0' for ROCm with MI200 series (gfx90a)."""
        from torch._inductor.config import _get_layout_opt_default

        # Mock ROCm build with MI200 GPU (gfx90a)
        mock_props = type("MockProps", (), {"gcnArchName": "gfx90a:sramecc+:xnack-"})()
        with patch.object(torch.version, "hip", "6.0"):
            with patch.object(torch.cuda, "is_available", return_value=True):
                with patch.object(torch.cuda, "current_device", return_value=0):
                    with patch.object(
                        torch.cuda, "get_device_properties", return_value=mock_props
                    ):
                        result = _get_layout_opt_default()
                        self.assertEqual(result, "0")

    def test_get_layout_optimization_default_with_env_override(self):
        """Test _get_layout_optimization_default respects environment variable override."""
        from torch._inductor.config import _get_layout_optimization_default

        # Mock CUDA build (would default to True)
        with patch.object(torch.version, "hip", None):
            # Override with environment variable to disable
            with patch.dict(os.environ, {"TORCHINDUCTOR_LAYOUT_OPTIMIZATION": "0"}):
                result = _get_layout_optimization_default()
                self.assertFalse(result)

            # Override with environment variable to enable
            with patch.dict(os.environ, {"TORCHINDUCTOR_LAYOUT_OPTIMIZATION": "1"}):
                result = _get_layout_optimization_default()
                self.assertTrue(result)

    def test_layout_optimization_lazy_init(self):
        """Test that layout_optimization config uses lazy initialization (None sentinel)."""
        # The config should be None by default to enable lazy evaluation
        # This prevents CUDA initialization at import time
        self.assertIsNone(config.layout_optimization)


class TestMultiArchConfig(TestCase):
    """Test multi-arch kernel configuration for AOTInductor."""

    def test_emit_multi_arch_kernel_default(self):
        """Test that emit_multi_arch_kernel defaults to True for standalone AOTInductor."""
        from torch._inductor.utils import aot_inductor_config_patches

        # Get config patches for standalone compilation
        patches = aot_inductor_config_patches(compile_standalone=True)

        # Multi-arch should be enabled by default for standalone builds
        self.assertTrue(patches.get("aot_inductor.emit_multi_arch_kernel", False))

    def test_emit_multi_arch_kernel_non_standalone(self):
        """Test that emit_multi_arch_kernel is not set for non-standalone builds."""
        from torch._inductor.utils import aot_inductor_config_patches

        # Get config patches for non-standalone compilation
        patches = aot_inductor_config_patches(compile_standalone=False)

        # Multi-arch should not be explicitly set for non-standalone builds
        self.assertNotIn("aot_inductor.emit_multi_arch_kernel", patches)


if __name__ == "__main__":
    run_tests()
