# Owner(s): ["module: inductor"]
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from types import SimpleNamespace
from unittest import mock

import torch
from torch._inductor.test_case import TestCase


try:
    import flydsl  # noqa: F401

    HAS_FLYDSL = True
except ImportError:
    HAS_FLYDSL = False

if HAS_FLYDSL:
    from torch._inductor.codegen.flydsl import flydsl_utils
    from torch._inductor.codegen.flydsl.flydsl_kernel import FlyDSLTemplateKernel


class TestFlyDSLTemplate(TestCase):
    def test_compiled_cache_keys_only_on_param(self):
        from torch._inductor.runtime.flydsl_cache import run_cached_flydsl

        jit_func = SimpleNamespace()
        compile_args = (object(),)
        dispatch_args = (object(),)
        compiled = mock.Mock()
        compiler = mock.Mock(return_value=compiled)

        class Param:
            def __cache_signature__(self):
                return ("param",)

        first = run_cached_flydsl(
            jit_func,
            *compile_args,
            constexpr_param=Param(),
            compiler=compiler,
            dispatch_args=dispatch_args,
        )
        second = run_cached_flydsl(
            jit_func,
            object(),
            constexpr_param=Param(),
            compiler=compiler,
            dispatch_args=dispatch_args,
        )

        self.assertIs(first, compiled)
        self.assertIs(second, compiled)
        compiler.assert_called_once_with(jit_func, *compile_args)
        compiled.assert_called_once_with(*dispatch_args)

    def test_compiled_cache_serializes_same_param(self):
        from torch._inductor.runtime.flydsl_cache import run_cached_flydsl

        jit_func = SimpleNamespace()
        compile_started = threading.Event()
        allow_compile = threading.Event()
        compiled = mock.Mock()
        compile_calls = 0

        class Param:
            def __cache_signature__(self):
                return ("param",)

        def compiler(*args):
            nonlocal compile_calls
            compile_calls += 1
            compile_started.set()
            self.assertTrue(allow_compile.wait(5))
            return compiled

        def invoke(value):
            return run_cached_flydsl(
                jit_func,
                object(),
                constexpr_param=Param(),
                compiler=compiler,
                dispatch_args=(value,),
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(invoke, "first")
            self.assertTrue(compile_started.wait(5))
            second = pool.submit(invoke, "second")
            allow_compile.set()
            self.assertIs(first.result(), compiled)
            self.assertIs(second.result(), compiled)

        self.assertEqual(compile_calls, 1)
        compiled.assert_called_once_with("second")

    def test_compiled_cache_retries_after_failure(self):
        from torch._inductor.runtime.flydsl_cache import run_cached_flydsl

        jit_func = SimpleNamespace()
        compiled = mock.Mock()
        compiler = mock.Mock(side_effect=[RuntimeError("compile failed"), compiled])

        class Param:
            def __cache_signature__(self):
                return ("param",)

        kwargs = {
            "constexpr_param": Param(),
            "compiler": compiler,
            "dispatch_args": (),
        }

        with self.assertRaisesRegex(RuntimeError, "compile failed"):
            run_cached_flydsl(jit_func, **kwargs)
        result = run_cached_flydsl(jit_func, **kwargs)

        self.assertIs(result, compiled)
        self.assertEqual(compiler.call_count, 2)
        compiled.assert_not_called()

    def test_gen_imports(self):
        if not HAS_FLYDSL:
            self.skipTest("requires flydsl")

        kernel = FlyDSLTemplateKernel(
            kernel_name="test_kernel",
            input_nodes=[],
            output_node=None,
        )

        imports = kernel.gen_imports()

        self.assertIn("import torch", imports)
        self.assertIn("import flydsl.compiler as flyc", imports)
        self.assertIn("import flydsl.expr as fx", imports)

    @unittest.skipUnless(HAS_FLYDSL, "requires flydsl")
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA/ROCm not available")
    @unittest.skipIf(torch.version.hip is None, "requires ROCm")
    @torch._inductor.config.patch(
        max_autotune_gemm=True,
        max_autotune_gemm_backends="FLYDSL",
    )
    def test_flydsl_gemm_transposed_rhs_e2e(self):
        from torch._inductor.utils import run_and_get_code

        if not flydsl_utils.runtime_available():
            self.skipTest("FlyDSL runtime unavailable")

        def fn(a, b):
            return torch.mm(a, b.t())

        cases = [
            (32, 128, 128),
            (32, 256, 128),
            (48, 96, 96),
        ]
        for dtype in (torch.bfloat16,):
            for m, n, k in cases:
                with self.subTest(dtype=dtype, m=m, n=n, k=k):
                    a = torch.randn(m, k, device="cuda", dtype=dtype)
                    b = torch.randn(n, k, device="cuda", dtype=dtype)

                    compiled_fn = torch.compile(fn, backend="inductor")
                    result, (code,) = run_and_get_code(compiled_fn, a, b)

                    self.assertIn("async_compile.flydsl", code)
                    self.assertIn(".mark_layout_dynamic()", code)
                    self.assertNotIn(".run(", code)
                    self.assertIn("TILE_M: fx.Constexpr", code)
                    self.assertIn("STAGES: fx.Constexpr", code)
                    self.assertIn("BLOCK_N_WARPS: fx.Constexpr", code)
                    self.assertIn("BLOCK_K_WARPS: fx.Constexpr", code)
                    self.assertTrue(
                        torch.allclose(result, fn(a, b), atol=3e-2, rtol=3e-2)
                    )

    @unittest.skipUnless(HAS_FLYDSL, "requires flydsl")
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA/ROCm not available")
    @unittest.skipIf(torch.version.hip is None, "requires ROCm")
    @torch._inductor.config.patch(
        max_autotune_gemm=True,
        max_autotune_gemm_backends="FLYDSL",
        flydsl_enable_autotuning=False,
    )
    def test_flydsl_gemm_strided_inputs(self):
        from torch._inductor.utils import run_and_get_code

        if not flydsl_utils.runtime_available():
            self.skipTest("FlyDSL runtime unavailable")

        def fn(a, b):
            return torch.mm(a, b.t())

        m, n, k = 64, 64, 128
        a_row_stride = 160
        b_row_stride = 192
        dtype = torch.bfloat16

        a_padded = torch.randn(
            m, a_row_stride, device="cuda", dtype=dtype
        )[:, :k]
        b_padded = torch.randn(
            n, b_row_stride, device="cuda", dtype=dtype
        )[:, :k]

        storage_offset = 8
        a_storage = torch.randn(m * k + storage_offset, device="cuda", dtype=dtype)
        b_storage = torch.randn(n * k + storage_offset, device="cuda", dtype=dtype)
        a_offset = torch.as_strided(
            a_storage, (m, k), (k, 1), storage_offset=storage_offset
        )
        b_offset = torch.as_strided(
            b_storage, (n, k), (k, 1), storage_offset=storage_offset
        )

        for name, a, b in (
            ("padded_rows", a_padded, b_padded),
            ("nonzero_offset", a_offset, b_offset),
        ):
            with self.subTest(name=name):
                torch._dynamo.reset()
                compiled_fn = torch.compile(fn, backend="inductor")
                result, (code,) = run_and_get_code(compiled_fn, a, b)

                self.assertIn("async_compile.flydsl", code)
                self.assertTrue(
                    torch.allclose(result, fn(a, b), atol=3e-2, rtol=3e-2)
                )

    @unittest.skipUnless(HAS_FLYDSL, "requires flydsl")
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA/ROCm not available")
    @unittest.skipIf(torch.version.hip is None, "requires ROCm")
    @torch._inductor.config.patch(
        max_autotune_gemm=True,
        max_autotune_gemm_backends="FLYDSL",
        flydsl_enable_autotuning=False,
    )
    def test_flydsl_gemm_internal_slices_preserve_offset(self):
        from torch._inductor.utils import run_and_get_code

        if not flydsl_utils.runtime_available():
            self.skipTest("FlyDSL runtime unavailable")

        m, n, k = 64, 64, 128
        column_offset = 8

        def fn(a_storage, b_storage):
            a = a_storage[1:, column_offset : column_offset + k]
            b = b_storage[1:, column_offset : column_offset + k]
            return torch.mm(a, b.t())

        a_storage = torch.randn(
            m + 1, 160, device="cuda", dtype=torch.bfloat16
        )
        b_storage = torch.randn(
            n + 1, 192, device="cuda", dtype=torch.bfloat16
        )

        compiled_fn = torch.compile(fn, backend="inductor")
        result, (code,) = run_and_get_code(compiled_fn, a_storage, b_storage)

        self.assertIn("async_compile.flydsl", code)
        self.assertTrue(
            torch.allclose(
                result,
                fn(a_storage, b_storage),
                atol=3e-2,
                rtol=3e-2,
            )
        )

    @unittest.skipUnless(HAS_FLYDSL, "requires flydsl")
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA/ROCm not available")
    @unittest.skipIf(torch.version.hip is None, "requires ROCm")
    @torch._inductor.config.patch(
        max_autotune_gemm=True,
        max_autotune_gemm_backends="FLYDSL",
        max_autotune_gemm_search_space="EXHAUSTIVE",
        flydsl_enable_autotuning=True,
    )
    def test_flydsl_autotune_transposed_rhs_uses_view_tensor(self):
        from torch._inductor.template_heuristics import flydsl as flydsl_heuristics
        from torch._inductor.utils import run_and_get_code

        if not flydsl_utils.runtime_available():
            self.skipTest("FlyDSL runtime unavailable")

        def fn(a, b):
            return torch.mm(a, b.t())

        configs = [
            asdict(config)
            for config in flydsl_heuristics.get_exhaustive_gemm_configs()[:2]
        ]
        a = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)

        with mock.patch.object(
            flydsl_heuristics, "get_gemm_configs", return_value=configs
        ):
            compiled_fn = torch.compile(fn, backend="inductor")
            result, (code,) = run_and_get_code(compiled_fn, a, b)

        self.assertIn("async_compile.flydsl", code)
        self.assertTrue(torch.allclose(result, fn(a, b), atol=3e-2, rtol=3e-2))

    @unittest.skipUnless(HAS_FLYDSL, "requires flydsl")
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA/ROCm not available")
    @unittest.skipIf(torch.version.hip is None, "requires ROCm")
    @torch._inductor.config.patch(
        max_autotune_gemm=True,
        max_autotune_gemm_backends="FLYDSL",
        max_autotune_gemm_search_space="DEFAULT",
        flydsl_enable_autotuning=True,
    )
    def test_flydsl_autotune_filters_invalid_small_k_configs(self):
        from torch._inductor.template_heuristics import flydsl as flydsl_heuristics
        from torch._inductor.utils import run_and_get_code

        if not flydsl_utils.runtime_available():
            self.skipTest("FlyDSL runtime unavailable")

        configs = [
            asdict(config) for config in flydsl_heuristics.get_default_gemm_configs()
        ]
        # Keep the regression focused on one valid choice and one HTI choice
        # whose K tile count is less than two for both tested shapes.
        full_tile_config = next(
            config
            for config in configs
            if not config["USE_HALF_TILE_INTERLEAVED"]
        )
        half_tile_config = next(
            config for config in configs if config["USE_HALF_TILE_INTERLEAVED"]
        )

        def fn(a, b):
            return torch.mm(a, b.t())

        with mock.patch.object(
            flydsl_heuristics,
            "get_gemm_configs",
            return_value=[full_tile_config, half_tile_config],
        ):
            for k in (32, 64):
                with self.subTest(k=k):
                    torch._dynamo.reset()
                    a = torch.randn(64, k, device="cuda", dtype=torch.bfloat16)
                    b = torch.randn(64, k, device="cuda", dtype=torch.bfloat16)
                    compiled_fn = torch.compile(fn, backend="inductor")
                    result, (code,) = run_and_get_code(compiled_fn, a, b)

                    self.assertIn("async_compile.flydsl", code)
                    self.assertTrue(
                        torch.allclose(result, fn(a, b), atol=3e-2, rtol=3e-2)
                    )


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
