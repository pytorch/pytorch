# Owner(s): ["module: inductor"]
import os
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn.functional as F
from torch._inductor.codegen.flydsl import flydsl_utils
from torch._inductor.codegen.flydsl.flydsl_kernel import FlyDSLTemplateKernel
from torch._inductor.codegen.flydsl.flydsl_scheduling import (
    _get_flydsl_device_arch,
    FlyDSLScheduling,
)
from torch._inductor.codegen.flydsl.flydsl_template import FlyDSLTemplate
from torch._inductor.runtime.flydsl_cache import run_cached_flydsl
from torch._inductor.select_algorithm import PartialRender
from torch._inductor.test_case import TestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


class _CacheParam:
    def __init__(self, key="param"):
        self.key = key

    def __cache_signature__(self):
        return (self.key,)


class TestFlyDSLTemplate(TestCase):
    def setUp(self):
        super().setUp()
        _get_flydsl_device_arch.cache_clear()

    def test_gen_imports(self):
        kernel = FlyDSLTemplateKernel(
            kernel_name="test_kernel",
            input_nodes=[],
            output_node=None,
        )

        imports = kernel.gen_imports()

        self.assertIn("import torch", imports)
        self.assertIn("import flydsl.compiler as flyc", imports)
        self.assertIn("import flydsl.expr as fx", imports)
        self.assertIsInstance(imports, str)

    def test_gen_defines(self):
        kernel = FlyDSLTemplateKernel(
            kernel_name="test_kernel",
            input_nodes=[],
            output_node=None,
        )

        defines = kernel.gen_defines(
            TILE_M=128,
            ENABLE_FEATURE=True,
            SCALE=1.5,
        )

        self.assertEqual(
            defines,
            (
                "TILE_M: fx.Constexpr = 128\n"
                "ENABLE_FEATURE: fx.Constexpr = True\n"
                "SCALE: fx.Constexpr = 1.5\n"
            ),
        )

    def test_render_includes_imports(self):
        template = mock.Mock()
        template.render.return_value = (
            "@flyc.kernel\ndef test_kernel_kernel():\n    pass\n"
        )
        kernel = FlyDSLTemplateKernel(
            kernel_name="test_kernel",
            input_nodes=[],
            output_node=None,
        )

        result = kernel.render(template, TILE_M=128)

        self.assertIsInstance(result, PartialRender)
        self.assertTrue(result._code.lstrip().startswith("import torch"))
        self.assertIn("import flydsl.compiler as flyc", result._code)
        self.assertIn("@flyc.kernel", result._code)

    def test_template_env_contains_hooks(self):
        captured_env = {}

        def render(**kwargs):
            captured_env.update(kwargs)
            return "rendered"

        template = mock.Mock()
        template.render = render
        kernel = FlyDSLTemplateKernel(
            kernel_name="test_kernel",
            input_nodes=[],
            output_node=None,
        )

        kernel.render(template, BLOCK_SIZE=256)

        self.assertEqual(captured_env["kernel_name"], "test_kernel")
        self.assertEqual(captured_env["BLOCK_SIZE"], 256)
        self.assertTrue(callable(captured_env["def_kernel"]))
        self.assertTrue(callable(captured_env["gen_defines"]))
        self.assertTrue(callable(captured_env["get_output"]))

    def test_duplicate_template_name_is_rejected(self):
        template_name = f"flydsl_unique_test_{id(self)}"
        FlyDSLTemplate.all_templates.pop(template_name, None)

        try:
            with mock.patch.object(
                FlyDSLTemplate,
                "_template_from_string",
                return_value=mock.Mock(),
            ):
                FlyDSLTemplate(name=template_name, source="template1")
                FlyDSLTemplate(name=template_name, source="template1")
                with self.assertRaisesRegex(
                    AssertionError, f"duplicate template name, {template_name}"
                ):
                    FlyDSLTemplate(name=template_name, source="template2")
        finally:
            FlyDSLTemplate.all_templates.pop(template_name, None)

    def test_scheduling_disables_fusion(self):
        scheduling = FlyDSLScheduling(scheduler=None)
        node1 = mock.Mock()
        node2 = mock.Mock()

        self.assertFalse(scheduling.can_fuse_vertical(node1, node2))
        self.assertFalse(scheduling.can_fuse_horizontal(node1, node2))
        self.assertEqual(scheduling.get_backend_features(device=None), set())

    def test_scheduling_caches_device_arch(self):
        props = mock.Mock(gcnArchName="gfx950:sramecc+:xnack-")
        with (
            mock.patch.dict(
                os.environ,
                {"FLYDSL_GPU_ARCH": "", "HSA_OVERRIDE_GFX_VERSION": ""},
            ),
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.get_device_properties", return_value=props) as get,
        ):
            self.assertEqual(FlyDSLScheduling._build_flydsl_gpu_arch(0), "gfx950")
            self.assertEqual(FlyDSLScheduling._build_flydsl_gpu_arch(0), "gfx950")
            get.assert_called_once_with(0)

    def test_scheduling_uses_explicit_gpu_arch(self):
        with mock.patch.dict(
            os.environ,
            {
                "FLYDSL_GPU_ARCH": "gfx950:sramecc+:xnack-",
                "ARCH": "gfx942",
            },
        ):
            self.assertEqual(
                FlyDSLScheduling._build_flydsl_gpu_arch(device_index=0),
                "gfx950",
            )

    def test_scheduling_converts_hsa_override(self):
        with mock.patch.dict(
            os.environ,
            {
                "FLYDSL_GPU_ARCH": "",
                "ARCH": "",
                "HSA_OVERRIDE_GFX_VERSION": "9.0.10",
            },
        ):
            self.assertEqual(
                FlyDSLScheduling._build_flydsl_gpu_arch(device_index=0),
                "gfx90a",
            )

    def test_scheduling_ignores_generic_arch(self):
        with (
            mock.patch.dict(
                os.environ,
                {
                    "FLYDSL_GPU_ARCH": "",
                    "HSA_OVERRIDE_GFX_VERSION": "",
                    "ARCH": "x86_64",
                },
            ),
            mock.patch("torch.cuda.is_available", return_value=False),
        ):
            self.assertIsNone(FlyDSLScheduling._build_flydsl_gpu_arch(device_index=0))

    def test_compiled_cache_keys_only_on_param(self):
        jit_func = SimpleNamespace()
        compile_args = (object(),)
        dispatch_args = (object(),)
        compiled = mock.Mock()
        compiler = mock.Mock(return_value=compiled)

        first = run_cached_flydsl(
            jit_func,
            *compile_args,
            constexpr_param=_CacheParam(),
            compiler=compiler,
            dispatch_args=dispatch_args,
        )
        second = run_cached_flydsl(
            jit_func,
            object(),
            constexpr_param=_CacheParam(),
            compiler=compiler,
            dispatch_args=dispatch_args,
        )

        self.assertIs(first, compiled)
        self.assertIs(second, compiled)
        compiler.assert_called_once_with(jit_func, *compile_args)
        compiled.assert_called_once_with(*dispatch_args)

    def test_compiled_cache_serializes_same_param(self):
        jit_func = SimpleNamespace()
        compile_started = threading.Event()
        allow_compile = threading.Event()
        compiled = mock.Mock()
        compile_calls = 0

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
                constexpr_param=_CacheParam(),
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

    def test_compiled_cache_hit_skips_lock(self):
        compiled = mock.Mock()
        jit_func = SimpleNamespace(_compiled_cache={("param",): compiled})
        with mock.patch(
            "torch._inductor.runtime.flydsl_cache._compiled_cache_lock"
        ) as cache_lock:
            result = run_cached_flydsl(
                jit_func,
                constexpr_param=_CacheParam(),
                compiler=mock.Mock(),
                dispatch_args=("dispatch",),
            )

        self.assertIs(result, compiled)
        cache_lock.__enter__.assert_not_called()
        compiled.assert_called_once_with("dispatch")

    def test_compiled_cache_retries_after_failure(self):
        jit_func = SimpleNamespace()
        compiled = mock.Mock()
        compiler = mock.Mock(side_effect=[RuntimeError("compile failed"), compiled])
        kwargs = {
            "constexpr_param": _CacheParam(),
            "compiler": compiler,
            "dispatch_args": (),
        }

        with self.assertRaisesRegex(RuntimeError, "compile failed"):
            run_cached_flydsl(jit_func, **kwargs)
        result = run_cached_flydsl(jit_func, **kwargs)

        self.assertIs(result, compiled)
        self.assertEqual(compiler.call_count, 2)
        compiled.assert_not_called()

    def _assert_compiled_mm(self, a, b, *, expect_flydsl=True):
        from torch._inductor.utils import run_and_get_code

        def fn(lhs, rhs):
            return torch.mm(lhs, rhs.t())

        torch._dynamo.reset()
        result, (code,) = run_and_get_code(torch.compile(fn, backend="inductor"), a, b)
        assertion = self.assertIn if expect_flydsl else self.assertNotIn
        assertion("async_compile.flydsl", code)
        self.assertEqual(result, fn(a, b), atol=3e-2, rtol=3e-2)
        return code

    def _assert_compiled_grouped_mm(self, a, b, offs, *, expect_flydsl=True):
        from torch._inductor.utils import run_and_get_code

        def fn(a, b, offs):
            return F.grouped_mm(a, b, offs=offs)

        torch._dynamo.reset()
        result, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor"), a, b, offs
        )
        assertion = self.assertIn if expect_flydsl else self.assertNotIn
        assertion("async_compile.flydsl", code)
        self.assertEqual(result, fn(a, b, offs), atol=3e-2, rtol=3e-2)
        return code

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA/ROCm not available")
    @unittest.skipIf(torch.version.hip is None, "requires ROCm")
    @torch._inductor.config.patch(
        max_autotune_gemm=True,
        max_autotune_gemm_backends="FLYDSL",
    )
    def test_flydsl_gemm_transposed_rhs_e2e(self):
        if not flydsl_utils.runtime_available():
            self.skipTest("FlyDSL runtime unavailable")

        cases = (
            (torch.bfloat16, 32, 128, 128),
            (torch.float16, 32, 128, 128),
            (torch.bfloat16, 32, 256, 128),
            (torch.bfloat16, 48, 96, 96),
        )
        for dtype, m, n, k in cases:
            with self.subTest(dtype=dtype, m=m, n=n, k=k):
                a = torch.randn(m, k, device="cuda", dtype=dtype)
                b = torch.randn(n, k, device="cuda", dtype=dtype)
                code = self._assert_compiled_mm(a, b)
                self.assertIn(".mark_layout_dynamic()", code)
                self.assertIn(".run(", code)
                self.assertIn("TILE_M: fx.Constexpr", code)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA/ROCm not available")
    @unittest.skipIf(torch.version.hip is None, "requires ROCm")
    @torch._inductor.config.patch(
        max_autotune_gemm=True,
        max_autotune_gemm_backends="FLYDSL",
        flydsl_enable_autotuning=False,
    )
    def test_flydsl_gemm_strides_offsets_and_alignment(self):
        if not flydsl_utils.runtime_available():
            self.skipTest("FlyDSL runtime unavailable")

        m = n = 64
        k = 128
        dtype = torch.bfloat16
        a = torch.randn(m, k, device="cuda", dtype=dtype)
        b = torch.randn(n, k, device="cuda", dtype=dtype)
        a_storage = torch.randn(m + 1, 160, device="cuda", dtype=dtype)
        b_storage = torch.randn(n + 1, 192, device="cuda", dtype=dtype)
        supported = (
            a_storage[1:, 8 : 8 + k],
            b_storage[1:, 8 : 8 + k],
        )
        bad_stride = torch.empty_strided(
            (m, k), (k + 1, 1), device="cuda", dtype=dtype
        ).normal_()
        bad_offset = torch.as_strided(
            torch.randn(n * k + 1, device="cuda", dtype=dtype),
            (n, k),
            (k, 1),
            storage_offset=1,
        )

        self._assert_compiled_mm(*supported)
        with torch._inductor.config.patch(max_autotune_gemm_backends="ATEN,FLYDSL"):
            self._assert_compiled_mm(bad_stride, b, expect_flydsl=False)
            self._assert_compiled_mm(a, bad_offset, expect_flydsl=False)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA/ROCm not available")
    @unittest.skipIf(torch.version.hip is None, "requires ROCm")
    @torch._inductor.config.patch(
        max_autotune_gemm=True,
        max_autotune_gemm_backends="FLYDSL",
        max_autotune_gemm_search_space="EXHAUSTIVE",
        flydsl_enable_autotuning=True,
        autotune_in_subproc=True,
    )
    def test_flydsl_autotune_transposed_rhs_uses_view_tensor(self):
        from torch._inductor.template_heuristics import flydsl as flydsl_heuristics

        if not flydsl_utils.runtime_available():
            self.skipTest("FlyDSL runtime unavailable")

        configs = [
            asdict(config) for config in flydsl_heuristics.get_default_gemm_configs()
        ]
        configs = [
            next(
                config for config in configs if not config["USE_HALF_TILE_INTERLEAVED"]
            ),
            next(config for config in configs if config["USE_HALF_TILE_INTERLEAVED"]),
        ]

        with mock.patch.object(
            flydsl_heuristics, "get_gemm_configs", return_value=configs
        ):
            for k in (32, 64, 128):
                with self.subTest(k=k):
                    a = torch.randn(64, k, device="cuda", dtype=torch.bfloat16)
                    b = torch.randn(64, k, device="cuda", dtype=torch.bfloat16)
                    self._assert_compiled_mm(a, b)

    @parametrize(
        "dtype,k,n",
        [
            (torch.bfloat16, 128, 128),
            (torch.bfloat16, 160, 256),
        ],
    )
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA/ROCm not available")
    @unittest.skipIf(torch.version.hip is None, "requires ROCm")
    @torch._inductor.config.patch(
        max_autotune_gemm=True,
        max_autotune_gemm_backends="FLYDSL",
        flydsl_enable_autotuning=False,
    )
    def test_flydsl_grouped_mm_e2e(self, dtype, k, n):
        if not flydsl_utils.runtime_available():
            self.skipTest("FlyDSL runtime unavailable")

        group_sizes = torch.tensor(
            [0, 1, 67, 0, 130, 3], device="cuda", dtype=torch.int32
        )
        offs = group_sizes.cumsum(0).to(torch.int32)
        a = torch.randn(int(group_sizes.sum()), k, device="cuda", dtype=dtype)
        b = torch.randn(group_sizes.numel(), k, n, device="cuda", dtype=dtype)
        code = self._assert_compiled_grouped_mm(a, b, offs)
        self.assertIn(".mark_layout_dynamic()", code)
        self.assertIn("COMPILE_ONLY", code)
        self.assertIn("_precompile", code)

    @parametrize("k", (96, 192))
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA/ROCm not available")
    @unittest.skipIf(torch.version.hip is None, "requires ROCm")
    @torch._inductor.config.patch(
        max_autotune_gemm=True,
        max_autotune_gemm_backends="FLYDSL",
        max_autotune_gemm_search_space="EXHAUSTIVE",
        flydsl_enable_autotuning=True,
        autotune_in_subproc=True,
    )
    def test_flydsl_grouped_mm_autotune_e2e(self, k):
        from torch._inductor.template_heuristics import flydsl as flydsl_heuristics

        if not flydsl_utils.runtime_available():
            self.skipTest("FlyDSL runtime unavailable")

        configs = [
            asdict(config)
            for config in flydsl_heuristics.get_default_grouped_gemm_configs()
        ]
        configs = [
            next(config for config in configs if not config["USE_HALF_TILE_INTERLEAVED"]),
            next(config for config in configs if config["USE_HALF_TILE_INTERLEAVED"]),
        ]
        group_sizes = torch.tensor(
            [0, 1, 67, 0, 130, 3], device="cuda", dtype=torch.int32
        )
        offs = group_sizes.cumsum(0).to(torch.int32)
        a = torch.randn(int(group_sizes.sum()), k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(group_sizes.numel(), k, 128, device="cuda", dtype=torch.bfloat16)
        with mock.patch.object(
            flydsl_heuristics,
            "get_grouped_gemm_configs",
            return_value=configs,
        ):
            self._assert_compiled_grouped_mm(a, b, offs)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA/ROCm not available")
    @unittest.skipIf(torch.version.hip is None, "requires ROCm")
    @torch._inductor.config.patch(
        max_autotune_gemm=True,
        max_autotune_gemm_backends="FLYDSL",
        flydsl_enable_autotuning=True,
        autotune_in_subproc=False,
    )
    def test_flydsl_grouped_mm_kernel_paths_e2e(self):
        from torch._inductor.template_heuristics import flydsl as flydsl_heuristics

        if not flydsl_utils.runtime_available():
            self.skipTest("FlyDSL runtime unavailable")

        config_fields = (
            "TILE_M",
            "TILE_N",
            "TILE_K",
            "STAGES",
            "BLOCK_M_WARPS",
            "BLOCK_N_WARPS",
            "GROUP_M",
            "USE_HALF_TILE_INTERLEAVED",
        )
        cases = (
            ("block_swizzle", 4096, 1024, 128, (128, 128, 64, 2, 1, 4, 4, False)),
            ("deep_pipeline", 128, 128, 192, (64, 128, 64, 3, 1, 4, 0, False)),
            ("large_tile", 1024, 256, 128, (256, 256, 64, 2, 2, 4, 0, True)),
        )
        configs = [
            asdict(config)
            for config in flydsl_heuristics.get_default_grouped_gemm_configs()
        ]

        for name, m, n, k, expected_values in cases:
            with self.subTest(name=name):
                config = next(
                    config
                    for config in configs
                    if tuple(config[field] for field in config_fields)
                    == expected_values
                )
                group_sizes = torch.tensor([m], device="cuda", dtype=torch.int32)
                offs = group_sizes.cumsum(0).to(torch.int32)
                a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
                b = torch.randn(1, k, n, device="cuda", dtype=torch.bfloat16)
                with mock.patch.object(
                    flydsl_heuristics,
                    "get_grouped_gemm_configs",
                    return_value=[config],
                ):
                    code = self._assert_compiled_grouped_mm(a, b, offs)
                self.assertIn("launch_gemm_gfx950_grouped", code)
                for field, value in zip(config_fields, expected_values):
                    self.assertIn(f"{field}: fx.Constexpr = {value}", code)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA/ROCm not available")
    @unittest.skipIf(torch.version.hip is None, "requires ROCm")
    @torch._inductor.config.patch(
        max_autotune_gemm=True,
        max_autotune_gemm_backends="ATEN,FLYDSL",
    )
    def test_flydsl_grouped_mm_fallback_e2e(self):
        if not flydsl_utils.runtime_available():
            self.skipTest("FlyDSL runtime unavailable")

        group_sizes = torch.tensor([32, 64], device="cuda", dtype=torch.int32)
        offs = group_sizes.cumsum(0).to(torch.int32)
        k = 128
        a = torch.randn(int(group_sizes.sum()), k, device="cuda", dtype=torch.bfloat16)
        cases = (
            (
                "b_transposed",
                torch.randn(2, 256, k, device="cuda", dtype=torch.bfloat16).transpose(
                    -1, -2
                ),
            ),
            (
                "n_not_tile_divisible",
                torch.randn(2, k, 96, device="cuda", dtype=torch.bfloat16),
            ),
        )

        for name, b in cases:
            with self.subTest(name=name):
                self._assert_compiled_grouped_mm(a, b, offs, expect_flydsl=False)

instantiate_parametrized_tests(TestFlyDSLTemplate)

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
