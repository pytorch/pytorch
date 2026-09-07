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
from torch._inductor.ir import Buffer, FixedLayout
from torch._inductor.runtime.flydsl_cache import run_cached_flydsl
from torch._inductor.select_algorithm import PartialRender
from torch._inductor.test_case import TestCase
from torch._inductor.utils import OrderedSet
from torch._inductor.virtualized import V
from torch.nn.functional import ScalingType, SwizzleType
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


class _CacheParam:
    def __init__(self, key="param"):
        self.key = key

    def __cache_signature__(self):
        return (self.key,)


@instantiate_parametrized_tests
class TestFlyDSLTemplate(TestCase):
    def setUp(self):
        super().setUp()
        flydsl_utils._check_runtime_available.cache_clear()
        _get_flydsl_device_arch.cache_clear()

    def test_runtime_unavailable_when_package_missing(self):
        with mock.patch.object(flydsl_utils, "find_spec", return_value=None):
            reason = flydsl_utils._flydsl_runtime_unavailable_reason()
        self.assertIn("missing optional dependency", reason)

    def test_runtime_unavailable_when_mlir_missing(self):
        package_spec = SimpleNamespace(submodule_search_locations=["package"])
        with (
            mock.patch.object(flydsl_utils, "find_spec", return_value=package_spec),
            mock.patch.object(flydsl_utils, "_pathfinder_find_spec", return_value=None),
        ):
            reason = flydsl_utils._flydsl_runtime_unavailable_reason()
        self.assertIn("flydsl._mlir", reason)

    def test_runtime_available_for_supported_version(self):
        package_spec = SimpleNamespace(submodule_search_locations=["package"])
        with (
            mock.patch.object(flydsl_utils, "find_spec", return_value=package_spec),
            mock.patch.object(
                flydsl_utils, "_pathfinder_find_spec", return_value=SimpleNamespace()
            ),
            mock.patch.object(
                flydsl_utils,
                "_available_version",
                return_value=SimpleNamespace(release=(0, 3, 0)),
            ),
        ):
            reason = flydsl_utils._flydsl_runtime_unavailable_reason()
        self.assertIsNone(reason)

    def test_unavailable_runtime_declines_choice(self):
        template_name = f"flydsl_unavailable_test_{id(self)}"
        self.addCleanup(FlyDSLTemplate.all_templates.pop, template_name, None)
        with (
            mock.patch.object(
                FlyDSLTemplate, "_template_from_string", return_value=mock.Mock()
            ),
            mock.patch.object(flydsl_utils, "runtime_available", return_value=False),
        ):
            template = FlyDSLTemplate(name=template_name, source="template")
            choices = []
            result = template.maybe_append_choice(choices)

        self.assertIsInstance(result, NotImplementedError)
        self.assertEqual(choices, [])

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
        code = result.finalize_all()

        self.assertIsInstance(result, PartialRender)
        self.assertTrue(code.lstrip().startswith("import torch"))
        self.assertIn("import flydsl.compiler as flyc", code)
        self.assertIn("@flyc.kernel", code)

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

    def test_scheduling_codegen_template_calls_kernel_wrapper(self):
        layout = FixedLayout(torch.device("cpu"), torch.float32, [1], [1])
        input_node = Buffer(name="input", layout=layout)
        output_node = Buffer(name="output", layout=layout)
        kernel = FlyDSLTemplateKernel(
            kernel_name="test_kernel",
            input_nodes=[input_node],
            output_node=output_node,
        )

        ftb = mock.Mock()
        ftb.make_kernel_render.return_value = (kernel, lambda: "source")
        template_node = mock.Mock(node=ftb)
        wrapper = mock.Mock()
        graph = SimpleNamespace(
            get_dtype=lambda _name: torch.float32,
            removed_buffers=OrderedSet(),
            scheduler=None,
            wrapper_code=wrapper,
        )
        scheduling = FlyDSLScheduling(scheduler=mock.Mock())

        with (
            V.set_graph_handler(graph),
            mock.patch.object(scheduling, "is_flydsl_template", return_value=True),
            mock.patch.object(
                scheduling, "_build_precompile_metadata", return_value=None
            ),
            mock.patch.object(
                scheduling, "define_kernel", return_value="generated_kernel"
            ),
            mock.patch.object(scheduling, "codegen_comment"),
            mock.patch.object(scheduling, "free_buffers_in_scheduler"),
        ):
            kernel.def_kernel("input")
            scheduling.codegen_template(template_node, [], [])

        wrapper.generate_kernel_call.assert_called_once()
        self.assertTrue(wrapper.generate_kernel_call.call_args.kwargs["triton"])
        template_node.mark_run.assert_called_once()

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

    @parametrize(
        "env,cuda_available,gcn_arch,expected",
        (
            (
                {
                    "FLYDSL_GPU_ARCH": "gfx950:sramecc+:xnack-",
                    "HSA_OVERRIDE_GFX_VERSION": "",
                },
                False,
                None,
                "gfx950",
            ),
            (
                {"FLYDSL_GPU_ARCH": "", "HSA_OVERRIDE_GFX_VERSION": "9.0.10"},
                False,
                None,
                "gfx90a",
            ),
            (
                {"FLYDSL_GPU_ARCH": "", "HSA_OVERRIDE_GFX_VERSION": ""},
                False,
                None,
                None,
            ),
            (
                {"FLYDSL_GPU_ARCH": "", "HSA_OVERRIDE_GFX_VERSION": "9.0.x"},
                True,
                "gfx942:xnack-",
                "gfx942",
            ),
        ),
    )
    def test_scheduling_resolves_gpu_arch(
        self, env, cuda_available, gcn_arch, expected
    ):
        props = mock.Mock(gcnArchName=gcn_arch)
        with (
            mock.patch.dict(os.environ, env),
            mock.patch("torch.cuda.is_available", return_value=cuda_available),
            mock.patch(
                "torch.cuda.get_device_properties", return_value=props
            ) as get_properties,
        ):
            self.assertEqual(
                FlyDSLScheduling._build_flydsl_gpu_arch(device_index=0),
                expected,
            )
        if cuda_available:
            get_properties.assert_called_once_with(0)
        else:
            get_properties.assert_not_called()

    def test_precompile_metadata_requires_defined_signature(self):
        scheduling = FlyDSLScheduling(scheduler=None)
        kernel = SimpleNamespace(
            _template_signature_defined=False,
            _template_input_args=[],
        )
        layout = SimpleNamespace(
            size=[1],
            stride=[1],
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        self.assertIsNone(
            scheduling._build_precompile_metadata(
                kernel, SimpleNamespace(layout=layout)
            )
        )

    def test_precompile_metadata_supports_inputless_template(self):
        scheduling = FlyDSLScheduling(scheduler=None)
        layout = FixedLayout(torch.device("cpu"), torch.float32, [1], [1])
        kernel = FlyDSLTemplateKernel(
            kernel_name="inputless",
            input_nodes=[],
            output_node=Buffer(name="output", layout=layout),
        )
        graph = SimpleNamespace(
            removed_buffers=OrderedSet(),
            scheduler=None,
        )

        with (
            V.set_graph_handler(graph),
            mock.patch("torch.cuda.is_available", return_value=False),
        ):
            kernel.def_kernel()
            metadata = scheduling._build_precompile_metadata(
                kernel, SimpleNamespace(layout=layout)
            )

        self.assertIsNotNone(metadata)
        self.assertEqual(metadata["precompile_shapes"], {"output": [1]})
        self.assertEqual(metadata["precompile_strides"], {"output": [1]})
        self.assertEqual(metadata["precompile_dtypes"], {"output": "float32"})

    @parametrize(
        "size,dtype,stride,offset,n",
        (
            ([1, 64, 128], torch.float16, [8192, 128, 1], 0, 64),
            ([64, 128], torch.float32, [128, 1], 0, 64),
            ([64, 128], torch.float16, [129, 1], 0, 64),
            ([64, 128], torch.float16, [128, 1], 1, 64),
            ([64, 128], torch.float16, [128, 1], 0, 63),
        ),
    )
    def test_mm_gate_rejects_invalid_inputs(self, size, dtype, stride, offset, n):
        from torch._inductor.kernel import mm

        def node(size, stride, offset=0):
            return SimpleNamespace(
                get_size=lambda: size,
                get_stride=lambda: stride,
                get_dtype=lambda: dtype,
                get_layout=lambda: SimpleNamespace(offset=offset),
            )

        mat1 = node(size, stride, offset)
        mat2 = node([128, n], [1, 128])
        layout = SimpleNamespace(stride=[n, 1], dtype=dtype, device=torch.device("cpu"))
        sizevars = SimpleNamespace(
            statically_known_equals=lambda x, y: x == y,
            statically_known_multiple_of=lambda x, y: x % y == 0,
        )
        with (
            V.set_graph_handler(SimpleNamespace(sizevars=sizevars)),
            mock.patch.object(mm, "use_flydsl_gemm_template", return_value=True),
            mock.patch.object(mm, "is_unaligned", return_value=False),
        ):
            result = mm.get_flydsl_mm_template_kwargs(layout, mat1, mat2, True, True)
            self.assertEqual(result, [])

    def test_compiled_cache_keys_on_device_and_param(self):
        jit_func = SimpleNamespace()
        compiled = mock.Mock()
        compiler = mock.Mock(return_value=compiled)

        def invoke(device_index):
            dispatch = SimpleNamespace(device=SimpleNamespace(index=device_index))
            return run_cached_flydsl(
                jit_func,
                object(),
                constexpr_param=_CacheParam(),
                compiler=compiler,
                dispatch_args=(dispatch,),
            )

        first = invoke(0)
        with mock.patch(
            "torch._inductor.runtime.flydsl_cache._compiled_cache_lock"
        ) as cache_lock:
            second = invoke(0)
        third = invoke(1)

        self.assertIs(first, compiled)
        self.assertIs(second, compiled)
        self.assertIs(third, compiled)
        cache_lock.__enter__.assert_not_called()
        self.assertEqual(compiler.call_count, 2)
        compiled.assert_called_once()

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

    def _assert_compiled_mm(self, a, b, *, expect_flydsl: bool | None = True):
        from torch._inductor.utils import run_and_get_code

        def fn(lhs, rhs):
            return torch.mm(lhs, rhs.t())

        torch._dynamo.reset()
        result, (code,) = run_and_get_code(torch.compile(fn, backend="inductor"), a, b)
        if expect_flydsl is not None:
            assertion = self.assertIn if expect_flydsl else self.assertNotIn
            assertion("async_compile.flydsl", code)
        self.assertEqual(result, fn(a, b), atol=3e-2, rtol=3e-2)
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
                self.assertIn("mat2.transpose(0, 1)", code)
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

        with torch._inductor.config.patch(
            max_autotune_gemm_backends="ATEN,FLYDSL",
            autotune_in_subproc=False,
        ):
            self._assert_compiled_mm(*supported, expect_flydsl=None)
            self._assert_compiled_mm(bad_stride, b, expect_flydsl=False)
            self._assert_compiled_mm(a, bad_offset, expect_flydsl=False)
        self._assert_compiled_mm(*supported)

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
        from torch._inductor.heuristics.template import flydsl as flydsl_heuristics

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



    def test_compiled_cache_keys_on_device_and_param(self):
        jit_func = SimpleNamespace()
        compiled = mock.Mock()
        compiler = mock.Mock(return_value=compiled)

        def invoke(device_index):
            dispatch = SimpleNamespace(device=SimpleNamespace(index=device_index))
            return run_cached_flydsl(
                jit_func,
                object(),
                constexpr_param=_CacheParam(),
                compiler=compiler,
                dispatch_args=(dispatch,),
            )

        first = invoke(0)
        with mock.patch(
            "torch._inductor.runtime.flydsl_cache._compiled_cache_lock"
        ) as cache_lock:
            second = invoke(0)
        third = invoke(1)

        self.assertIs(first, compiled)
        self.assertIs(second, compiled)
        self.assertIs(third, compiled)
        cache_lock.__enter__.assert_not_called()
        self.assertEqual(compiler.call_count, 2)
        compiled.assert_called_once()

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


    # ------------------------------------------------------------------
    # MXFP8 ragged grouped GEMM (aten._scaled_grouped_mm_v2)
    # ------------------------------------------------------------------

    @staticmethod
    def _mxfp8_quantize(x, block=32):
        """Cast the last dim of `x` to MXFP8: e4m3 data + e8m0 block scales.

        Returns (fp8 data, e8m0 scales, f32 scales) -- the f32 scales are the
        exact values the e8m0 ones encode, so a reference can dequantize
        without re-deriving the exponent.
        """
        k = x.shape[-1]
        blocks = x.reshape(*x.shape[:-1], k // block, block)
        amax = blocks.abs().amax(-1)
        # e4m3 max is 448 = 2**8.8; take the exponent that maps amax below it.
        exponent = torch.floor(torch.log2(amax.clamp(min=1e-30))).to(torch.int32) - 7
        exponent = exponent.clamp(-127, 127)
        scale_f32 = torch.exp2(exponent.float())
        data = (
            (blocks / scale_f32.unsqueeze(-1))
            .clamp(-448, 448)
            .to(torch.float8_e4m3fn)
            .reshape(*x.shape[:-1], k)
        )
        scale_e8m0 = (exponent + 127).to(torch.uint8).view(torch.float8_e8m0fnu)
        return data, scale_e8m0, scale_f32

    @classmethod
    def _mxfp8_grouped_reference(cls, a, a_scale, b, b_scale, offs, block=32):
        """Dequantize and matmul per group, in f32."""
        m, k = a.shape
        g, n = b.shape[0], b.shape[1]
        a_deq = a.float().reshape(m, k // block, block) * a_scale.unsqueeze(-1)
        b_deq = b.float().reshape(g, n, k // block, block) * b_scale.unsqueeze(-1)
        a_deq = a_deq.reshape(m, k)
        b_deq = b_deq.reshape(g, n, k)
        out = torch.zeros(m, n, device=a.device, dtype=torch.float32)
        start = 0
        for group in range(g):
            end = int(offs[group])
            if end > start:
                out[start:end] = a_deq[start:end] @ b_deq[group].t()
            start = end
        return out.to(torch.bfloat16)

    @classmethod
    def _make_mxfp8_grouped_inputs(cls, group_sizes, k, n, device="cuda"):
        offs = torch.tensor(group_sizes, device=device, dtype=torch.int32).cumsum(0)
        offs = offs.to(torch.int32)
        m = int(sum(group_sizes))
        g = len(group_sizes)
        a_hp = torch.randn(m, k, device=device) * 0.5
        # The weight is generated as [G, N, K] row-major and handed to the op
        # as the [G, K, N] view of it, which is the layout every scaled GEMM
        # already requires of mat_b.
        b_hp = torch.randn(g, n, k, device=device) * 0.5
        a, a_scale, a_scale_f32 = cls._mxfp8_quantize(a_hp)
        b, b_scale, b_scale_f32 = cls._mxfp8_quantize(b_hp)
        reference = cls._mxfp8_grouped_reference(a, a_scale_f32, b, b_scale_f32, offs)
        return (
            a,
            b.transpose(-2, -1),
            a_scale,
            b_scale.reshape(g, -1),
            offs,
            reference,
        )

    @staticmethod
    def _scaled_grouped_mm_mxfp8(a, b, a_scale, b_scale, offs, **kwargs):
        return F.scaled_grouped_mm(
            a,
            b,
            a_scale,
            ScalingType.BlockWise1x32,
            b_scale,
            ScalingType.BlockWise1x32,
            swizzle_a=SwizzleType.NO_SWIZZLE,
            swizzle_b=SwizzleType.NO_SWIZZLE,
            offs=offs,
            output_dtype=torch.bfloat16,
            **kwargs,
        )

    def test_flydsl_mxfp8_grouped_gemm_config_schema(self):
        from torch._inductor.heuristics.template import flydsl as flydsl_heuristics

        # Non-autotuned selection is the tile the kernel's own heuristic picks,
        # not a fixed one -- so it has to be shape-dependent.
        with (
            mock.patch.object(
                flydsl_heuristics,
                "get_default_mxfp8_grouped_gemm_config",
                return_value=flydsl_heuristics.FlyDSLMXFP8GroupedGemmConfig(64, 128),
            ),
            mock.patch.object(flydsl_heuristics, "_make_mxfp8_grouped_gemm_param"),
            torch._inductor.config.patch(flydsl_enable_autotuning=False),
        ):
            selected = flydsl_heuristics.get_mxfp8_grouped_gemm_configs(
                512, 2048, 2048, 8
            )
        self.assertEqual(selected, [{"BLOCK_R": 64, "BLOCK_C": 128}])

        with (
            mock.patch.object(flydsl_heuristics, "_make_mxfp8_grouped_gemm_param"),
            torch._inductor.config.patch(flydsl_enable_autotuning=True),
        ):
            exhaustive = flydsl_heuristics.get_mxfp8_grouped_gemm_configs(
                512, 2048, 2048, 8
            )
        self.assertEqual(len(exhaustive), 6)
        self.assertIn({"BLOCK_R": 256, "BLOCK_C": 256}, exhaustive)
        self.assertIn({"BLOCK_R": 64, "BLOCK_C": 128}, exhaustive)

    @parametrize(
        "k,n,g,block_r,block_c,expected",
        (
            (2048, 2048, 8, 256, 256, True),
            (2048, 2048, 8, 64, 128, True),
            # K must be a whole number of 128-element pipeline steps...
            (2080, 2048, 8, 256, 256, False),
            # ...and there must be at least four of them (2 prologue, 2 tails).
            (384, 2048, 8, 256, 256, False),
            # A 128-wide tile only pays when it divides N.
            (2048, 1408, 8, 256, 128, True),
            (2048, 1344, 8, 256, 128, False),
            # Tiles the kernel does not implement.
            (2048, 2048, 8, 32, 256, False),
            (2048, 2048, 8, 256, 64, False),
        ),
    )
    def test_flydsl_mxfp8_grouped_gemm_shape_validation(
        self, k, n, g, block_r, block_c, expected
    ):
        from torch._inductor.heuristics.template import flydsl as flydsl_heuristics

        if not flydsl_utils.runtime_available():
            self.skipTest("FlyDSL runtime unavailable")

        valid = flydsl_heuristics.is_mxfp8_grouped_gemm_config_valid_for_shape(
            n, k, g, {"BLOCK_R": block_r, "BLOCK_C": block_c}
        )
        self.assertEqual(valid, expected)

    @parametrize(
        "case",
        (
            "a_dtype",
            "out_dtype",
            "scale_dtype",
            "b_not_k_major",
            "a_padded_stride",
            "scale_a_padded_stride",
            "scale_b_wrong_shape",
            "offs_dtype",
            "k_not_scale_aligned",
        ),
    )
    def test_flydsl_mxfp8_grouped_gate_rejects_invalid_inputs(self, case):
        """The gate rejects everything the kernel's layout contract excludes.

        Driven through fake IR nodes rather than a compile so it runs without a
        GPU, and -- more to the point -- without ATen, which has no MXFP8
        grouped kernel on ROCm to fall back to.
        """
        from torch._inductor.kernel import mm_grouped

        m, k, n, g = 512, 2048, 256, 4
        scale_k = k // 32

        def node(size, stride, dtype, offset=0):
            return SimpleNamespace(
                get_size=lambda: size,
                get_stride=lambda: stride,
                get_dtype=lambda: dtype,
                get_layout=lambda: SimpleNamespace(offset=offset),
            )

        fp8 = torch.float8_e4m3fn
        e8m0 = torch.float8_e8m0fnu
        mat_a = node([m, k], [k, 1], fp8)
        mat_b = node([g, k, n], [k * n, 1, k], fp8)
        scale_a = node([m, scale_k], [scale_k, 1], e8m0)
        scale_b = node([g, n * scale_k], [n * scale_k, 1], e8m0)
        offs = node([g], [1], torch.int32)
        out_dtype = torch.bfloat16

        if case == "a_dtype":
            mat_a = node([m, k], [k, 1], torch.float8_e5m2)
        elif case == "out_dtype":
            out_dtype = torch.float16
        elif case == "scale_dtype":
            scale_a = node([m, scale_k], [scale_k, 1], torch.float32)
        elif case == "b_not_k_major":
            mat_b = node([g, k, n], [k * n, n, 1], fp8)
        elif case == "a_padded_stride":
            mat_a = node([m, k], [k + 32, 1], fp8)
        elif case == "scale_a_padded_stride":
            scale_a = node([m, scale_k], [scale_k + 4, 1], e8m0)
        elif case == "scale_b_wrong_shape":
            scale_b = node([g, n, scale_k], [n * scale_k, scale_k, 1], e8m0)
        elif case == "offs_dtype":
            offs = node([g], [1], torch.int64)
        elif case == "k_not_scale_aligned":
            mat_a = node([m, k + 16], [k + 16, 1], fp8)

        layout = SimpleNamespace(
            stride=[n, 1],
            dtype=out_dtype,
            size=[m, n],
            device=torch.device("cpu"),
        )
        sizevars = SimpleNamespace(
            statically_known_equals=lambda x, y: x == y,
            statically_known_multiple_of=lambda x, y: x % y == 0,
        )
        with (
            V.set_graph_handler(SimpleNamespace(sizevars=sizevars)),
            mock.patch.object(
                mm_grouped, "use_flydsl_gemm_template", return_value=True
            ),
            mock.patch.object(mm_grouped, "is_unaligned", return_value=False),
        ):
            result = mm_grouped.get_flydsl_mxfp8_grouped_mm_template_kwargs(
                mat_a, mat_b, scale_a, scale_b, offs, layout, True
            )
        self.assertEqual(result, [])

    def _assert_compiled_mxfp8_grouped_mm(
        self, group_sizes, k, n, *, expect_flydsl: bool = True
    ):
        from torch._inductor.utils import run_and_get_code

        a, b, a_scale, b_scale, offs, reference = self._make_mxfp8_grouped_inputs(
            group_sizes, k, n
        )

        def fn(a, b, a_scale, b_scale, offs):
            return self._scaled_grouped_mm_mxfp8(a, b, a_scale, b_scale, offs)

        torch._dynamo.reset()
        result, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor"), a, b, a_scale, b_scale, offs
        )
        assertion = self.assertIn if expect_flydsl else self.assertNotIn
        assertion("async_compile.flydsl", code)
        # Rows past the last group are written by no block and are not part of
        # the result, so only the grouped region is compared.
        rows = int(offs[-1])
        self.assertEqual(
            result[:rows].float(), reference[:rows].float(), atol=6e-2, rtol=6e-2
        )
        return code

    @parametrize(
        "group_sizes,k,n",
        (
            ([512, 300, 700, 1000], 2048, 2048),
            ([512] * 8, 4096, 4096),
            # Empty groups, and a group boundary inside a row tile.
            ([0, 1024, 0, 512, 2048, 128, 0, 896], 2048, 2048),
            # The minimum K the pipeline supports: 4 steps of 128.
            ([256] * 4, 512, 2048),
            # N that only the 128-wide tile divides.
            ([1, 3, 7, 15, 31, 63, 127, 255], 2048, 1408),
        ),
    )
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA/ROCm not available")
    @unittest.skipIf(torch.version.hip is None, "requires ROCm")
    @torch._inductor.config.patch(
        max_autotune_gemm=True,
        max_autotune_gemm_backends="FLYDSL",
        flydsl_enable_autotuning=False,
    )
    def test_flydsl_mxfp8_grouped_mm_e2e(self, group_sizes, k, n):
        if not flydsl_utils.runtime_available():
            self.skipTest("FlyDSL runtime unavailable")

        code = self._assert_compiled_mxfp8_grouped_mm(group_sizes, k, n)
        self.assertIn(".mark_layout_dynamic()", code)
        self.assertIn("_precompile", code)
        # Compile-only is an argument, not a process-global env flag: warming
        # and a real dispatch can share a process.
        self.assertIn("compile_only=True", code)
        self.assertNotIn("FLYDSL_COMPILE_ONLY", code)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA/ROCm not available")
    @unittest.skipIf(torch.version.hip is None, "requires ROCm")
    @torch._inductor.config.patch(
        max_autotune_gemm=True,
        max_autotune_gemm_backends="FLYDSL",
        flydsl_enable_autotuning=True,
        autotune_in_subproc=False,
    )
    def test_flydsl_mxfp8_grouped_mm_autotune_e2e(self):
        """Autotuning walks every tile the kernel implements, and each is correct."""
        if not flydsl_utils.runtime_available():
            self.skipTest("FlyDSL runtime unavailable")

        self._assert_compiled_mxfp8_grouped_mm([512, 300, 700, 1000], 2048, 2048)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA/ROCm not available")
    @unittest.skipIf(torch.version.hip is None, "requires ROCm")
    def test_flydsl_mxfp8_grouped_mm_tile_parity_e2e(self):
        """Every tile must agree bitwise, on the shapes that once did not.

        The N_ACCUMS=4 tiles (BLOCK_R=64, and 128x128) computed a small
        fraction of elements wrong until `wait_barrier` was made to wait on
        lgkmcnt as well as vmcnt: an s2r fragment issued in one cluster and
        consumed in the next could cross the barrier still outstanding. Only
        the MFMAs in between covered that gap, so exactly the tiles with four
        accumulators broke. `pick_block_r` can return those tiles, so this
        pins the property rather than trusting it.
        """
        import importlib

        if not flydsl_utils.runtime_available():
            self.skipTest("FlyDSL runtime unavailable")
        module = importlib.import_module(
            "torch._inductor.kernel.vendored_templates.flydsl.kernels."
            "mxfp8_grouped_gemm_gfx950"
        )
        import flydsl.compiler as flyc

        group_sizes = [1, 3, 7, 15, 31, 63, 127, 255]
        k, n, g = 2048, 1408, len(group_sizes)
        a, b, a_scale, b_scale, offs, reference = self._make_mxfp8_grouped_inputs(
            group_sizes, k, n
        )
        # The kernel wants B as a stack of [N, K] planes, the layout the
        # lowering hands it after the template's permute.
        weight = b.permute(0, 2, 1)
        rows = int(offs[-1])
        outputs = {}
        for block_r in (64, 128, 256):
            for block_c in (128, 256):
                param = module.make_mxfp8_grouped_gemm_param_and_validate(
                    k, n, g, block_r, block_c
                )
                if param is None:
                    continue
                out = torch.zeros(a.shape[0], n, dtype=torch.bfloat16, device=a.device)
                module.launch_mxfp8_grouped_gemm_gfx950(
                    out,
                    a,
                    weight,
                    a_scale,
                    b_scale.reshape(g, n, -1),
                    offs,
                    param,
                    torch.cuda.current_stream(),
                    tensor_arg=lambda t: flyc.from_torch_tensor(
                        t
                    ).mark_layout_dynamic(),
                )
                outputs[(block_r, block_c)] = out
        self.assertGreaterEqual(len(outputs), 4)

        tiles = list(outputs)
        base = outputs[tiles[0]]
        for tile in tiles[1:]:
            differing = int((outputs[tile][:rows] != base[:rows]).sum())
            self.assertEqual(
                differing,
                0,
                f"tile {tile} disagrees with {tiles[0]} on {differing} elements",
            )
        self.assertEqual(
            base[:rows].float(), reference[:rows].float(), atol=6e-2, rtol=6e-2
        )

    def test_flydsl_mxfp8_grouped_mm_row_windows(self):
        """A token dim past the int32 operand limit is split, not truncated."""
        import importlib

        if not flydsl_utils.runtime_available():
            self.skipTest("FlyDSL runtime unavailable")

        module = importlib.import_module(
            "torch._inductor.kernel.vendored_templates.flydsl.kernels."
            "mxfp8_grouped_gemm_gfx950"
        )
        param = module.make_mxfp8_grouped_gemm_param(2048, 2048, 2, 256, 256)
        total_m = 1 << 22
        offs = torch.tensor([total_m // 2, total_m], dtype=torch.int32)
        windows = list(
            module._row_windows(total_m, param.k, param.n, offs, param.block_r)
        )
        # M * K here is 2**33, so this must split.
        self.assertGreater(len(windows), 1)
        covered = 0
        for row_start, rows, window_offs in windows:
            # Disjoint, contiguous, and never splitting a row tile.
            self.assertEqual(row_start, covered)
            self.assertEqual(rows % param.block_r, 0)
            # Offsets are rebased into the window and clamped to it, so a group
            # straddling the boundary ends at `rows` here and starts at 0 next.
            self.assertEqual(int(window_offs.max()), rows)
            self.assertGreaterEqual(int(window_offs.min()), 0)
            covered += rows
        self.assertEqual(covered, total_m)

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
