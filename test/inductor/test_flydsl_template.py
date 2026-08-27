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
from torch._inductor import config
from torch._inductor.codegen.flydsl import flydsl_utils
from torch._inductor.codegen.flydsl.flydsl_kernel import FlyDSLTemplateKernel
from torch._inductor.codegen.flydsl.flydsl_scheduling import (
    _get_flydsl_device_arch,
    FlyDSLScheduling,
)
from torch._inductor.codegen.flydsl.flydsl_template import FlyDSLTemplate
from torch._inductor.ir import Buffer, FixedLayout
from torch._inductor.kernel import mm
from torch._inductor.runtime.flydsl_cache import run_cached_flydsl
from torch._inductor.select_algorithm import PartialRender
from torch._inductor.test_case import TestCase
from torch._inductor.utils import OrderedSet, run_and_get_code
from torch._inductor.virtualized import V
from torch.nn.functional import ScalingType, SwizzleType  # type: ignore[attr-defined]
from torch.testing._internal.common_device_type import instantiate_device_type_tests
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
            ([70, 128], torch.bfloat16, [1, 72], 0, 64),
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

    def test_mm_gate_accepts_all_layouts(self):
        from torch._inductor.heuristics.template import flydsl as flydsl_heuristics
        from torch._inductor.kernel import mm
        from torch._inductor.kernel.vendored_templates.flydsl import (
            kernels as flydsl_kernels,
        )

        def node(size, stride):
            return SimpleNamespace(
                get_size=lambda: size,
                get_stride=lambda: stride,
                get_dtype=lambda: torch.bfloat16,
                get_layout=lambda: SimpleNamespace(offset=0),
            )

        m = n = 64
        k = 128
        layout = SimpleNamespace(
            stride=[n, 1],
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        sizevars = SimpleNamespace(
            statically_known_equals=lambda x, y: x == y,
            statically_known_multiple_of=lambda x, y: x % y == 0,
        )
        gemm_config = {"TILE_M": 128}

        with (
            V.set_graph_handler(SimpleNamespace(sizevars=sizevars)),
            mock.patch.object(mm, "use_flydsl_gemm_template", return_value=True),
            mock.patch.object(mm, "is_unaligned", return_value=False),
            mock.patch.object(
                flydsl_heuristics, "get_gemm_configs", return_value=[gemm_config]
            ),
            mock.patch.object(
                flydsl_heuristics,
                "is_gemm_config_valid_for_shape",
                return_value=True,
            ) as validate,
            mock.patch.dict(
                flydsl_kernels.__dict__,
                {"GEMM_DTYPE_BF16": 2, "GEMM_DTYPE_FP16": 3},
            ),
        ):
            for a_is_transposed, b_is_transposed in (
                (False, False),
                (False, True),
                (True, False),
                (True, True),
            ):
                with self.subTest(
                    a_is_transposed=a_is_transposed,
                    b_is_transposed=b_is_transposed,
                ):
                    mat1_stride = [1, m] if a_is_transposed else [k, 1]
                    mat2_stride = [1, k] if b_is_transposed else [n, 1]
                    result = mm.get_flydsl_mm_template_kwargs(
                        layout,
                        node([m, k], mat1_stride),
                        node([k, n], mat2_stride),
                        True,
                        True,
                    )

                    self.assertEqual(len(result), 1)
                    self.assertIs(result[0]["IS_MXFP"], False)
                    self.assertEqual(result[0]["A_IS_TRANSPOSED"], a_is_transposed)
                    self.assertEqual(result[0]["B_IS_TRANSPOSED"], b_is_transposed)
                    validate.assert_called_once_with(
                        m,
                        n,
                        k,
                        2,
                        gemm_config,
                        a_is_transposed=a_is_transposed,
                        b_is_transposed=b_is_transposed,
                    )
                    validate.reset_mock()

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

    def _assert_compiled_mm(
        self,
        a,
        b,
        *,
        expect_flydsl: bool | None = True,
        transpose_rhs: bool = True,
    ):
        from torch._inductor.utils import run_and_get_code

        def fn(lhs, rhs):
            return torch.mm(lhs, rhs.t() if transpose_rhs else rhs)

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
                self.assertNotIn("mat2.transpose(0, 1)", code)
                self.assertIn("_inductor_tensor_arg(mat2)", code)
                self.assertIn(".run(", code)
                self.assertIn("TILE_M: fx.Constexpr", code)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA/ROCm not available")
    @unittest.skipIf(torch.version.hip is None, "requires ROCm")
    @torch._inductor.config.patch(
        max_autotune_gemm=True,
        max_autotune_gemm_backends="FLYDSL",
    )
    def test_flydsl_gemm_all_layouts_accuracy(self):
        if not flydsl_utils.runtime_available():
            self.skipTest("FlyDSL runtime unavailable")
        if _get_flydsl_device_arch(torch.cuda.current_device()) != "gfx950":
            self.skipTest("requires gfx950")

        m = n = 64
        k = 96
        dtype = torch.bfloat16
        for layout in ("nn", "nt", "tn", "tt"):
            a_is_transposed = layout[0] == "t"
            b_is_transposed = layout[1] == "t"
            with self.subTest(layout=layout):
                a = (
                    torch.randn(k, m, device="cuda", dtype=dtype).t()
                    if a_is_transposed
                    else torch.randn(m, k, device="cuda", dtype=dtype)
                )
                b = (
                    torch.randn(n, k, device="cuda", dtype=dtype).t()
                    if b_is_transposed
                    else torch.randn(k, n, device="cuda", dtype=dtype)
                )
                code = self._assert_compiled_mm(
                    a,
                    b,
                    transpose_rhs=False,
                )
                self.assertIn(
                    f"A_IS_TRANSPOSED: fx.Constexpr = {a_is_transposed}", code
                )
                self.assertIn(
                    f"B_IS_TRANSPOSED: fx.Constexpr = {b_is_transposed}", code
                )

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


# E2M1 has eight magnitudes and no inf/nan; code = sign << 3 | magnitude index.
E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


class _FakeNode:
    def __init__(self, shape, stride, dtype, *, offset=0, device=None):
        self._shape = list(shape)
        self._stride = list(stride)
        self._dtype = dtype
        self._layout = SimpleNamespace(offset=offset)
        self._device = device or torch.device("cuda", 0)

    def get_device(self):
        return self._device

    def get_size(self):
        return self._shape

    def get_stride(self):
        return self._stride

    def get_dtype(self):
        return self._dtype

    def get_layout(self):
        return self._layout


def make_mxfp4_operand(rows, k, device, generator=None):
    """Return (packed fp4x2 [rows, k // 2], e8m0 scale [rows, k // 32], fp32).

    Values are drawn on the E2M1 grid and the scales are exact powers of two, so
    the fp32 tensor is the operand's exact value rather than an approximation of
    it. That keeps a numerics check measuring the kernel instead of the
    quantizer.
    """
    codes = torch.randint(
        0, 16, (rows, k), device=device, dtype=torch.uint8, generator=generator
    )
    lut = torch.tensor(E2M1_MAGNITUDES, device=device, dtype=torch.float32)
    magnitude = lut[(codes & 7).long()]
    values = torch.where(codes >= 8, -magnitude, magnitude)

    exponents = torch.randint(
        124, 131, (rows, k // 32), device=device, dtype=torch.uint8, generator=generator
    )
    scales = torch.pow(2.0, exponents.float() - 127.0)

    packed = (
        codes[:, 0::2].to(torch.int16) | (codes[:, 1::2].to(torch.int16) << 4)
    ).to(torch.uint8)
    return (
        packed.contiguous().view(torch.float4_e2m1fn_x2),
        exponents.contiguous().view(torch.float8_e8m0fnu),
        values * scales.repeat_interleave(32, dim=1),
    )


def scaled_mm_mxfp4(a, b_t, scale_a, scale_b, out_dtype):
    """A [M, K // 2] x B [K // 2, N] under the ROCm MXFP4 contract."""
    return torch._scaled_mm_v2(
        a,
        b_t,
        [scale_a],
        [ScalingType.BlockWise1x32.value],
        [SwizzleType.NO_SWIZZLE.value],
        [scale_b],
        [ScalingType.BlockWise1x32.value],
        [SwizzleType.NO_SWIZZLE.value],
        None,
        out_dtype,
    )


def _candidate_args(mxfp_format, **overrides):
    m, n, k = 64, 96, 256
    if mxfp_format == "mxfp4":
        dtype = torch.float4_e2m1fn_x2
        storage_k = k // 2
        contraction_dim = None
    elif mxfp_format == "mxfp8":
        dtype = torch.float8_e4m3fn
        storage_k = k
        contraction_dim = []
    else:
        raise AssertionError(f"unsupported MXFP format: {mxfp_format}")
    args = {
        "mat_a": _FakeNode((m, storage_k), (storage_k, 1), dtype),
        "mat_b": _FakeNode((storage_k, n), (1, storage_k), dtype),
        "scale_a": [
            _FakeNode((m, k // 32), (k // 32, 1), torch.float8_e8m0fnu)
        ],
        "recipe_a": [ScalingType.BlockWise1x32.value],
        "swizzle_a": [SwizzleType.NO_SWIZZLE.value],
        "scale_b": [
            _FakeNode((n, k // 32), (k // 32, 1), torch.float8_e8m0fnu)
        ],
        "recipe_b": [ScalingType.BlockWise1x32.value],
        "swizzle_b": [SwizzleType.NO_SWIZZLE.value],
        "bias": None,
        "out_dtype": torch.bfloat16,
        "contraction_dim": contraction_dim,
        "use_fast_accum": False,
    }
    args.update(overrides)
    return args


def _run_mxfp_tile(
    mxfp_format,
    shape,
    tile,
    out_dtype,
    a,
    b,
    scale_a,
    scale_b,
    *,
    a_is_transposed=False,
    b_is_transposed=True,
):
    import flydsl.compiler as flyc

    from torch._inductor.kernel.vendored_templates.flydsl.kernels import (
        make_mxfp_scaled_mm_gfx950,
    )

    m, n, k = shape
    block_m, block_n, block_k, stages, m_waves, n_waves, group_m = tile[:7]
    lds_scale = tile[7] if len(tile) == 8 else 0
    out = torch.zeros(m, n, device=a.device, dtype=out_dtype)
    a_u8 = a.view(torch.uint8)
    b_u8 = b.view(torch.uint8)
    scale_a_u8 = scale_a.view(torch.uint8)
    scale_b_u8 = scale_b.view(torch.uint8)
    launcher = make_mxfp_scaled_mm_gfx950(
        mxfp_format=mxfp_format,
        m=m,
        n=n,
        k=k,
        out_dtype="bfloat16" if out_dtype == torch.bfloat16 else "float16",
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        stages=stages,
        m_waves=m_waves,
        n_waves=n_waves,
        group_m=group_m,
        lds_scale=lds_scale,
        a_is_transposed=a_is_transposed,
        b_is_transposed=b_is_transposed,
    )
    runtime_args = (a_u8, b_u8, scale_a_u8, scale_b_u8, out, 0)
    compiled = flyc.compile(
        launcher,
        *[
            flyc.from_torch_tensor(t).mark_layout_dynamic()
            for t in (a_u8, b_u8, scale_a_u8, scale_b_u8, out)
        ],
        0,
    )
    compiled(*runtime_args)
    torch.cuda.synchronize()
    return out


def _mxfp8_reference(a, b, scale_a, scale_b, out_dtype):
    a_dequant = a.float() * scale_a.float().repeat_interleave(32, 1)
    b_dequant = b.float() * scale_b.float().repeat_interleave(32, 1)
    return (a_dequant @ b_dequant.t()).to(out_dtype)


def _with_outer_contiguous_storage(tensor):
    return (
        tensor.view(torch.uint8)
        .t()
        .contiguous()
        .t()
        .view(tensor.dtype)
    )


def _mxfp_operand_layouts(a, b_nk, a_is_transposed, b_is_transposed):
    a_arg = _with_outer_contiguous_storage(a) if a_is_transposed else a
    b_view = (
        b_nk if b_is_transposed else _with_outer_contiguous_storage(b_nk)
    )
    b_arg = b_view.view(torch.uint8).t().view(b_nk.dtype)
    return a_arg, b_arg, b_view


class TestFlyDSLMXFPMetadata(TestCase):
    @parametrize(
        "mxfp_format,contraction_dim",
        [("mxfp4", None), ("mxfp8", None), ("mxfp8", [])],
    )
    def test_exact_v2_signature(self, mxfp_format, contraction_dim):
        with mock.patch.object(torch.version, "hip", "test"):
            self.assertEqual(
                mm._get_rocm_mxfp_v2_format(
                    **_candidate_args(
                        mxfp_format, contraction_dim=contraction_dim
                    )
                ),
                mxfp_format,
            )

    @parametrize(
        "mxfp_format,override",
        [
            ("mxfp4", {"swizzle_a": [SwizzleType.SWIZZLE_32_4_4.value]}),
            ("mxfp4", {"recipe_b": [ScalingType.BlockWise1x16.value]}),
            ("mxfp4", {"out_dtype": torch.float32}),
            ("mxfp4", {"use_fast_accum": True}),
            ("mxfp4", {"contraction_dim": [1]}),
            ("mxfp8", {"swizzle_a": [SwizzleType.SWIZZLE_32_4_4.value]}),
            ("mxfp8", {"use_fast_accum": True}),
        ],
    )
    def test_rejects_out_of_contract_signature(self, mxfp_format, override):
        with mock.patch.object(torch.version, "hip", "test"):
            self.assertIsNone(
                mm._get_rocm_mxfp_v2_format(
                    **_candidate_args(mxfp_format, **override)
                )
            )

    @unittest.skipUnless(flydsl_utils.runtime_available(), "FlyDSL unavailable")
    def test_tile_config_units_are_elements(self):
        from torch._inductor.kernel.vendored_templates.flydsl.kernels import (
            mxfp_gemm_derived,
        )

        mxfp4 = mxfp_gemm_derived(
            "mxfp4",
            block_m=64, block_n=64, block_k=256, stages=2, m_waves=1, n_waves=1
        )
        mxfp8 = mxfp_gemm_derived(
            "mxfp8",
            block_m=64,
            block_n=64,
            block_k=256,
            stages=2,
            m_waves=1,
            n_waves=1,
        )
        mxfp8_lds_scale = mxfp_gemm_derived(
            "mxfp8",
            block_m=64,
            block_n=64,
            block_k=256,
            stages=2,
            m_waves=1,
            n_waves=1,
            lds_scale_req=1,
        )
        # TILE_K is logical elements for both formats; only storage width changes.
        self.assertEqual(mxfp4.block_k_bytes, 128)
        self.assertEqual(mxfp8.block_k_bytes, 256)
        self.assertEqual(mxfp4.a_stage_bytes, 64 * 128)
        self.assertEqual(mxfp8.a_stage_bytes, 64 * 256)
        self.assertEqual(mxfp4.k_halves, mxfp8.k_halves)
        self.assertTrue(mxfp8_lds_scale.lds_scale)

    @unittest.skipUnless(flydsl_utils.runtime_available(), "FlyDSL unavailable")
    def test_cache_signature_includes_lds_scale(self):
        from torch._inductor.kernel.vendored_templates.flydsl.kernels import (
            MXFPGemmParams,
        )

        global_scale = MXFPGemmParams(
            mxfp_format="mxfp4",
            m=256,
            n=256,
            k=512,
            out_dtype="bfloat16",
            lds_scale=0,
        )
        lds_scale = MXFPGemmParams(
            mxfp_format="mxfp4",
            m=256,
            n=256,
            k=512,
            out_dtype="bfloat16",
            lds_scale=1,
        )
        mxfp8 = MXFPGemmParams(
            mxfp_format="mxfp8",
            m=256,
            n=256,
            k=512,
            out_dtype="bfloat16",
            lds_scale=0,
        )
        transposed_a = MXFPGemmParams(
            mxfp_format="mxfp4",
            m=256,
            n=256,
            k=512,
            out_dtype="bfloat16",
            lds_scale=0,
            a_is_transposed=True,
        )
        self.assertNotEqual(
            global_scale.__cache_signature__(), lds_scale.__cache_signature__()
        )
        self.assertNotEqual(
            global_scale.__cache_signature__(), mxfp8.__cache_signature__()
        )
        self.assertNotEqual(
            global_scale.__cache_signature__(), transposed_a.__cache_signature__()
        )

    @unittest.skipUnless(flydsl_utils.runtime_available(), "FlyDSL unavailable")
    def test_unsupported_tile_k_is_rejected(self):
        from torch._inductor.kernel.vendored_templates.flydsl.kernels import (
            mxfp_gemm_derived,
        )

        with self.assertRaises(ValueError):
            mxfp_gemm_derived(
                "mxfp4",
                block_m=128, block_n=128, block_k=64, stages=2, m_waves=1, n_waves=1
            )


class TestFlyDSLMXFPMetadataConfig(TestCase):
    @parametrize("mxfp_format", ["mxfp8", "mxfp4"])
    @config.patch(flydsl_enable_autotuning=False)
    @unittest.skipUnless(flydsl_utils.runtime_available(), "FlyDSL unavailable")
    def test_operand_layouts(self, mxfp_format):
        args = _candidate_args(mxfp_format)
        m, k_storage = args["mat_a"].get_size()
        _, n = args["mat_b"].get_size()
        scale_a = args["scale_a"]
        scale_b = args["scale_b"]
        layout = SimpleNamespace(
            size=[64, 96],
            stride=[96, 1],
            dtype=torch.bfloat16,
            device=torch.device("cuda", 0),
            offset=0,
        )

        with mock.patch.object(mm, "use_flydsl_gemm_template", return_value=True):
            for a_is_transposed, b_is_transposed in (
                (False, False),
                (False, True),
                (True, False),
                (True, True),
            ):
                a = _FakeNode(
                    (m, k_storage),
                    (1, m) if a_is_transposed else (k_storage, 1),
                    args["mat_a"].get_dtype(),
                )
                b = _FakeNode(
                    (k_storage, n),
                    (1, k_storage) if b_is_transposed else (n, 1),
                    args["mat_b"].get_dtype(),
                )
                configs = mm.get_flydsl_mxfp_template_kwargs(
                    mxfp_format, layout, a, b, scale_a[0], scale_b[0]
                )
                self.assertEqual(len(configs), 1)
                gemm_config = configs[0]
                self.assertIs(gemm_config["IS_MXFP"], True)
                self.assertEqual(gemm_config["GEMM_M"], 64)
                self.assertEqual(gemm_config["GEMM_N"], 96)
                self.assertEqual(gemm_config["GEMM_K"], 256)
                self.assertEqual(gemm_config["OUT_DTYPE"], "bfloat16")
                self.assertEqual(gemm_config["A_IS_TRANSPOSED"], a_is_transposed)
                self.assertEqual(gemm_config["B_IS_TRANSPOSED"], b_is_transposed)
                self.assertEqual(64 % gemm_config["TILE_M"], 0)
                self.assertEqual(96 % gemm_config["TILE_N"], 0)
                self.assertEqual(256 % gemm_config["TILE_K"], 0)

            a = args["mat_a"]
            b = args["mat_b"]
            bad_b = _FakeNode(
                (k_storage, n),
                (n * 2, 2),
                args["mat_b"].get_dtype(),
            )
            self.assertEqual(
                mm.get_flydsl_mxfp_template_kwargs(
                    mxfp_format, layout, a, bad_b, scale_a[0], scale_b[0]
                ),
                [],
            )

            oversized_a = _FakeNode(
                (m, k_storage),
                (1, 1 << 31),
                args["mat_a"].get_dtype(),
            )
            self.assertEqual(
                mm.get_flydsl_mxfp_template_kwargs(
                    mxfp_format,
                    layout,
                    oversized_a,
                    b,
                    scale_a[0],
                    scale_b[0],
                ),
                [],
            )

            oversized_span_a = _FakeNode(
                (m, k_storage),
                (1, 1 << 26),
                args["mat_a"].get_dtype(),
            )
            self.assertEqual(
                mm.get_flydsl_mxfp_template_kwargs(
                    mxfp_format,
                    layout,
                    oversized_span_a,
                    b,
                    scale_a[0],
                    scale_b[0],
                ),
                [],
            )

            bad_scale = _FakeNode((64, 8), (8, 1), torch.float8_e8m0fnu, offset=1)
            self.assertEqual(
                mm.get_flydsl_mxfp_template_kwargs(
                    mxfp_format, layout, a, b, bad_scale, scale_b[0]
                ),
                [],
            )

            bad_device_scale = _FakeNode(
                (64, 8),
                (8, 1),
                torch.float8_e8m0fnu,
                device=torch.device("cpu"),
            )
            self.assertEqual(
                mm.get_flydsl_mxfp_template_kwargs(
                    mxfp_format, layout, a, b, bad_device_scale, scale_b[0]
                ),
                [],
            )

            with mock.patch.object(
                mm,
                "is_unaligned",
                side_effect=lambda node: node is a,
            ):
                self.assertEqual(
                    mm.get_flydsl_mxfp_template_kwargs(
                        mxfp_format, layout, a, b, scale_a[0], scale_b[0]
                    ),
                    [],
                )

    def test_precompile_uses_flydsl_compile_only_contract(self):
        source = mm.flydsl_gemm_template.source

        self.assertIn("{% if IS_MXFP %}", source)
        self.assertIn(
            '{{def_kernel("mat1", "mat2", "scale_a", "scale_b")}}', source
        )
        self.assertIn('{{def_kernel("mat1", "mat2")}}', source)
        self.assertIn(
            '{"COMPILE_ONLY": "1", "FLYDSL_COMPILE_ONLY": "1"}', source
        )


class _MXFPDeviceTest(TestCase):
    def _skip_unless_supported(self, device):
        if torch.version.hip is None:
            self.skipTest("requires ROCm")
        arch = torch.cuda.get_device_properties(device).gcnArchName.split(":", 1)[0]
        if arch != "gfx950":
            self.skipTest(f"requires gfx950, got {arch}")
        if not flydsl_utils.runtime_available():
            self.skipTest("FlyDSL runtime unavailable")


class TestFlyDSLMXFP8Device(_MXFPDeviceTest):
    def _make_inputs(self, m, n, k, device):
        a_values = ((torch.arange(m * k, device=device) % 9) - 4) / 2
        b_values = ((torch.arange(n * k, device=device) % 11) - 5) / 2
        a = a_values.reshape(m, k).to(torch.float8_e4m3fn)
        b = b_values.reshape(n, k).to(torch.float8_e4m3fn)

        k32 = k // 32
        a_exp = (
            torch.arange(m, device=device)[:, None]
            + torch.arange(k32, device=device)[None, :]
        ) % 5 - 2
        b_exp = (
            2 * torch.arange(n, device=device)[:, None]
            + torch.arange(k32, device=device)[None, :]
        ) % 5 - 2
        scale_a = torch.pow(2.0, a_exp.float()).to(torch.float8_e8m0fnu)
        scale_b = torch.pow(2.0, b_exp.float()).to(torch.float8_e8m0fnu)
        return a, b, scale_a, scale_b

    @parametrize("out_dtype", [torch.bfloat16, torch.float16])
    @parametrize(
        "a_is_transposed,b_is_transposed",
        [(False, False), (False, True), (True, False), (True, True)],
    )
    def test_scaled_mm_v2_flydsl_baseline(
        self, device, out_dtype, a_is_transposed, b_is_transposed
    ):
        self._skip_unless_supported(device)

        m, n, k = 64, 96, 256
        a, b, scale_a, scale_b = self._make_inputs(m, n, k, device)
        a_arg, b_arg, b_nk = _mxfp_operand_layouts(
            a, b, a_is_transposed, b_is_transposed
        )

        def fn(a, b_arg, scale_a, scale_b):
            return F.scaled_mm(
                a,
                b_arg,
                scale_a,
                ScalingType.BlockWise1x32,
                scale_b,
                ScalingType.BlockWise1x32,
                swizzle_a=SwizzleType.NO_SWIZZLE,
                swizzle_b=SwizzleType.NO_SWIZZLE,
                output_dtype=out_dtype,
            )

        reference = _mxfp8_reference(a_arg, b_nk, scale_a, scale_b, out_dtype)

        torch._dynamo.reset()
        with config.patch(
            max_autotune_gemm=True,
            max_autotune_gemm_backends="FLYDSL",
            flydsl_enable_autotuning=False,
        ):
            compiled = torch.compile(fn, backend="inductor", fullgraph=True)
            actual, (code,) = run_and_get_code(
                compiled, a_arg, b_arg, scale_a, scale_b
            )

        self.assertEqual(actual, reference, rtol=2e-2, atol=5e-1)
        self.assertIn("async_compile.flydsl", code)
        self.assertIn("make_mxfp_scaled_mm_gfx950", code)
        self.assertIn("mat2.transpose(0, 1)", code)
        self.assertIn(
            f"A_IS_TRANSPOSED: fx.Constexpr = {a_is_transposed}", code
        )
        self.assertIn(
            f"B_IS_TRANSPOSED: fx.Constexpr = {b_is_transposed}", code
        )
        self.assertNotIn("extern_kernels._scaled_mm_v2(", code)

    @parametrize(
        "a_is_transposed,b_is_transposed",
        [(False, False), (False, True), (True, False), (True, True)],
    )
    def test_mxfp8_tile_all_operand_layouts(
        self, device, a_is_transposed, b_is_transposed
    ):
        self._skip_unless_supported(device)
        shape = (64, 96, 256)
        tile = (32, 32, 128, 2, 1, 1, 0, 0)
        a, b, scale_a, scale_b = self._make_inputs(*shape, device)
        a_arg, _, b_nk = _mxfp_operand_layouts(
            a, b, a_is_transposed, b_is_transposed
        )
        out = _run_mxfp_tile(
            "mxfp8",
            shape,
            tile,
            torch.bfloat16,
            a_arg,
            b_nk,
            scale_a,
            scale_b,
            a_is_transposed=a_is_transposed,
            b_is_transposed=b_is_transposed,
        )
        reference = _mxfp8_reference(
            a_arg, b_nk, scale_a, scale_b, torch.bfloat16
        )
        self.assertEqual(out, reference, rtol=2e-2, atol=5e-1)

    # One entry per tiling feature the parameterized kernel exposes, so a
    # regression in LDS staging, register blocking or the staged pipeline shows
    # up as a numerical failure on the specific config that broke.
    @parametrize(
        "shape,tile,out_dtype",
        [
            # (m, n, k), (TILE_M, TILE_N, TILE_K, STAGES, M_WARPS, N_WARPS,
            # GROUP_M[, LDS_SCALE])
            (
                (64, 96, 256),
                (16, 16, 128, 2, 1, 1, 0),
                torch.bfloat16,
            ),  # minimal, 1 wave
            (
                (64, 64, 512),
                (64, 64, 128, 2, 1, 1, 0),
                torch.bfloat16,
            ),  # 4x4 register blocking
            (
                (128, 128, 512),
                (64, 64, 128, 2, 2, 2, 0),
                torch.bfloat16,
            ),  # 2x2 waves over LDS
            (
                (128, 128, 512),
                (64, 64, 256, 2, 2, 2, 0),
                torch.bfloat16,
            ),  # two MFMA steps / tile
            (
                (64, 64, 1024),
                (64, 64, 512, 2, 1, 1, 0),
                torch.bfloat16,
            ),  # four MFMA steps / tile
            (
                (128, 128, 512),
                (64, 64, 128, 3, 2, 2, 0),
                torch.bfloat16,
            ),  # odd stage count
            (
                (128, 128, 1024),
                (64, 64, 128, 4, 2, 2, 0),
                torch.bfloat16,
            ),  # 4-stage pipeline
            (
                (128, 128, 512),
                (128, 128, 128, 2, 4, 4, 0),
                torch.bfloat16,
            ),  # 1024 threads
            (
                (256, 256, 512),
                (128, 128, 128, 2, 2, 4, 0),
                torch.bfloat16,
            ),  # asymmetric waves
            (
                (1024, 256, 512),
                (128, 128, 128, 2, 2, 2, 4),
                torch.bfloat16,
            ),  # GROUP_M swizzle
            (
                (256, 256, 512),
                (256, 256, 128, 2, 2, 2, 0),
                torch.bfloat16,
            ),  # 8x8 deep blocking
            (
                (256, 256, 512),
                (256, 256, 128, 2, 2, 2, 0),
                torch.float16,
            ),  # fp16 direct store
            (
                (256, 128, 512),
                (256, 128, 128, 2, 2, 1, 0),
                torch.bfloat16,
            ),  # rectangular 8x8
            (
                (64, 64, 512),
                (64, 64, 256, 2, 1, 1, 0, 1),
                torch.bfloat16,
            ),  # shared LDS-staged scale path
            (
                (128, 128, 512),
                (128, 128, 256, 2, 2, 2, 0, 1),
                torch.bfloat16,
            ),  # multi-wave LDS-staged scale path
        ],
    )
    def test_mxfp8_tile_configs_match_reference(self, device, shape, tile, out_dtype):
        self._skip_unless_supported(device)
        m, n, k = shape
        a, b, scale_a, scale_b = self._make_inputs(m, n, k, device)
        out = _run_mxfp_tile(
            "mxfp8", shape, tile, out_dtype, a, b, scale_a, scale_b
        )
        reference = _mxfp8_reference(a, b, scale_a, scale_b, out_dtype)
        self.assertEqual(out, reference, rtol=2e-2, atol=5e-1)

    def test_scaled_mm_v2_flydsl_autotunes_multiple_configs(self, device):
        self._skip_unless_supported(device)

        from torch._inductor.heuristics.template import flydsl as flydsl_heuristics

        m, n, k = 128, 128, 512
        a, b, scale_a, scale_b = self._make_inputs(m, n, k, device)

        def fn(a, b, scale_a, scale_b):
            return F.scaled_mm(
                a,
                b.t(),
                scale_a,
                ScalingType.BlockWise1x32,
                scale_b,
                ScalingType.BlockWise1x32,
                swizzle_a=SwizzleType.NO_SWIZZLE,
                swizzle_b=SwizzleType.NO_SWIZZLE,
                output_dtype=torch.bfloat16,
            )

        expected = fn(a, b, scale_a, scale_b)

        with config.patch(
            max_autotune_gemm=True,
            max_autotune_gemm_backends="FLYDSL",
            max_autotune_gemm_search_space="DEFAULT",
            flydsl_enable_autotuning=True,
        ):
            # More than one tile divides this shape, so autotuning has a real
            # choice to make here.
            candidates = flydsl_heuristics.get_mxfp_gemm_configs_for_shape(
                "mxfp8", m, n, k, "bfloat16"
            )
            self.assertGreater(len(candidates), 1)

            torch._dynamo.reset()
            compiled = torch.compile(fn, backend="inductor", fullgraph=True)
            actual, (code,) = run_and_get_code(compiled, a, b, scale_a, scale_b)

        self.assertEqual(actual, expected, rtol=2e-2, atol=5e-1)
        self.assertIn("async_compile.flydsl", code)


class TestFlyDSLMXFP4Device(_MXFPDeviceTest):
    @parametrize(
        "shape,tile,out_dtype",
        [
            (
                (32, 32, 256),
                (16, 16, 128, 2, 1, 1, 0, 0),
                torch.bfloat16,
            ),  # scalar scale fallback
            (
                (32, 64, 1024),
                (32, 64, 512, 2, 1, 2, 0, 0),
                torch.bfloat16,
            ),  # packed-unit global scale path
            (
                (32, 64, 1024),
                (32, 64, 512, 2, 1, 2, 0, 1),
                torch.bfloat16,
            ),  # shared LDS-staged scale path
            (
                (128, 128, 512),
                (128, 128, 256, 2, 2, 2, 0, 1),
                torch.bfloat16,
            ),  # multi-wave LDS-staged scale path
            (
                (128, 128, 1024),
                (64, 64, 128, 4, 2, 2, 0, 0),
                torch.bfloat16,
            ),  # deep pipeline with FP4 DMA wait counts
            (
                (256, 256, 512),
                (256, 256, 256, 2, 4, 2, 0, 0),
                torch.float16,
            ),  # asymmetric waves and fp16 output
        ],
    )
    def test_mxfp4_tile_configs_match_reference(
        self, device, shape, tile, out_dtype
    ):
        self._skip_unless_supported(device)
        m, n, k = shape
        a, scale_a, a_ref = make_mxfp4_operand(m, k, device)
        b, scale_b, b_ref = make_mxfp4_operand(n, k, device)
        out = _run_mxfp_tile(
            "mxfp4", shape, tile, out_dtype, a, b, scale_a, scale_b
        )
        reference = (a_ref @ b_ref.t()).to(out_dtype)
        rel_l2 = ((out.float() - reference.float()).norm() / reference.norm()).item()
        self.assertLess(rel_l2, 5e-3)

    @parametrize(
        "a_is_transposed,b_is_transposed",
        [(False, False), (False, True), (True, False), (True, True)],
    )
    def test_mxfp4_tile_all_operand_layouts(
        self, device, a_is_transposed, b_is_transposed
    ):
        self._skip_unless_supported(device)
        shape = (64, 96, 512)
        tile = (32, 32, 256, 2, 1, 1, 0, 0)
        a, scale_a, a_ref = make_mxfp4_operand(shape[0], shape[2], device)
        b, scale_b, b_ref = make_mxfp4_operand(shape[1], shape[2], device)
        a_arg, _, b_nk = _mxfp_operand_layouts(
            a, b, a_is_transposed, b_is_transposed
        )
        out = _run_mxfp_tile(
            "mxfp4",
            shape,
            tile,
            torch.bfloat16,
            a_arg,
            b_nk,
            scale_a,
            scale_b,
            a_is_transposed=a_is_transposed,
            b_is_transposed=b_is_transposed,
        )
        reference = a_ref @ b_ref.t()
        rel_l2 = ((out.float() - reference).norm() / reference.norm()).item()
        self.assertLess(rel_l2, 5e-3)

    @parametrize("out_dtype", [torch.bfloat16, torch.float16])
    def test_scaled_mm_v2_flydsl_matches_reference(self, device, out_dtype):
        self._skip_unless_supported(device)
        m, n, k = 256, 256, 512
        a, scale_a, a_ref = make_mxfp4_operand(m, k, device)
        b, scale_b, b_ref = make_mxfp4_operand(n, k, device)
        b_t = b.view(torch.uint8).t().view(torch.float4_e2m1fn_x2)

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "FLYDSL",
                "flydsl_enable_autotuning": False,
            }
        ):
            compiled = torch.compile(scaled_mm_mxfp4, dynamic=False)
            out, code = run_and_get_code(compiled, a, b_t, scale_a, scale_b, out_dtype)

        self.assertIn("make_mxfp_scaled_mm_gfx950", "\n".join(code))
        # The tolerance covers FP32 accumulation-order and output-rounding
        # differences relative to the reference matmul.
        reference = a_ref @ b_ref.t()
        rel_l2 = ((out.float() - reference).norm() / reference.norm()).item()
        self.assertLess(rel_l2, 5e-3)

    @parametrize(
        "a_is_transposed,b_is_transposed",
        [(False, False), (False, True), (True, False), (True, True)],
    )
    def test_scaled_mm_v2_all_operand_layouts(
        self, device, a_is_transposed, b_is_transposed
    ):
        self._skip_unless_supported(device)
        m, n, k = 64, 96, 512
        a, scale_a, a_ref = make_mxfp4_operand(m, k, device)
        b, scale_b, b_ref = make_mxfp4_operand(n, k, device)
        a_arg, b_arg, _ = _mxfp_operand_layouts(
            a, b, a_is_transposed, b_is_transposed
        )

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "FLYDSL",
                "flydsl_enable_autotuning": False,
            }
        ):
            compiled = torch.compile(scaled_mm_mxfp4, dynamic=False)
            out, code = run_and_get_code(
                compiled,
                a_arg,
                b_arg,
                scale_a,
                scale_b,
                torch.bfloat16,
            )

        generated = "\n".join(code)
        self.assertIn("make_mxfp_scaled_mm_gfx950", generated)
        self.assertIn(
            f"A_IS_TRANSPOSED: fx.Constexpr = {a_is_transposed}", generated
        )
        self.assertIn(
            f"B_IS_TRANSPOSED: fx.Constexpr = {b_is_transposed}", generated
        )
        reference = a_ref @ b_ref.t()
        rel_l2 = ((out.float() - reference).norm() / reference.norm()).item()
        self.assertLess(rel_l2, 5e-3)

    def test_unsupported_signature_falls_back(self, device):
        self._skip_unless_supported(device)
        m, n, k = 64, 64, 128
        a = torch.randn(m, k, device=device).to(torch.float8_e4m3fn)
        b = torch.randn(n, k, device=device).to(torch.float8_e4m3fn).t()
        scale_a = torch.ones((), device=device)
        scale_b = torch.ones((), device=device)

        def tensorwise(a, b, scale_a, scale_b):
            return torch._scaled_mm_v2(
                a,
                b,
                [scale_a],
                [ScalingType.TensorWise.value],
                [SwizzleType.NO_SWIZZLE.value],
                [scale_b],
                [ScalingType.TensorWise.value],
                [SwizzleType.NO_SWIZZLE.value],
                None,
                torch.bfloat16,
            )

        with config.patch(
            {"max_autotune": True, "max_autotune_gemm_backends": "ATEN,FLYDSL"}
        ):
            _, code = run_and_get_code(
                torch.compile(tensorwise, dynamic=False), a, b, scale_a, scale_b
            )
        self.assertNotIn("make_mxfp_scaled_mm_gfx950", "\n".join(code))


instantiate_parametrized_tests(TestFlyDSLMXFPMetadata)
instantiate_parametrized_tests(TestFlyDSLMXFPMetadataConfig)
instantiate_device_type_tests(TestFlyDSLMXFP8Device, globals(), only_for="cuda")
instantiate_device_type_tests(TestFlyDSLMXFP4Device, globals(), only_for="cuda")

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
