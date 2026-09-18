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
from torch._inductor import config as inductor_config
from torch._inductor.codegen.flydsl import flydsl_utils
from torch._inductor.codegen.flydsl.flydsl_kernel import FlyDSLTemplateKernel
from torch._inductor.codegen.flydsl.flydsl_scheduling import (
    _get_flydsl_device_arch,
    FlyDSLScheduling,
)
from torch._inductor.codegen.flydsl.flydsl_template import FlyDSLTemplate
from torch._inductor.heuristics.template.flydsl import FlyDSLGemmConfig
from torch._inductor.ir import Buffer, FixedLayout
from torch._inductor.kernel import mm
from torch._inductor.kernel.vendored_templates.flydsl import kernels
from torch._inductor.runtime.flydsl_cache import run_cached_flydsl
from torch._inductor.select_algorithm import PartialRender
from torch._inductor.test_case import TestCase
from torch._inductor.utils import OrderedSet, run_and_get_code
from torch._inductor.virtualized import V
from torch.nn.functional import ScalingType, SwizzleType  # type: ignore[attr-defined]
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_quantized import _floatx_unpacked_to_f32, to_mxfp
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
    def _grouped_gemm_grid_param_stub(self, **overrides):
        defaults = {
            "stages": 2,
            "block_m": 64,
            "block_n": 128,
            "block_k": 64,
            "in_data_bytes": 2,
            "out_data_bytes": 2,
            "block_threads": 256,
        }
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def _gfx950_device_stub(self, **overrides):
        defaults = {
            "multi_processor_count": 256,
            "shared_memory_per_multiprocessor": 163840,
            "max_threads_per_multi_processor": 2048,
        }
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

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

    @parametrize(
        "input_names",
        ((), ("mat1", "mat2", "scale_a", "scale_b")),
    )
    def test_precompile_metadata_supports_inputs(self, input_names):
        scheduling = FlyDSLScheduling(scheduler=None)
        layout = FixedLayout(torch.device("cpu"), torch.float32, [1], [1])
        kernel = FlyDSLTemplateKernel(
            kernel_name="metadata",
            input_nodes=[Buffer(name=name, layout=layout) for name in input_names],
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
            kernel.def_kernel(*input_names)
            metadata = scheduling._build_precompile_metadata(
                kernel, SimpleNamespace(layout=layout)
            )

        self.assertIsNotNone(metadata)
        names = (*input_names, "output")
        self.assertEqual(metadata["precompile_shapes"], dict.fromkeys(names, [1]))
        self.assertEqual(metadata["precompile_strides"], dict.fromkeys(names, [1]))
        self.assertEqual(metadata["precompile_dtypes"], dict.fromkeys(names, "float32"))

    @parametrize(
        "size,dtype,stride,offset,n",
        (
            ([1, 64, 128], torch.float16, [8192, 128, 1], 0, 64),
            ([64, 128], torch.float32, [128, 1], 0, 64),
            ([64, 128], torch.float16, [129, 1], 0, 64),
            ([64, 128], torch.float16, [128, 1], 1, 64),
            ([64, 72], torch.float16, [72, 1], 0, 64),
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

    @parametrize(
        "a_is_transposed,b_is_transposed",
        ((False, False), (False, True), (True, False), (True, True)),
    )
    def test_mm_gate_accepts_all_layouts(self, a_is_transposed, b_is_transposed):
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

        m = 64
        n = 40
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
            ) as get_configs,
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
            self.assertEqual(result[0]["A_IS_TRANSPOSED"], a_is_transposed)
            self.assertEqual(result[0]["B_IS_TRANSPOSED"], b_is_transposed)
            get_configs.assert_called_once_with()
            validate.assert_called_once_with(
                m,
                n,
                k,
                2,
                gemm_config,
                a_is_transposed=a_is_transposed,
                b_is_transposed=b_is_transposed,
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA/ROCm not available")
    @unittest.skipIf(torch.version.hip is None, "requires ROCm")
    @parametrize(
        "m,n,a_is_transposed,b_is_transposed,expected",
        (
            (64, 40, False, False, True),
            (64, 36, False, True, False),
            (70, 40, True, True, False),
        ),
    )
    def test_gemm_config_shape_alignment(
        self, m, n, a_is_transposed, b_is_transposed, expected
    ):
        from torch._inductor.heuristics.template import flydsl as flydsl_heuristics

        if not flydsl_utils.runtime_available():
            self.skipTest("FlyDSL runtime unavailable")
        if _get_flydsl_device_arch(torch.cuda.current_device()) != "gfx950":
            self.skipTest("requires gfx950")

        from torch._inductor.kernel.vendored_templates.flydsl.kernels import (
            GEMM_DTYPE_BF16,
        )

        gemm_config = asdict(flydsl_heuristics.FlyDSLGemmConfig())
        self.assertEqual(
            flydsl_heuristics.is_gemm_config_valid_for_shape(
                m,
                n,
                128,
                GEMM_DTYPE_BF16,
                gemm_config,
                a_is_transposed=a_is_transposed,
                b_is_transposed=b_is_transposed,
            ),
            expected,
        )

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
        assertion = self.assertIn if expect_flydsl else self.assertNotIn
        assertion("async_compile.flydsl", code)
        self.assertEqual(result, fn(a, b), atol=3e-2, rtol=3e-2)
        return code

    def _assert_compiled_grouped_mm(self, a, b, offs, *, expect_flydsl: bool = True):
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
                self.assertNotIn("mat2.transpose(0, 1)", code)
                self.assertRegex(
                code, r"tensor_args\s*=\s*\(\s*output\s*,\s*mat1\s*,\s*mat2\s*\)"
            )
                self.assertIn("_inductor_tensor_arg(arg) for arg in tensor_args", code)
                self.assertIn(".run(", code)
                self.assertIn("TILE_M: fx.Constexpr", code)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA/ROCm not available")
    @unittest.skipIf(torch.version.hip is None, "requires ROCm")
    @parametrize(
        "layout,dtype,k,tile_size,tile_k,use_half_tile_interleaved",
        (
            ("nn", torch.bfloat16, 96, 128, 64, False),
            ("nt", torch.bfloat16, 96, 128, 64, False),
            ("tn", torch.bfloat16, 96, 128, 64, False),
            ("tt", torch.bfloat16, 96, 128, 64, False),
            ("nn", torch.bfloat16, 96, 128, 64, True),
            ("nt", torch.bfloat16, 96, 128, 64, True),
            ("tn", torch.bfloat16, 96, 128, 64, True),
            ("tt", torch.bfloat16, 96, 128, 64, True),
            ("nn", torch.float16, 128, 128, 64, False),
            ("tn", torch.float16, 128, 128, 64, False),
            ("tt", torch.float16, 128, 128, 64, False),
            ("nn", torch.float16, 128, 128, 64, True),
            ("tn", torch.float16, 128, 128, 64, True),
            ("tt", torch.float16, 128, 128, 64, True),
            ("tt", torch.bfloat16, 96, 256, 64, False),
            ("nt", torch.bfloat16, 128, 128, 128, False),
            ("nt", torch.bfloat16, 256, 64, 256, False),
        ),
    )
    @torch._inductor.config.patch(
        max_autotune_gemm=True,
        max_autotune_gemm_backends="FLYDSL",
        flydsl_enable_autotuning=True,
    )
    def test_flydsl_gemm_all_layouts_accuracy(
        self, layout, dtype, k, tile_size, tile_k, use_half_tile_interleaved
    ):
        from torch._inductor.heuristics.template import flydsl as flydsl_heuristics

        if not flydsl_utils.runtime_available():
            self.skipTest("FlyDSL runtime unavailable")
        if _get_flydsl_device_arch(torch.cuda.current_device()) != "gfx950":
            self.skipTest("requires gfx950")

        m = 64
        n = 40
        # HTI halves the 128 tile, covering 64-, 128-, and 256-row LDS layouts.
        waves = 2 if use_half_tile_interleaved or tile_size == 64 else 4
        gemm_config = asdict(
            flydsl_heuristics.FlyDSLGemmConfig(
                TILE_M=tile_size,
                TILE_N=tile_size,
                TILE_K=tile_k,
                M_WAVES=waves,
                N_WAVES=waves,
                USE_HALF_TILE_INTERLEAVED=use_half_tile_interleaved,
            )
        )
        a_is_transposed = layout[0] == "t"
        b_is_transposed = layout[1] == "t"
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
        with mock.patch.object(
            flydsl_heuristics, "get_gemm_configs", return_value=[gemm_config]
        ) as get_configs:
            code = self._assert_compiled_mm(
                a,
                b,
                transpose_rhs=False,
            )
        get_configs.assert_called_once_with()
        self.assertIn(f"TILE_M: fx.Constexpr = {tile_size}", code)
        self.assertIn(f"TILE_N: fx.Constexpr = {tile_size}", code)
        self.assertIn(f"TILE_K: fx.Constexpr = {tile_k}", code)
        self.assertIn(f"GEMM_N: fx.Constexpr = {n}", code)
        self.assertIn(
            f"USE_HALF_TILE_INTERLEAVED: fx.Constexpr = {use_half_tile_interleaved}",
            code,
        )
        self.assertIn(f"A_IS_TRANSPOSED: fx.Constexpr = {a_is_transposed}", code)
        self.assertIn(f"B_IS_TRANSPOSED: fx.Constexpr = {b_is_transposed}", code)

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
        configs_by_hti = {}
        for config in configs:
            configs_by_hti.setdefault(config["USE_HALF_TILE_INTERLEAVED"], config)
        for use_hti in (False, True):
            self.assertIn(
                use_hti,
                configs_by_hti,
                f"missing config with USE_HALF_TILE_INTERLEAVED={use_hti}; "
                f"available configs: {configs}",
            )
        configs = [configs_by_hti[False], configs_by_hti[True]]

        with mock.patch.object(
            flydsl_heuristics, "get_gemm_configs", return_value=configs
        ):
            for k in (32, 64, 128):
                with self.subTest(k=k):
                    a = torch.randn(64, k, device="cuda", dtype=torch.bfloat16)
                    b = torch.randn(64, k, device="cuda", dtype=torch.bfloat16)
                    self._assert_compiled_mm(a, b)

    def test_flydsl_grouped_gemm_config_schema(self):
        from torch._inductor.heuristics.template import flydsl as flydsl_heuristics

        flydsl_heuristics.get_default_grouped_gemm_configs.cache_clear()
        self.addCleanup(flydsl_heuristics.get_default_grouped_gemm_configs.cache_clear)
        default_config = flydsl_heuristics.DEFAULT_GROUPED_GEMM_CONFIG
        self.assertEqual(
            tuple(asdict(default_config).items()),
            (
                ("TILE_M", 128),
                ("TILE_N", 128),
                ("TILE_K", 64),
                ("STAGES", 2),
                ("M_WAVES", 1),
                ("N_WAVES", 4),
                ("GROUP_M", 0),
                ("USE_HALF_TILE_INTERLEAVED", False),
            ),
        )
        if flydsl_utils.runtime_available():
            self.assertIsNotNone(
                flydsl_heuristics._make_gemm_param(asdict(default_config))
            )
        with (
            mock.patch.object(flydsl_heuristics, "_make_gemm_param"),
            torch._inductor.config.patch(flydsl_enable_autotuning=False),
        ):
            configs = flydsl_heuristics.get_default_grouped_gemm_configs()
            selected = flydsl_heuristics.get_grouped_gemm_configs()
        self.assertIn(default_config, configs)
        self.assertEqual(selected, [asdict(default_config)])

    def test_flydsl_grouped_gemm_exhaustive_layout_filter(self):
        from torch._inductor.heuristics.template import flydsl as flydsl_heuristics

        getter = flydsl_heuristics.get_exhaustive_grouped_gemm_configs
        getter.cache_clear()
        self.addCleanup(getter.cache_clear)
        valid = flydsl_heuristics.DEFAULT_GROUPED_GEMM_CONFIG
        small_n = flydsl_heuristics.FlyDSLGemmConfig(32, 32, 64, 2, 1, 2, 0)
        invalid_cshuffle = flydsl_heuristics.FlyDSLGemmConfig(16, 96, 64, 2, 1, 2, 0)
        with mock.patch.object(
            flydsl_heuristics,
            "get_exhaustive_gemm_configs",
            return_value=[small_n, invalid_cshuffle, valid],
        ):
            configs = getter()
        self.assertEqual(configs, [valid])

    @parametrize(
        "config_args,n,expected",
        [
            ((32, 32, 64, 2, 1, 2, 0), 32, False),
            ((16, 96, 64, 2, 1, 2, 0), 96, False),
            ((128, 128, 64, 2, 1, 4, 0), 128, True),
        ],
    )
    def test_flydsl_grouped_gemm_layout_validation(self, config_args, n, expected):
        from torch._inductor.heuristics.template import flydsl as flydsl_heuristics

        gemm_config = asdict(flydsl_heuristics.FlyDSLGemmConfig(*config_args))
        with mock.patch.object(
            flydsl_heuristics,
            "is_gemm_config_valid_for_shape",
            return_value=True,
        ):
            valid = flydsl_heuristics.is_grouped_gemm_config_valid_for_shape(
                128, n, 128, 0, gemm_config
            )
        self.assertEqual(valid, expected)

    @parametrize(
        "total_m,n,group_count,param_overrides,device_overrides,expected",
        [
            (0, 128, 6, {}, {}, 1),
            (201, 128, 6, {}, {}, 9),
            (
                201,
                128,
                6,
                {"block_m": 128, "block_n": 128, "block_threads": 256},
                {},
                7,
            ),
        ],
    )
    def test_flydsl_grouped_gemm_persistent_grid_size(
        self,
        total_m,
        n,
        group_count,
        param_overrides,
        device_overrides,
        expected,
    ):
        from torch._inductor.kernel.vendored_templates.flydsl.kernels import (
            get_grouped_gemm_persistent_grid_size,
        )

        param = self._grouped_gemm_grid_param_stub(**param_overrides)
        device = self._gfx950_device_stub(**device_overrides)
        grid_size = get_grouped_gemm_persistent_grid_size(
            param, total_m, n, group_count, device
        )
        self.assertEqual(grid_size, expected)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA/ROCm not available")
    @unittest.skipIf(torch.version.hip is None, "requires ROCm")
    @parametrize("dtype", (torch.bfloat16, torch.float16))
    @torch._inductor.config.patch(
        max_autotune_gemm=True,
        max_autotune_gemm_backends="FLYDSL",
        flydsl_enable_autotuning=False,
    )
    def test_flydsl_grouped_mm_grid_stride_e2e(self, dtype):
        if not flydsl_utils.runtime_available():
            self.skipTest("FlyDSL runtime unavailable")

        group_sizes = torch.tensor(
            [0, 1, 640, 0, 130, 3], device="cuda", dtype=torch.int32
        )
        offs = group_sizes.cumsum(0).to(torch.int32)
        a = torch.randn(int(group_sizes.sum()), 128, device="cuda", dtype=dtype)
        b = torch.randn(group_sizes.numel(), 128, 128, device="cuda", dtype=dtype)
        with mock.patch(
            "torch._inductor.kernel.vendored_templates.flydsl.kernels.get_grouped_gemm_persistent_grid_size",
            return_value=4,
        ) as grid_size:
            code = self._assert_compiled_grouped_mm(a, b, offs)
        grid_size.assert_called_once()
        self.assertIn("FLYDSL_COMPILE_ONLY", code)
        self.assertIn("_precompile", code)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA/ROCm not available")
    @unittest.skipIf(torch.version.hip is None, "requires ROCm")
    @torch._inductor.config.patch(
        max_autotune_gemm=True,
        max_autotune_gemm_backends="FLYDSL",
        max_autotune_gemm_search_space="EXHAUSTIVE",
        flydsl_enable_autotuning=True,
        autotune_in_subproc=True,
    )
    def test_flydsl_grouped_mm_autotune_e2e(self):
        from torch._inductor.heuristics.template import flydsl as flydsl_heuristics

        if not flydsl_utils.runtime_available():
            self.skipTest("FlyDSL runtime unavailable")

        configs = [
            asdict(config)
            for config in flydsl_heuristics.get_default_grouped_gemm_configs()
        ]
        configs_by_hti = {}
        for config in configs:
            configs_by_hti.setdefault(config["USE_HALF_TILE_INTERLEAVED"], config)
        for use_hti in (False, True):
            self.assertIn(
                use_hti,
                configs_by_hti,
                f"missing config with USE_HALF_TILE_INTERLEAVED={use_hti}; "
                f"available configs: {configs}",
            )
        configs = [configs_by_hti[False], configs_by_hti[True]]
        group_sizes = torch.tensor(
            [0, 1, 67, 0, 130, 3], device="cuda", dtype=torch.int32
        )
        offs = group_sizes.cumsum(0).to(torch.int32)
        k = 96
        a = torch.randn(int(group_sizes.sum()), k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(
            group_sizes.numel(), k, 128, device="cuda", dtype=torch.bfloat16
        )
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
        from torch._inductor.heuristics.template import flydsl as flydsl_heuristics

        if not flydsl_utils.runtime_available():
            self.skipTest("FlyDSL runtime unavailable")

        config_fields = (
            "TILE_M",
            "TILE_N",
            "TILE_K",
            "STAGES",
            "M_WAVES",
            "N_WAVES",
            "GROUP_M",
            "USE_HALF_TILE_INTERLEAVED",
        )
        cases = (
            (
                "block_swizzle_across_groups",
                2,
                4096,
                1024,
                128,
                (128, 128, 64, 2, 1, 4, 4, False),
            ),
            (
                "hti_odd_k_tiles",
                1,
                128,
                128,
                320,
                (128, 128, 64, 2, 2, 2, 0, True),
            ),
        )
        configs = [
            asdict(config)
            for config in flydsl_heuristics.get_default_grouped_gemm_configs()
        ]
        configs_by_values = {
            tuple(config[field] for field in config_fields): config
            for config in configs
        }

        for name, groups, m, n, k, expected_values in cases:
            with self.subTest(name=name):
                self.assertIn(
                    expected_values,
                    configs_by_values,
                    f"missing grouped GEMM config {expected_values}; "
                    f"available configs: {tuple(configs_by_values)}",
                )
                config = configs_by_values[expected_values]
                group_sizes = torch.full((groups,), m, device="cuda", dtype=torch.int32)
                offs = group_sizes.cumsum(0).to(torch.int32)
                a = torch.randn(groups * m, k, device="cuda", dtype=torch.bfloat16)
                b = torch.randn(groups, k, n, device="cuda", dtype=torch.bfloat16)
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
    @parametrize("dtype", (torch.bfloat16, torch.float16))
    @torch._inductor.config.patch(
        max_autotune_gemm=True,
        max_autotune_gemm_backends="ATEN,FLYDSL",
    )
    def test_flydsl_grouped_mm_fallback_e2e(self, dtype):
        if not flydsl_utils.runtime_available():
            self.skipTest("FlyDSL runtime unavailable")

        group_sizes = torch.tensor([32, 64], device="cuda", dtype=torch.int32)
        offs = group_sizes.cumsum(0).to(torch.int32)
        k = 128
        total_m = int(group_sizes.sum())
        a = torch.randn(total_m, k, device="cuda", dtype=dtype)
        b = torch.randn(2, k, 128, device="cuda", dtype=dtype)
        a_padded = torch.randn(total_m, k + 8, device="cuda", dtype=dtype)[:, :k]
        b_padded = torch.randn(2, k, 136, device="cuda", dtype=dtype)[..., :128]
        a_unaligned_base = torch.as_strided(
            torch.randn(total_m * k + 8, device="cuda", dtype=dtype),
            (total_m * k + 7,),
            (1,),
            storage_offset=1,
        )
        a_aligned = torch.as_strided(
            torch.randn(total_m * k + 8, device="cuda", dtype=dtype),
            (total_m, k),
            (k, 1),
            storage_offset=8,
        )
        b_aligned = torch.as_strided(
            torch.randn(2 * k * 128 + 8, device="cuda", dtype=dtype),
            (2, k, 128),
            (k * 128, 128, 1),
            storage_offset=8,
        )
        with torch._inductor.config.patch(max_autotune_gemm_backends="FLYDSL"):
            self._assert_compiled_grouped_mm(a_aligned, b_aligned, offs)

        cases = (
            (
                "b_transposed",
                a,
                torch.randn(2, 256, k, device="cuda", dtype=dtype).transpose(-1, -2),
            ),
            (
                "n_not_tile_divisible",
                a,
                torch.randn(2, k, 96, device="cuda", dtype=dtype),
            ),
            ("a_padded_stride", a_padded, b),
            ("b_padded_stride", a, b_padded),
        )

        for name, case_a, case_b in cases:
            with self.subTest(name=name):
                self._assert_compiled_grouped_mm(
                    case_a, case_b, offs, expect_flydsl=False
                )

        from torch._inductor.utils import run_and_get_code

        def fn(a_base, b, offs):
            a = torch.as_strided(
                a_base,
                (total_m, k),
                (k, 1),
                storage_offset=8,
            )
            return F.grouped_mm(a, b, offs=offs)

        aligned_view = torch.as_strided(
            a_unaligned_base,
            (total_m, k),
            (k, 1),
            storage_offset=8,
        )
        self.assertNotEqual(a_unaligned_base.data_ptr() % 16, 0)
        self.assertEqual(aligned_view.data_ptr() % 16, 0)
        torch._dynamo.reset()
        result, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor"), a_unaligned_base, b, offs
        )
        self.assertNotIn("async_compile.flydsl", code)
        self.assertIn("extern_kernels._grouped_mm", code)
        self.assertEqual(result, fn(a_unaligned_base, b, offs), atol=3e-2, rtol=3e-2)


def _mxfp_case(mxfp_format, shape, device, a_is_transposed=False, b_is_transposed=True):
    operands, scales, references = [], [], []
    for rows, transposed in zip(shape[:2], (a_is_transposed, not b_is_transposed)):
        k = shape[2]
        scale, operand = to_mxfp(
            torch.randn(rows, k, device=device), format=mxfp_format
        )
        if mxfp_format == "mxfp4":
            packed = operand.view(torch.uint8)
            codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
            reference = _floatx_unpacked_to_f32(codes, 2, 1)
        else:
            reference = operand.float()
        # Independent K-group scales catch premature recycling of live LDS stages.
        scale = (
            scale.view(torch.uint8).int()
            + torch.randint(-2, 3, scale.shape, device=device)
        ).to(torch.uint8)
        reference *= torch.exp2(scale.float() - 127).repeat_interleave(32, dim=1)
        operand = operand.view(torch.uint8)
        operands.append(operand.t().contiguous().t() if transposed else operand)
        scales.append(scale.view(torch.float8_e8m0fnu))
        references.append(reference)
    dtype = torch.float4_e2m1fn_x2 if mxfp_format == "mxfp4" else torch.float8_e4m3fn
    a, b = operands
    a_ref, b_ref = references
    return (a.view(dtype), b.t().view(dtype), *scales), a_ref @ b_ref.t()


def _mxfp_param(
    mxfp_format, tile, k, out_dtype=torch.bfloat16, transposed=(False, True), **kwargs
):
    cfg = {key.lower(): value for key, value in asdict(FlyDSLGemmConfig(*tile)).items()}
    kt = cfg["tile_k"]
    return kernels.make_gemm_gfx950_param(
        **cfg,
        dtype_id=4 if mxfp_format == "mxfp4" else 5,
        out_dtype_id=2 if out_dtype == torch.bfloat16 else 3,
        a_is_transposed=transposed[0],
        b_is_transposed=transposed[1],
        has_k_tail=kernels.infer_has_k_tail(k, kt, cfg["stages"])
        or (cfg["use_half_tile_interleaved"] and (k + kt - 1) // kt % 2 != 0),
        **kwargs,
    )


def _mxfp_call(inputs, out, param, bias=None, compiler=None):
    import flydsl.compiler as flyc

    args = (out, *(t.view(torch.uint8) for t in inputs), out if bias is None else bias)
    stream = torch.cuda.current_stream().cuda_stream
    compiled = run_cached_flydsl(
        kernels.gemm_mxfp_gfx950,
        *(flyc.from_torch_tensor(t).mark_layout_dynamic() for t in args),
        param,
        stream,
        constexpr_param=param,
        compiler=compiler or flyc.compile,
        dispatch_args=(*args, param, stream),
    )
    return lambda: compiled(*args, param, torch.cuda.current_stream().cuda_stream)


def _scaled_mm_mxfp(
    a, b, sa, sb, out_dtype=torch.bfloat16, scaling_type=ScalingType.BlockWise1x32
):
    return F.scaled_mm(
        a,
        b,
        sa,
        scaling_type,
        sb,
        scaling_type,
        SwizzleType.NO_SWIZZLE,
        SwizzleType.NO_SWIZZLE,
        output_dtype=out_dtype,
    )


@instantiate_parametrized_tests
class TestFlyDSLMXFPMetadata(TestCase):
    def _args(self, mxfp_format, shape=(64, 96, 256), transposed=(False, True)):
        m, n, k = shape
        dtype = (
            torch.float4_e2m1fn_x2 if mxfp_format == "mxfp4" else torch.float8_e4m3fn
        )
        sk = k // 2 if mxfp_format == "mxfp4" else k

        def node(size, stride, dtype):
            return Buffer(
                name="input",
                layout=FixedLayout(
                    torch.device("cuda", 0), dtype, size, stride, offset=0
                ),
            )

        return dict(
            mat_a=node([m, sk], [1, m] if transposed[0] else [sk, 1], dtype),
            mat_b=node([sk, n], [1, sk] if transposed[1] else [n, 1], dtype),
            scale_a=[node([m, k // 32], [k // 32, 1], torch.float8_e8m0fnu)],
            scale_b=[node([n, k // 32], [k // 32, 1], torch.float8_e8m0fnu)],
            recipe_a=[ScalingType.BlockWise1x32.value],
            recipe_b=[ScalingType.BlockWise1x32.value],
            swizzle_a=[SwizzleType.NO_SWIZZLE.value],
            swizzle_b=[SwizzleType.NO_SWIZZLE.value],
            bias=None,
            out_dtype=torch.bfloat16,
            contraction_dim=[],
            use_fast_accum=False,
        )

    @parametrize("mxfp_format", ("mxfp4", "mxfp8"))
    def test_contract(self, mxfp_format):
        args = self._args(mxfp_format)
        bias = Buffer(
            name="bias",
            layout=FixedLayout(torch.device("cuda", 0), torch.float32, [96], [1]),
        )
        with mock.patch.object(torch.version, "hip", "test"):
            for overrides, expected in (
                ({}, mxfp_format),
                ({"contraction_dim": None}, mxfp_format),
                ({"swizzle_a": [SwizzleType.SWIZZLE_32_4_4.value]}, None),
                ({"recipe_b": [ScalingType.BlockWise1x16.value]}, None),
                ({"out_dtype": torch.float32}, None),
                ({"use_fast_accum": True}, None),
                ({"contraction_dim": [1]}, None),
                ({"bias": object()}, None),
                ({"bias": bias}, mxfp_format),
                ({"scale_a": []}, None),
                ({"mat_b": args["scale_b"][0]}, None),
            ):
                with self.subTest(overrides=overrides):
                    self.assertEqual(
                        mm._get_rocm_mxfp_v2_format(**(args | overrides)), expected
                    )

    @parametrize("mxfp_format", ("mxfp4", "mxfp8"))
    @unittest.skipUnless(flydsl_utils.runtime_available(), "FlyDSL unavailable")
    @inductor_config.patch(flydsl_enable_autotuning=False)
    def test_configs(self, mxfp_format):
        from torch._inductor.sizevars import SizeVarAllocator

        param = _mxfp_param(mxfp_format, (64, 64, 256, 2, 1, 1, 0), 256)
        self.assertEqual(
            param.ldg_x_threads * param.async_load_bytes,
            128 if mxfp_format == "mxfp4" else 256,
        )
        for shape, transposed, expected, unaligned in (
            *(
                ((80, 112, 384), t, True, None)
                for t in ((False, False), (False, True), (True, False), (True, True))
            ),
            ((65, 97, 384), (False, True), False, None),
            ((65, 104, 384), (False, True), True, None),
            ((64, 96, 160), (False, True), False, None),
            ((65, 96, 256), (True, True), False, None),
            ((64, 97, 256), (False, False), False, None),
            ((80, 112, 384), (False, True), False, "scale_a"),
            ((80, 112, 384), (False, True), False, "scale_b"),
        ):
            with self.subTest(shape=shape, transposed=transposed, unaligned=unaligned):
                args = self._args(mxfp_format, shape, transposed)
                bad_node = args[unaligned][0] if unaligned else None
                graph = SimpleNamespace(sizevars=SizeVarAllocator())
                layout = FixedLayout(
                    torch.device("cuda", 0),
                    torch.bfloat16,
                    list(shape[:2]),
                    [shape[1], 1],
                    offset=0,
                )
                with (
                    V.set_graph_handler(graph),
                    mock.patch.object(
                        mm, "use_flydsl_gemm_template", return_value=True
                    ),
                    mock.patch.object(
                        mm, "is_unaligned", side_effect=lambda t: t is bad_node
                    ),
                ):
                    configs = mm.get_flydsl_mxfp_template_kwargs(
                        mxfp_format,
                        layout,
                        args["mat_a"],
                        args["mat_b"],
                        args["scale_a"][0],
                        args["scale_b"][0],
                    )
                self.assertEqual(bool(configs), expected)
                for cfg in configs:
                    self.assertEqual(
                        tuple(cfg[x] for x in ("GEMM_M", "GEMM_N", "GEMM_K")), shape
                    )
                    self.assertEqual(
                        (cfg["A_IS_TRANSPOSED"], cfg["B_IS_TRANSPOSED"]), transposed
                    )

        for tile, error in (
            ((128, 128, 64, 2, 1, 1, 0), "MFMA K depth"),
            ((128, 128, 128, 1, 1, 1, 0), "stages must be at least 2"),
            ((128, 128, 128, 2, 3, 1, 0), "divisible by m_waves/n_waves"),
            ((144, 128, 128, 2, 1, 1, 0), "register budget"),
        ):
            with self.subTest(tile=tile), self.assertRaisesRegex(ValueError, error):
                _mxfp_param(mxfp_format, tile, 2048)


class TestFlyDSLMXFPDevice(TestCase):
    def setUp(self):
        super().setUp()
        if not torch.version.hip or not flydsl_utils.runtime_available():
            self.skipTest("requires ROCm and FlyDSL")
        if torch.cuda.get_device_properties().gcnArchName.split(":")[0] != "gfx950":
            self.skipTest("requires gfx950")
        torch.manual_seed(2026)

    def _check(
        self,
        mxfp_format,
        shape,
        tile,
        out_dtype,
        device,
        transposed=(False, True),
        bias=False,
    ):
        inputs, reference = _mxfp_case(mxfp_format, shape, device, *transposed)
        out = torch.empty(shape[:2], device=device, dtype=out_dtype)
        bias = torch.randn(shape[1], device=device) if bias else None
        param = _mxfp_param(
            mxfp_format,
            tile,
            shape[2],
            out_dtype,
            transposed,
            has_bias=bias is not None,
        )
        _mxfp_call(inputs, out, param, bias)()
        expected = reference if bias is None else reference + bias
        self.assertEqual(out, expected.to(out_dtype), atol=3e-2, rtol=2e-2)

    @parametrize("mxfp_format", ("mxfp4", "mxfp8"))
    @parametrize("out_dtype", (torch.bfloat16, torch.float16))
    @parametrize(
        "transposed", ((False, True), (False, False), (True, True), (True, False))
    )
    def test_scale_pipeline(self, device, mxfp_format, out_dtype, transposed):
        # Short prologue, odd/even double tiles, K tails, and repeated scale-ring wraps.
        for tile_k in (128, 256):
            for k in (256, 384, 512, 640, 1152, 2048):
                if k < 2 * tile_k:
                    continue
                for hti in (False, True):
                    with self.subTest(tile_k=tile_k, k=k, hti=hti):
                        self._check(
                            mxfp_format,
                            (256, 256, k),
                            (128, 128, tile_k, 2, 2, 2, 0, hti),
                            out_dtype,
                            device,
                            transposed,
                        )

    @parametrize("mxfp_format", ("mxfp4", "mxfp8"))
    @parametrize("out_dtype", (torch.bfloat16, torch.float16))
    def test_tiles_and_epilogue(self, device, mxfp_format, out_dtype):
        cases = [
            ((64, 96, 256), (32, 32, 128, 2, 1, 1, 0)),
            ((65, 104, 384), (32, 32, 256, 2, 1, 1, 0)),
            ((65, 104, 640), (32, 32, 512, 2, 1, 1, 0)),
            ((256, 256, 1024), (64, 64, 128, 4, 2, 2, 4)),
            ((256, 256, 512), (256, 256, 128, 2, 2, 2, 0)),
        ]
        for block in (128, 256):
            k = block if mxfp_format == "mxfp4" else 128
            tile = (block, block, k, 2, 2, block // 64, 0, True)
            cases.extend((shape, tile) for shape in ((65, 104, 384), (256, 256, 1152)))
        for shape, tile in cases:
            for bias in (False, True):
                with self.subTest(shape=shape, tile=tile, bias=bias):
                    self._check(mxfp_format, shape, tile, out_dtype, device, bias=bias)

    @parametrize("mxfp_format", ("mxfp4", "mxfp8"))
    @parametrize("out_dtype", (torch.bfloat16, torch.float16))
    def test_dynamic_graph(self, device, mxfp_format, out_dtype):
        import flydsl.compiler as flyc

        tile = (256, 256, 256 if mxfp_format == "mxfp4" else 128, 2, 2, 4, 0, True)
        param = _mxfp_param(mxfp_format, tile, 640, out_dtype, has_bias=True)
        compiler = mock.Mock(wraps=flyc.compile)
        with mock.patch.object(
            kernels.gemm_mxfp_gfx950, "_compiled_cache", {}, create=True
        ):
            for i, shape in enumerate(
                (
                    (256, 512, 512),
                    (65, 104, 640),
                    (512, 256, 1024),
                    (256, 256, 8192),
                    (256, 512, 512),
                )
            ):
                with self.subTest(shape=shape):
                    m, n, _ = shape
                    inputs, reference = _mxfp_case(mxfp_format, shape, device)
                    inputs = [t.view(torch.uint8) for t in inputs]
                    if i % 2:
                        inputs = [
                            torch.empty_strided(
                                t.shape,
                                (1, t.stride(1) + 16)
                                if j == 1
                                else (t.stride(0) + 16, 1),
                                device=device,
                                dtype=t.dtype,
                            ).copy_(t)
                            for j, t in enumerate(inputs)
                        ]
                    stride = n + 8 * (i % 2)
                    arena = torch.full(
                        (m * stride + 32,), -317, device=device, dtype=out_dtype
                    )
                    out = arena[16:-16].view(m, stride)[:, :n]
                    bias = torch.randn(n, device=device)
                    call = _mxfp_call(inputs, out, param, bias, compiler)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        call()
                    for factor in (1, 2):
                        if factor == 2:
                            inputs[2].add_(1)
                        out.fill_(float("nan"))
                        graph.replay()
                        self.assertEqual(
                            out,
                            (reference * factor + bias).to(out_dtype),
                            atol=3e-2,
                            rtol=2e-2,
                        )
                        for guard in (
                            arena[:16],
                            arena[-16:],
                            arena[16:-16].view(m, stride)[:, n:],
                        ):
                            self.assertEqual(guard, torch.full_like(guard, -317))
        self.assertEqual(compiler.call_count, 1)

    @parametrize("mxfp_format", ("mxfp4", "mxfp8", None))
    def test_compiled_routes(self, device, mxfp_format):
        if mxfp_format is None:
            a = torch.randn(64, 128, device=device).to(torch.float8_e4m3fn)
            b = torch.randn(64, 128, device=device).to(torch.float8_e4m3fn).t()
            scale = torch.ones((), device=device)
            inputs = (a, b, scale, scale, torch.bfloat16, ScalingType.TensorWise)
            reference = _scaled_mm_mxfp(*inputs)
        else:
            inputs, reference = _mxfp_case(mxfp_format, (64, 96, 256), device)
            eager = mm._scaled_mm_v2_mxfp(*inputs, out_dtype=torch.bfloat16)
            self.assertEqual(eager, reference.bfloat16(), atol=3e-2, rtol=2e-2)
            inputs, reference = _mxfp_case(mxfp_format, (64, 4096, 4096), device)
        with inductor_config.patch(
            max_autotune=True,
            flydsl_enable_autotuning=False,
            max_autotune_gemm_backends="FLYDSL" if mxfp_format else "ATEN,FLYDSL",
        ):
            torch._dynamo.reset()
            actual, code = run_and_get_code(
                torch.compile(_scaled_mm_mxfp, fullgraph=True), *inputs
            )
        if mxfp_format is None:
            self.assertEqual(actual, reference)
        self.assertEqual(actual, reference.bfloat16(), atol=3e-2, rtol=2e-2)
        self.assertEqual(
            "async_compile.flydsl" in "\n".join(code), mxfp_format is not None
        )


instantiate_device_type_tests(TestFlyDSLMXFPDevice, globals(), only_for="cuda")

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
