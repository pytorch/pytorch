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
        layout = SimpleNamespace(
            size=[64, n],
            stride=[n, 1],
            dtype=dtype,
            device=torch.device("cpu"),
        )
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
            size=[m, n],
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
            self.assertIs(result[0]["IS_MXFP"], False)
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
                self.assertIn("_inductor_tensor_arg(mat2)", code)
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


def _make_mxfp4_operand(rows, k, device):
    codes = torch.arange(rows * k, device=device, dtype=torch.int64).reshape(rows, k)
    codes = (codes * 5 + 3) % 16
    lut = torch.tensor(E2M1_MAGNITUDES, device=device, dtype=torch.float32)
    values = torch.where(codes >= 8, -lut[(codes & 7).long()], lut[(codes & 7).long()])
    exponents = (
        torch.arange(rows * (k // 32), device=device, dtype=torch.int64)
        .reshape(rows, k // 32)
        .remainder(7)
        .add(124)
        .to(torch.uint8)
    )
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    scales = torch.pow(2.0, exponents.float() - 127.0)
    return (
        packed.contiguous().view(torch.float4_e2m1fn_x2),
        exponents.contiguous().view(torch.float8_e8m0fnu),
        values * scales.repeat_interleave(32, dim=1),
    )


def _make_mxfp8_operand(rows, k, device):
    values = (
        (torch.arange(rows * k, device=device).reshape(rows, k) * 5 + 2) % 11 - 5
    ) / 2
    operand = values.to(torch.float8_e4m3fn)
    exponents = (
        torch.arange(rows * (k // 32), device=device, dtype=torch.int64)
        .reshape(rows, k // 32)
        .remainder(7)
        .add(124)
        .to(torch.uint8)
    )
    scale = exponents.contiguous().view(torch.float8_e8m0fnu)
    reference = operand.float() * torch.pow(
        2.0, exponents.float() - 127.0
    ).repeat_interleave(32, dim=1)
    return operand, scale, reference


def _make_mxfp_operand(mxfp_format, rows, k, device):
    if mxfp_format == "mxfp4":
        return _make_mxfp4_operand(rows, k, device)
    if mxfp_format == "mxfp8":
        return _make_mxfp8_operand(rows, k, device)
    raise AssertionError(f"unsupported MXFP format: {mxfp_format}")


def _scaled_mm_mxfp(a, b, scale_a, scale_b, out_dtype):
    recipe = [ScalingType.BlockWise1x32.value]
    swizzle = [SwizzleType.NO_SWIZZLE.value]
    return torch._scaled_mm_v2(
        a,
        b,
        [scale_a],
        recipe,
        swizzle,
        [scale_b],
        recipe,
        swizzle,
        None,
        out_dtype,
        [],
        False,
    )


def _candidate_args(mxfp_format, m=64, n=96, k=256, **overrides):
    dtype, elements_per_byte = (
        (torch.float4_e2m1fn_x2, 2)
        if mxfp_format == "mxfp4"
        else (torch.float8_e4m3fn, 1)
    )
    storage_k = k // elements_per_byte
    args = {
        "mat_a": _FakeNode((m, storage_k), (storage_k, 1), dtype),
        "mat_b": _FakeNode((storage_k, n), (1, storage_k), dtype),
        "scale_a": [_FakeNode((m, k // 32), (k // 32, 1), torch.float8_e8m0fnu)],
        "recipe_a": [ScalingType.BlockWise1x32.value],
        "swizzle_a": [SwizzleType.NO_SWIZZLE.value],
        "scale_b": [_FakeNode((n, k // 32), (k // 32, 1), torch.float8_e8m0fnu)],
        "recipe_b": [ScalingType.BlockWise1x32.value],
        "swizzle_b": [SwizzleType.NO_SWIZZLE.value],
        "bias": None,
        "out_dtype": torch.bfloat16,
        "contraction_dim": [],
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
        gemm_mxfp_gfx950,
        make_mxfp_param_and_validate,
    )

    m, n, k = shape
    block_m, block_n, block_k, stages, m_waves, n_waves, group_m, lds_scale = tile
    out = torch.zeros(m, n, device=a.device, dtype=out_dtype)
    tensors = (
        out,
        a.view(torch.uint8),
        b.view(torch.uint8),
        scale_a.view(torch.uint8),
        scale_b.view(torch.uint8),
    )
    param = make_mxfp_param_and_validate(
        mxfp_format,
        m,
        n,
        k,
        "bfloat16" if out_dtype == torch.bfloat16 else "float16",
        {
            "TILE_M": block_m,
            "TILE_N": block_n,
            "TILE_K": block_k,
            "STAGES": stages,
            "M_WAVES": m_waves,
            "N_WAVES": n_waves,
            "GROUP_M": group_m,
            "LDS_SCALE": lds_scale,
        },
        a_is_transposed=a_is_transposed,
        b_is_transposed=b_is_transposed,
    )
    assert param is not None
    compile_args = tuple(
        flyc.from_torch_tensor(tensor).mark_layout_dynamic() for tensor in tensors
    ) + (param, 0)
    compiled = flyc.compile(gemm_mxfp_gfx950, *compile_args)
    compiled(*tensors, param, 0)
    torch.cuda.synchronize()
    return out


def _with_outer_contiguous_storage(tensor):
    return tensor.view(torch.uint8).t().contiguous().t().view(tensor.dtype)


def _mxfp_operand_layouts(a, b, a_is_transposed, b_is_transposed):
    a_arg = _with_outer_contiguous_storage(a) if a_is_transposed else a
    b_nk = b if b_is_transposed else _with_outer_contiguous_storage(b)
    return a_arg, b_nk.view(torch.uint8).t().view(b.dtype), b_nk


class TestFlyDSLMXFPMetadata(TestCase):
    @parametrize(
        "mxfp_format,contraction_dim",
        (("mxfp4", None), ("mxfp8", None), ("mxfp8", [])),
    )
    def test_supported_contract(self, mxfp_format, contraction_dim):
        with mock.patch.object(torch.version, "hip", "test"):
            self.assertEqual(
                mm._get_rocm_mxfp_v2_format(
                    **_candidate_args(mxfp_format, contraction_dim=contraction_dim)
                ),
                mxfp_format,
            )

    @parametrize(
        "mxfp_format,override",
        (
            ("mxfp4", {"swizzle_a": [SwizzleType.SWIZZLE_32_4_4.value]}),
            ("mxfp4", {"recipe_b": [ScalingType.BlockWise1x16.value]}),
            ("mxfp4", {"out_dtype": torch.float32}),
            ("mxfp4", {"use_fast_accum": True}),
            ("mxfp4", {"contraction_dim": [1]}),
            ("mxfp8", {"bias": object()}),
            ("mxfp8", {"scale_a": []}),
            ("mxfp8", {"mat_b": _FakeNode((256, 96), (1, 256), torch.float16)}),
        ),
    )
    def test_rejected_contract(self, mxfp_format, override):
        with mock.patch.object(torch.version, "hip", "test"):
            self.assertIsNone(
                mm._get_rocm_mxfp_v2_format(
                    **_candidate_args(mxfp_format, **override)
                )
            )

    @parametrize("mxfp_format,block_k_bytes", (("mxfp4", 128), ("mxfp8", 256)))
    @unittest.skipUnless(flydsl_utils.runtime_available(), "FlyDSL unavailable")
    def test_tile_storage_units(self, mxfp_format, block_k_bytes):
        from torch._inductor.kernel.vendored_templates.flydsl.kernels import (
            mxfp_gemm_derived,
        )

        derived = mxfp_gemm_derived(mxfp_format, 64, 64, 256, 2, 1, 1)
        self.assertEqual(derived.block_k_bytes, block_k_bytes)
        self.assertEqual(derived.k_halves, 2)

    @unittest.skipUnless(flydsl_utils.runtime_available(), "FlyDSL unavailable")
    def test_invalid_tile_reports_reason(self):
        from torch._inductor.kernel.vendored_templates.flydsl.kernels import (
            mxfp_gemm_derived,
        )

        with self.assertRaisesRegex(ValueError, "multiple of the MFMA K depth"):
            mxfp_gemm_derived("mxfp4", 128, 128, 64, 2, 1, 1)

    @parametrize("mxfp_format", ("mxfp4", "mxfp8"))
    @parametrize(
        "a_is_transposed,b_is_transposed",
        ((False, False), (False, True), (True, False), (True, True)),
    )
    @config.patch(flydsl_enable_autotuning=False)
    @unittest.skipUnless(flydsl_utils.runtime_available(), "FlyDSL unavailable")
    def test_layouts_and_config_filtering(
        self, mxfp_format, a_is_transposed, b_is_transposed
    ):
        args = _candidate_args(mxfp_format, m=80, n=112, k=384)
        m, storage_k = args["mat_a"].get_size()
        _, n = args["mat_b"].get_size()
        dtype = args["mat_a"].get_dtype()
        a = _FakeNode(
            (m, storage_k),
            (1, m) if a_is_transposed else (storage_k, 1),
            dtype,
        )
        b = _FakeNode(
            (storage_k, n),
            (1, storage_k) if b_is_transposed else (n, 1),
            dtype,
        )
        layout = SimpleNamespace(
            size=[m, n],
            stride=[n, 1],
            dtype=torch.bfloat16,
            device=torch.device("cuda", 0),
            offset=0,
        )
        graph = SimpleNamespace(
            sizevars=SimpleNamespace(
                statically_known_multiple_of=lambda value, multiple: (
                    value % multiple == 0
                )
            )
        )
        with V.set_graph_handler(graph), mock.patch.object(
            mm, "use_flydsl_gemm_template", return_value=True
        ):
            configs = mm.get_flydsl_mxfp_template_kwargs(
                mxfp_format, layout, a, b, args["scale_a"][0], args["scale_b"][0]
            )
        self.assertTrue(configs)
        self.assertTrue(all(config_["GEMM_M"] == m for config_ in configs))
        self.assertTrue(all(config_["GEMM_N"] == n for config_ in configs))
        self.assertTrue(all(config_["GEMM_K"] == 384 for config_ in configs))
        self.assertTrue(
            all(
                config_["A_IS_TRANSPOSED"] == a_is_transposed
                for config_ in configs
            )
        )
        self.assertTrue(
            all(
                config_["B_IS_TRANSPOSED"] == b_is_transposed
                for config_ in configs
            )
        )

    @parametrize("mxfp_format", ("mxfp4", "mxfp8"))
    @config.patch(flydsl_enable_autotuning=False)
    @unittest.skipUnless(flydsl_utils.runtime_available(), "FlyDSL unavailable")
    def test_row_major_mn_tail_has_config(self, mxfp_format):
        args = _candidate_args(mxfp_format, m=65, n=97, k=384)
        layout = SimpleNamespace(
            size=[65, 97],
            stride=[97, 1],
            dtype=torch.bfloat16,
            device=torch.device("cuda", 0),
            offset=0,
        )
        graph = SimpleNamespace(
            sizevars=SimpleNamespace(
                statically_known_multiple_of=lambda value, multiple: (
                    value % multiple == 0
                )
            )
        )
        with V.set_graph_handler(graph), mock.patch.object(
            mm, "use_flydsl_gemm_template", return_value=True
        ):
            configs = mm.get_flydsl_mxfp_template_kwargs(
                mxfp_format,
                layout,
                args["mat_a"],
                args["mat_b"],
                args["scale_a"][0],
                args["scale_b"][0],
            )
        self.assertTrue(configs)

    @parametrize("mxfp_format", ("mxfp4", "mxfp8"))
    @parametrize(
        "m,n,k,a_is_transposed,b_is_transposed",
        (
            (64, 96, 160, False, True),
            (65, 96, 256, True, True),
            (64, 97, 256, False, False),
        ),
    )
    @config.patch(flydsl_enable_autotuning=False)
    @unittest.skipUnless(flydsl_utils.runtime_available(), "FlyDSL unavailable")
    def test_metadata_rejects_unsupported_shape_layout(
        self,
        mxfp_format,
        m,
        n,
        k,
        a_is_transposed,
        b_is_transposed,
    ):
        args = _candidate_args(mxfp_format, m=m, n=n, k=k)
        _, storage_k = args["mat_a"].get_size()
        dtype = args["mat_a"].get_dtype()
        a = _FakeNode(
            (m, storage_k),
            (1, m) if a_is_transposed else (storage_k, 1),
            dtype,
        )
        b = _FakeNode(
            (storage_k, n),
            (1, storage_k) if b_is_transposed else (n, 1),
            dtype,
        )
        layout = SimpleNamespace(
            size=[m, n],
            stride=[n, 1],
            dtype=torch.bfloat16,
            device=torch.device("cuda", 0),
            offset=0,
        )
        graph = SimpleNamespace(
            sizevars=SimpleNamespace(
                statically_known_multiple_of=lambda value, multiple: (
                    value % multiple == 0
                )
            )
        )
        with V.set_graph_handler(graph), mock.patch.object(
            mm, "use_flydsl_gemm_template", return_value=True
        ):
            configs = mm.get_flydsl_mxfp_template_kwargs(
                mxfp_format,
                layout,
                a,
                b,
                args["scale_a"][0],
                args["scale_b"][0],
            )
        self.assertEqual(configs, [])

    @parametrize("mxfp_format", ("mxfp4", "mxfp8"))
    @parametrize("k,tile_k", ((384, 256), (640, 512)))
    @unittest.skipUnless(flydsl_utils.runtime_available(), "FlyDSL unavailable")
    def test_runtime_block_k_tail(self, mxfp_format, k, tile_k):
        from torch._inductor.kernel.vendored_templates.flydsl.kernels import (
            make_mxfp_param_and_validate,
        )

        config_ = {
            "TILE_M": 32,
            "TILE_N": 32,
            "TILE_K": tile_k,
            "STAGES": 2,
            "M_WAVES": 1,
            "N_WAVES": 1,
            "GROUP_M": 0,
            "LDS_SCALE": 0,
        }
        param = make_mxfp_param_and_validate(
            mxfp_format, 65, 97, k, "bfloat16", config_
        )
        self.assertIsNotNone(param)
        self.assertTrue(param.has_k_tail)
        self.assertIsNone(
            make_mxfp_param_and_validate(
                mxfp_format, 65, 97, 160, "bfloat16", config_
            )
        )

    def test_mxfp_precompile_metadata(self):
        layout = FixedLayout(torch.device("cpu"), torch.uint8, [1], [1])
        inputs = [
            Buffer(name=name, layout=layout)
            for name in ("mat1", "mat2", "scale_a", "scale_b")
        ]
        output = Buffer(name="output", layout=layout)
        kernel = FlyDSLTemplateKernel(
            kernel_name="mxfp",
            input_nodes=inputs,
            output_node=output,
        )
        graph = SimpleNamespace(removed_buffers=OrderedSet(), scheduler=None)
        with V.set_graph_handler(graph), mock.patch(
            "torch.cuda.is_available", return_value=False
        ):
            kernel.def_kernel("mat1", "mat2", "scale_a", "scale_b")
            metadata = FlyDSLScheduling(None)._build_precompile_metadata(
                kernel, SimpleNamespace(layout=layout)
            )
        self.assertEqual(
            set(metadata["precompile_shapes"]),
            {"mat1", "mat2", "scale_a", "scale_b", "output"},
        )


class TestFlyDSLMXFPDevice(TestCase):
    def _skip_unless_supported(self, device):
        if torch.version.hip is None:
            self.skipTest("requires ROCm")
        arch = torch.cuda.get_device_properties(device).gcnArchName.split(":", 1)[0]
        if arch != "gfx950":
            self.skipTest(f"requires gfx950, got {arch}")
        if not flydsl_utils.runtime_available():
            self.skipTest("FlyDSL runtime unavailable")

    def _assert_close(self, actual, reference, out_dtype):
        self.assertEqual(actual, reference.to(out_dtype), rtol=2e-2, atol=5e-1)

    @parametrize("mxfp_format", ("mxfp4", "mxfp8"))
    @parametrize(
        "a_is_transposed,b_is_transposed",
        ((False, False), (False, True), (True, False), (True, True)),
    )
    def test_all_operand_layouts(
        self, device, mxfp_format, a_is_transposed, b_is_transposed
    ):
        self._skip_unless_supported(device)
        m, n, k = 64, 96, 256
        a, scale_a, a_ref = _make_mxfp_operand(mxfp_format, m, k, device)
        b, scale_b, b_ref = _make_mxfp_operand(mxfp_format, n, k, device)
        a_arg, _, b_nk = _mxfp_operand_layouts(
            a, b, a_is_transposed, b_is_transposed
        )
        actual = _run_mxfp_tile(
            mxfp_format,
            (m, n, k),
            (32, 32, 128, 2, 1, 1, 0, 0),
            torch.bfloat16,
            a_arg,
            b_nk,
            scale_a,
            scale_b,
            a_is_transposed=a_is_transposed,
            b_is_transposed=b_is_transposed,
        )
        self._assert_close(actual, a_ref @ b_ref.t(), torch.bfloat16)

    @parametrize("mxfp_format", ("mxfp4", "mxfp8"))
    def test_compiled_flydsl_route(self, device, mxfp_format):
        self._skip_unless_supported(device)
        m, n, k = 64, 4096, 4096
        a, scale_a, a_ref = _make_mxfp_operand(mxfp_format, m, k, device)
        b, scale_b, b_ref = _make_mxfp_operand(mxfp_format, n, k, device)
        b_t = b.view(torch.uint8).t().view(b.dtype)
        with config.patch(
            max_autotune=True,
            max_autotune_gemm_backends="FLYDSL",
            flydsl_enable_autotuning=False,
        ):
            torch._dynamo.reset()
            compiled = torch.compile(
                _scaled_mm_mxfp, backend="inductor", fullgraph=True
            )
            actual, code = run_and_get_code(
                compiled, a, b_t, scale_a, scale_b, torch.bfloat16
            )
        self._assert_close(actual, a_ref @ b_ref.t(), torch.bfloat16)
        self.assertIn("async_compile.flydsl", "\n".join(code))

    @parametrize(
        "mxfp_format,shape,tile,out_dtype",
        (
            ("mxfp8", (64, 96, 256), (32, 32, 128, 2, 1, 1, 0, 0), torch.bfloat16),
            ("mxfp8", (128, 128, 512), (128, 128, 256, 2, 2, 2, 0, 1), torch.bfloat16),
            ("mxfp8", (256, 256, 1024), (128, 128, 128, 4, 2, 2, 4, 0), torch.bfloat16),
            ("mxfp8", (256, 256, 512), (256, 256, 128, 2, 2, 2, 0, 0), torch.float16),
            ("mxfp4", (32, 32, 256), (16, 16, 128, 2, 1, 1, 0, 0), torch.bfloat16),
            ("mxfp4", (128, 128, 512), (128, 128, 256, 2, 2, 2, 0, 1), torch.bfloat16),
            ("mxfp4", (256, 256, 1024), (64, 64, 128, 4, 2, 2, 4, 0), torch.bfloat16),
            ("mxfp4", (256, 256, 512), (256, 256, 256, 2, 4, 2, 0, 0), torch.float16),
            ("mxfp8", (65, 97, 384), (32, 32, 256, 2, 1, 1, 0, 0), torch.bfloat16),
            ("mxfp4", (65, 97, 384), (32, 32, 256, 2, 1, 1, 0, 0), torch.bfloat16),
        ),
    )
    def test_tile_paths_match_reference(
        self, device, mxfp_format, shape, tile, out_dtype
    ):
        self._skip_unless_supported(device)
        m, n, k = shape
        a, scale_a, a_ref = _make_mxfp_operand(mxfp_format, m, k, device)
        b, scale_b, b_ref = _make_mxfp_operand(mxfp_format, n, k, device)
        actual = _run_mxfp_tile(
            mxfp_format, shape, tile, out_dtype, a, b, scale_a, scale_b
        )
        self._assert_close(actual, a_ref @ b_ref.t(), out_dtype)

    @parametrize("mxfp_format", ("mxfp4", "mxfp8"))
    def test_eager_candidate(self, device, mxfp_format):
        self._skip_unless_supported(device)
        m, n, k = 64, 96, 256
        a, scale_a, a_ref = _make_mxfp_operand(mxfp_format, m, k, device)
        b, scale_b, b_ref = _make_mxfp_operand(mxfp_format, n, k, device)
        b_t = b.view(torch.uint8).t().view(b.dtype)
        actual = mm._scaled_mm_v2_mxfp(
            a, b_t, scale_a, scale_b, out_dtype=torch.bfloat16
        )
        self._assert_close(actual, a_ref @ b_ref.t(), torch.bfloat16)

    def test_unsupported_signature_falls_back(self, device):
        self._skip_unless_supported(device)
        m, n, k = 64, 64, 128
        a = torch.randn(m, k, device=device).to(torch.float8_e4m3fn)
        b = torch.randn(n, k, device=device).to(torch.float8_e4m3fn).t()
        scale_a = torch.ones((), device=device)
        scale_b = torch.ones((), device=device)

        def tensorwise(a, b, scale_a, scale_b):
            recipe = [ScalingType.TensorWise.value]
            swizzle = [SwizzleType.NO_SWIZZLE.value]
            return torch._scaled_mm_v2(
                a,
                b,
                [scale_a],
                recipe,
                swizzle,
                [scale_b],
                recipe,
                swizzle,
                None,
                torch.bfloat16,
                [],
                False,
            )

        with config.patch(max_autotune=True, max_autotune_gemm_backends="ATEN,FLYDSL"):
            _, code = run_and_get_code(
                torch.compile(tensorwise, dynamic=False), a, b, scale_a, scale_b
            )
        self.assertNotIn("async_compile.flydsl", "\n".join(code))


instantiate_parametrized_tests(TestFlyDSLMXFPMetadata)
instantiate_device_type_tests(TestFlyDSLMXFPDevice, globals(), only_for="cuda")

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
