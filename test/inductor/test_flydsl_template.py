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

    @parametrize("raises", (False, True))
    def test_precompile_flydsl_restores_environment(self, raises):
        from torch._inductor.runtime.flydsl_cache import precompile_flydsl
        from torch._subclasses.fake_tensor import FakeTensor

        seen = []

        def kernel(*, mat1, output, stream, compile_only):
            self.assertIsInstance(mat1, FakeTensor)
            self.assertEqual(mat1.shape, (2, 3))
            self.assertEqual(mat1.stride(), (4, 1))
            self.assertEqual(output.dtype, torch.bfloat16)
            self.assertTrue(compile_only)
            self.assertEqual(stream, 0)
            self.assertEqual(os.environ["COMPILE_ONLY"], "1")
            self.assertEqual(os.environ["FLYDSL_GPU_ARCH"], "gfx950")
            seen.append(True)
            if raises:
                raise RuntimeError("compile failed")

        with mock.patch.dict(
            os.environ, {"COMPILE_ONLY": "0", "FLYDSL_GPU_ARCH": "gfx942"}
        ):

            def precompile():
                precompile_flydsl(
                    kernel,
                    {"mat1": (2, 3), "output": (2, 5)},
                    {"mat1": (4, 1), "output": (5, 1)},
                    {"mat1": "float32", "output": "bfloat16"},
                    "gfx950",
                )

            if raises:
                with self.assertRaisesRegex(RuntimeError, "compile failed"):
                    precompile()
            else:
                precompile()
            self.assertEqual(os.environ["COMPILE_ONLY"], "0")
            self.assertEqual(os.environ["FLYDSL_GPU_ARCH"], "gfx942")
        self.assertEqual(seen, [True])

    @parametrize("kind", ("mm", "grouped_mm", "mxfp8_grouped_mm"))
    def test_flydsl_precompile_fake_tensors(self, kind):
        if not flydsl_utils.runtime_available():
            self.skipTest("FlyDSL runtime unavailable")
        import flydsl.compiler as flyc
        import flydsl.expr as fx
        import jinja2

        from torch._inductor.kernel.mm_common import load_kernel_template

        namespace = dict(
            torch=torch,
            flyc=flyc,
            fx=fx,
            GEMM_M=256,
            GEMM_N=256,
            GEMM_K=512,
            GEMM_G=2,
            GEMM_DTYPE_ID=2,
            TILE_M=128,
            TILE_N=128,
            TILE_K=64,
            STAGES=2,
            M_WAVES=1,
            N_WAVES=4,
            GROUP_M=0,
            USE_HALF_TILE_INTERLEAVED=False,
            A_IS_TRANSPOSED=False,
            B_IS_TRANSPOSED=True,
            BLOCK_R=64,
            BLOCK_C=128,
        )
        source = jinja2.Template(load_kernel_template(f"flydsl_{kind}")).render(
            gen_defines=lambda: "",
            def_kernel=lambda *names: f"def kernel_main({','.join(names)}, output, stream):",
            get_output=lambda: "output",
            kernel_name="kernel",
        )
        exec(compile(source, "<flydsl precompile test>", "exec"), namespace)
        shapes = {"mat1": (256, 512), "mat2": (512, 256), "output": (256, 256)}
        strides = {"mat1": (512, 1), "mat2": (1, 512), "output": (256, 1)}
        dtypes = dict.fromkeys(shapes, "bfloat16")
        if kind != "mm":
            shapes.update(mat2=(2, 512, 256), offs=(2,))
            strides.update(mat2=(131072, 256, 1), offs=(1,))
            dtypes["offs"] = "int32"
        if kind == "mxfp8_grouped_mm":
            strides["mat2"] = (131072, 1, 512)
            dtypes.update(
                mat1="float8_e4m3fn",
                mat2="float8_e4m3fn",
                scale_a="float8_e8m0fnu",
                scale_b="float8_e8m0fnu",
            )
            shapes.update(scale_a=(256, 16), scale_b=(2, 4096))
            strides.update(scale_a=(16, 1), scale_b=(4096, 1))
        # Exercise the real compiler with metadata only, including on hosts
        # without a GPU. A normal flyc.compile call attempts to execute.
        with mock.patch.object(
            torch.cuda,
            "get_device_properties",
            side_effect=AssertionError("device queried"),
        ):
            namespace["kernel_precompile"](
                shapes, strides, dtypes, flydsl_gpu_arch="gfx950"
            )

    @parametrize("block_m", (2, 4, 8))
    def test_grouped_row_tile_upper_bound(self, block_m):
        from itertools import product

        from torch._inductor.kernel.vendored_templates.flydsl.kernels.grouped_config import (
            grouped_row_tiles_upper_bound,
        )

        for total in range(9):
            for groups in range(1, 5):
                actual_max = max(
                    sum((rows + block_m - 1) // block_m for rows in partition)
                    for partition in product(range(total + 1), repeat=groups)
                    if sum(partition) == total
                )
                self.assertEqual(
                    grouped_row_tiles_upper_bound(total, groups, block_m),
                    actual_max,
                )

    def test_precompile_flydsl_concurrent_dispatch(self):
        from torch._inductor.runtime.flydsl_cache import precompile_flydsl

        precompile_started = threading.Event()
        release_precompile = threading.Event()
        cold_started = threading.Event()
        cold_compiler_started = threading.Event()
        warm_dispatch = mock.Mock()
        cold_jit = SimpleNamespace()

        def precompile_kernel(**kwargs):
            precompile_started.set()
            self.assertTrue(release_precompile.wait(5))

        def cold_compiler(*args):
            cold_compiler_started.set()
            self.assertNotEqual(os.environ.get("COMPILE_ONLY"), "1")
            return mock.Mock()

        def cold_launch():
            cold_started.set()
            return run_cached_flydsl(
                cold_jit,
                constexpr_param=_CacheParam(),
                compiler=cold_compiler,
                dispatch_args=(dispatch,),
            )

        # Supply a dispatch argument because the cache keys include its device.
        dispatch = SimpleNamespace(device=SimpleNamespace(index=0))
        warm_jit = SimpleNamespace()
        run_cached_flydsl(
            warm_jit,
            constexpr_param=_CacheParam(),
            compiler=lambda *args: warm_dispatch,
            dispatch_args=(dispatch,),
        )
        with ThreadPoolExecutor(max_workers=2) as pool:
            precompile = pool.submit(precompile_flydsl, precompile_kernel, {}, {}, {})
            try:
                self.assertTrue(precompile_started.wait(5))
                run_cached_flydsl(
                    warm_jit,
                    constexpr_param=_CacheParam(),
                    compiler=mock.Mock(
                        side_effect=AssertionError("unexpected compile")
                    ),
                    dispatch_args=(dispatch,),
                )
                warm_dispatch.assert_called_once_with(dispatch)
                cold = pool.submit(cold_launch)
                self.assertTrue(cold_started.wait(5))
                self.assertFalse(cold_compiler_started.wait(0.05))
            finally:
                release_precompile.set()
            precompile.result()
            cold.result()

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
                self.assertIn("flydsl_tensor_arg", code)
                self.assertNotIn("mat2.transpose(0, 1)", code)
                self.assertIn("flydsl_tensor_arg(mat2)", code)
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
        self.assertIn("precompile_flydsl", code)
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
        self.assertIn("flydsl_tensor_arg", code)
        self.assertIn("_precompile", code)
        self.assertIn("precompile_flydsl", code)
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
