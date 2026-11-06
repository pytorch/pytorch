# Owner(s): ["module: inductor"]
from __future__ import annotations

import importlib.util
import os
import unittest
from unittest.mock import patch

import torch
from torch._inductor import config
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    onlyCUDA,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    parametrize,
    run_tests,
    TestCase,
)
from torch.testing._internal.inductor_utils import HAS_CUDA_AND_TRITON


HAS_HELION = importlib.util.find_spec("helion") is not None
HAS_CUTEDSL = importlib.util.find_spec("cutlass") is not None


@unittest.skipUnless(
    HAS_CUDA_AND_TRITON and HAS_HELION, "requires CUDA, Triton and Helion"
)
class TestNativeMatmulHelion(TestCase):
    hw_classification = HardwareClassification.CUDA
    helion_backend = "triton"

    def setUp(self):
        super().setUp()
        torch._dynamo.reset()
        # Keep strict FP32 checks independent of Helion's environment default.
        self.enterContext(patch.dict(os.environ, {"TRITON_F32_DEFAULT": "ieee"}))
        self.enterContext(
            config.patch(
                {
                    "native_matmul_helion": True,
                    "helion_backend": self.helion_backend,
                    "helion_autotune_effort": "none",
                    "force_disable_caches": True,
                    "triton.native_matmul": False,
                    "benchmark_fusion": False,
                    "max_autotune": False,
                    "triton.cudagraphs": False,
                }
            )
        )

    def check(self, fn, args, *, expected=None):
        if expected is None:
            expected = fn(*args)
        result, sources = run_and_get_code(torch.compile(fn, fullgraph=True), *args)
        self.assertEqual(result, expected)
        source = "\n".join(sources)
        self.assertIn("PyCodeCache.load(", source)
        self.assertIn('_source = """', source)
        self.assertIn("\n@helion.kernel(", source)
        self.assertIn("PyCodeCache.load(helion_", source)
        self.assertIn("@helion.kernel(", source)
        self.assertNotIn("tl.dot(", source)
        self.assertNotIn("extern_kernels.mm(", source)
        return source

    @onlyCUDA
    @dtypes(torch.float16, torch.bfloat16, torch.float32)
    @parametrize("batched", [False, True])
    def test_fused_matmul(self, device, dtype, batched):
        batch = (3,) if batched else ()
        args = [
            torch.randn(*batch, 40, 72, device=device, dtype=dtype),
            torch.randn(*batch, 72, 136, device=device, dtype=dtype),
            torch.randn(136, device=device, dtype=dtype),
        ]

        def fn(a, b, bias):
            return (
                torch.relu(torch.bmm(a, b) + bias)
                if batched
                else torch.relu(a @ b + bias)
            )

        # Native fusion keeps the accumulators in FP32 through the epilogue.
        expected = fn(*(arg.float() for arg in args)).to(dtype)
        source = self.check(fn, args, expected=expected)
        self.assertIn("torch.baddbmm(" if batched else "torch.addmm(", source)

    @onlyCUDA
    @dtypes(torch.bfloat16)
    @parametrize("backend", ["triton", "cutedsl"])
    def test_mixed_wrapper(self, device, dtype, backend):
        def fn(a, b):
            result = torch.relu((a + 1) @ (b - 2))
            return result, torch.sin(a), result.sum(dim=-1)

        args = [
            torch.randn(40, 72, device=device, dtype=dtype),
            torch.randn(72, 136, device=device, dtype=dtype),
        ]
        if backend == "cutedsl" and not HAS_CUTEDSL:
            self.skipTest("requires CuTeDSL")
        with config.patch(helion_backend=backend):
            source = self.check(fn, args)
        self.assertIn("@triton.jit", source)
        self.assertIn(".run(", source)

    @onlyCUDA
    @dtypes(torch.bfloat16)
    @parametrize("backend", ["triton", "cutedsl"])
    def test_backend_selection(self, device, dtype, backend):
        args = [
            torch.randn(32, 64, device=device, dtype=dtype),
            torch.randn(64, 48, device=device, dtype=dtype),
        ]
        if backend == "cutedsl" and not HAS_CUTEDSL:
            self.skipTest("requires CuTeDSL")
        with config.patch(helion_backend=backend):
            source = self.check(lambda a, b: a @ b, args)
        helion_backend = "cute" if backend == "cutedsl" else "triton"
        self.assertIn(f'backend="{helion_backend}"', source)
        self.assertEqual(config.cuda_backend, "triton")

    @onlyCUDA
    @dtypes(torch.bfloat16)
    def test_invalid_backend(self, device, dtype):
        args = [
            torch.randn(32, 64, device=device, dtype=dtype),
            torch.randn(64, 48, device=device, dtype=dtype),
        ]
        with config.patch(helion_backend="invalid"):
            with self.assertRaisesRegex(
                Exception, "Unsupported Helion native matmul backend: invalid"
            ):
                torch.compile(lambda a, b: a @ b, fullgraph=True)(*args)

    @onlyCUDA
    @dtypes(torch.float32)
    @parametrize("precision", ["ieee", "tf32"])
    def test_precision_environment(self, device, dtype, precision):
        from torch._inductor.codecache import PyCodeCache
        from torch._inductor.runtime.helion import HelionKernelWrapper

        settings = []
        original_load = PyCodeCache.load

        def capture_load(source, *args, **kwargs):
            module = original_load(source, *args, **kwargs)
            settings.extend(
                value.kernel.settings
                for value in vars(module).values()
                if isinstance(value, HelionKernelWrapper)
            )
            return module

        args = [
            torch.full((16, 16), 0.5, device=device, dtype=dtype),
            torch.full((16, 16), 0.125, device=device, dtype=dtype),
        ]
        with (
            patch.dict(os.environ, {"TRITON_F32_DEFAULT": precision}),
            patch.dict(PyCodeCache.modules_no_attr, clear=True),
            patch.object(PyCodeCache, "load", capture_load),
        ):
            source = self.check(lambda a, b: a @ b, args)
        self.assertEqual(len(settings), 1)
        self.assertEqual(settings[0].dot_precision, precision)
        self.assertNotIn("dot_precision", source)

    @onlyCUDA
    @dtypes(torch.bfloat16)
    def test_invalid_autotune_effort(self, device, dtype):
        args = [
            torch.randn(32, 64, device=device, dtype=dtype),
            torch.randn(64, 48, device=device, dtype=dtype),
        ]
        with config.patch(helion_autotune_effort="invalid"):
            with self.assertRaisesRegex(
                Exception, "Unsupported Helion autotune effort: invalid"
            ):
                torch.compile(lambda a, b: a @ b, fullgraph=True)(*args)

    @onlyCUDA
    @dtypes(torch.bfloat16)
    def test_cuda_backend_required(self, device, dtype):
        args = [
            torch.randn(32, 64, device=device, dtype=dtype),
            torch.randn(64, 48, device=device, dtype=dtype),
        ]
        with config.patch(cuda_backend="halide"):
            with self.assertRaisesRegex(
                Exception, "Helion native matmul requires the CUDA Triton backend"
            ):
                torch.compile(lambda a, b: a @ b, fullgraph=True)(*args)

    @onlyCUDA
    @dtypes(torch.bfloat16)
    @parametrize("batched", [False, True])
    def test_horizontal(self, device, dtype, batched):
        batch = (2,) if batched else ()
        args = [
            torch.randn(*batch, 32, 64, device=device, dtype=dtype),
            torch.randn(*batch, 64, 48, device=device, dtype=dtype),
            torch.randn(*batch, 64, 48, device=device, dtype=dtype),
        ]

        def fn(a, b, c):
            if batched:
                return torch.relu(torch.bmm(a, b) - torch.bmm(a, c))
            return torch.relu(a @ b - a @ c)

        expected = fn(*(arg.float() for arg in args)).to(dtype)
        source = self.check(fn, args, expected=expected)
        self.assertEqual(
            source.count(" = torch.baddbmm(" if batched else " = torch.addmm("), 2
        )
        self.assertEqual(source.count("PyCodeCache.load("), 1)

    @onlyCUDA
    @dtypes(torch.bfloat16)
    @parametrize("batched", [False, True])
    def test_views(self, device, dtype, batched):
        batch = (2,) if batched else ()
        args = [
            torch.randn(*batch, 33, 9, 8, device=device, dtype=dtype),
            torch.randn(*batch, 136, 72, device=device, dtype=dtype),
        ]

        def fn(a, b):
            lhs = a[..., 1:, :, :].reshape(*batch, 32, 72)
            return torch.bmm(lhs, b.transpose(-1, -2)) if batched else lhs @ b.T

        source = self.check(fn, args)
        self.assertIn("reinterpret_tensor(", source)

    @onlyCUDA
    @dtypes(torch.float32)
    @parametrize("shape", [(1, 8, 7), (7, 1, 8), (7, 8, 1)])
    def test_singleton(self, device, dtype, shape):
        m, k, n = shape
        args = [
            torch.randn(m, k, device=device, dtype=dtype),
            torch.randn(k, n, device=device, dtype=dtype),
        ]
        self.check(lambda a, b: a @ b, args)

    @onlyCUDA
    @dtypes(torch.bfloat16)
    def test_addmm_out_dtype(self, device, dtype):
        args = [
            torch.randn(48, device=device, dtype=torch.float32),
            torch.randn(32, 64, device=device, dtype=dtype),
            torch.randn(64, 48, device=device, dtype=dtype),
        ]
        self.check(
            lambda bias, a, b: torch.addmm(
                bias, a, b, out_dtype=torch.float32, alpha=1.25, beta=0.75
            ),
            args,
        )

    @onlyCUDA
    @dtypes(torch.float64)
    def test_unsupported_no_fallback(self, device, dtype):
        args = [
            torch.randn(32, 64, device=device, dtype=dtype),
            torch.randn(64, 48, device=device, dtype=dtype),
        ]
        with self.assertRaisesRegex(Exception, "Helion requires native matmul"):
            torch.compile(lambda a, b: a @ b, fullgraph=True)(*args)

    @onlyCUDA
    @dtypes(torch.bfloat16)
    def test_cudagraph_tree(self, device, dtype):
        def fn(a, b):
            return torch.relu((a + 1) @ b), a.sum(dim=-1)

        args = [
            torch.randn(32, 64, device=device, dtype=dtype),
            torch.randn(64, 48, device=device, dtype=dtype),
        ]
        with config.patch("triton.cudagraphs", True):
            compiled = torch.compile(fn, fullgraph=True)
            for _ in range(3):
                torch.compiler.cudagraph_mark_step_begin()
                self.assertEqual(compiled(*args), fn(*args))
            args = [arg + 0.25 for arg in args]
            torch.compiler.cudagraph_mark_step_begin()
            self.assertEqual(compiled(*args), fn(*args))
            from torch._inductor.cudagraph_trees import get_container

            manager = get_container(torch.device(device).index or 0).tree_manager
            self.assertIsNotNone(manager)
            self.assertIsNotNone(manager.current_node)
            self.assertIsNotNone(manager.current_node.graph)

    @onlyCUDA
    @dtypes(torch.bfloat16)
    def test_cudagraph_aliased_static_inputs(self, device, dtype):
        a = torch.full((32, 64), 0.5, device=device, dtype=dtype)
        args = [
            a,
            a.view(32, 64),
            torch.full((64, 48), 0.25, device=device, dtype=dtype),
            torch.full((64, 48), 0.125, device=device, dtype=dtype),
        ]
        for arg in args:
            torch._dynamo.mark_static_address(arg, guard=True)

        def fn(a, alias, b, c):
            return torch.relu(a @ b - alias @ c)

        with config.patch("triton.cudagraphs", True):
            compiled = torch.compile(fn, fullgraph=True, dynamic=False)
            for _ in range(4):
                torch.compiler.cudagraph_mark_step_begin()
                self.assertEqual(compiled(*args), fn(*args))
            a.add_(0.25)
            torch.compiler.cudagraph_mark_step_begin()
            self.assertEqual(compiled(*args), fn(*args))
            from torch._inductor.cudagraph_trees import get_container

            manager = get_container(torch.device(device).index or 0).tree_manager
            self.assertIsNotNone(manager)
            self.assertIsNotNone(manager.current_node)
            self.assertIsNotNone(manager.current_node.graph)

    @onlyCUDA
    @dtypes(torch.bfloat16)
    def test_horizontal_outputs(self, device, dtype):
        args = [
            torch.randn(32, 64, device=device, dtype=dtype),
            torch.randn(64, 48, device=device, dtype=dtype),
            torch.randn(64, 48, device=device, dtype=dtype),
        ]
        source = self.check(lambda a, b, c: (a @ b, a @ c), args)
        self.assertEqual(source.count("PyCodeCache.load("), 1)

    @onlyCUDA
    @dtypes(torch.bfloat16)
    def test_matmul_chain(self, device, dtype):
        args = [
            torch.randn(32, 64, device=device, dtype=dtype),
            torch.randn(64, 48, device=device, dtype=dtype),
            torch.randn(48, 40, device=device, dtype=dtype),
        ]
        source = self.check(lambda a, b, c: torch.relu((a @ b) @ c), args)
        self.assertEqual(source.count("PyCodeCache.load("), 2)

    @onlyCUDA
    @dtypes(torch.bfloat16)
    def test_output_alias(self, device, dtype):
        def fn(a):
            result = a @ a.T
            return result, result.T, torch.relu(result)

        a = torch.randn(32, 64, device=device, dtype=dtype)
        result, sources = run_and_get_code(torch.compile(fn, fullgraph=True), a)
        self.assertEqual(result, fn(a))
        self.assertEqual(result[0].data_ptr(), result[1].data_ptr())
        self.assertIn("@helion.kernel(", "\n".join(sources))

    @onlyCUDA
    @dtypes(torch.float32)
    def test_training_cudagraph(self, device, dtype):
        def fn(a, b):
            return torch.relu((a + 1) @ b).sum()

        a = torch.randn(32, 64, device=device, dtype=dtype, requires_grad=True)
        b = torch.randn(64, 48, device=device, dtype=dtype, requires_grad=True)
        with config.patch("triton.cudagraphs", True):
            compiled = torch.compile(fn, fullgraph=True)
            for _ in range(3):
                torch.compiler.cudagraph_mark_step_begin()
                a.grad = b.grad = None
                loss = compiled(a, b)
                loss.backward()
                ra, rb = (
                    a.detach().clone().requires_grad_(),
                    b.detach().clone().requires_grad_(),
                )
                expected = fn(ra, rb)
                expected.backward()
                self.assertEqual(loss, expected)
                self.assertEqual(a.grad, ra.grad, atol=1e-4, rtol=1e-4)
                self.assertEqual(b.grad, rb.grad, atol=1e-4, rtol=1e-4)

    @onlyCUDA
    @dtypes(torch.bfloat16)
    def test_manual_cudagraph(self, device, dtype):
        def fn(a, b):
            return torch.relu((a + 1) @ b), a.sum(dim=-1)

        args = [
            torch.randn(32, 64, device=device, dtype=dtype),
            torch.randn(64, 48, device=device, dtype=dtype),
        ]
        compiled = torch.compile(fn, fullgraph=True)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                compiled(*args)
        torch.cuda.current_stream(device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            result = compiled(*args)
        for value in (0.25, -0.5):
            args[0].add_(value)
            graph.replay()
            self.assertEqual(result, fn(*args))

    @onlyCUDA
    @dtypes(torch.float32)
    @parametrize("option", ["benchmark_fusion", "combo_kernels"])
    def test_optional_triton_paths(self, device, dtype, option):
        def fn(a, b):
            return torch.relu(a @ b), torch.sin(a), torch.cos(b)

        args = [
            torch.randn(32, 64, device=device, dtype=dtype),
            torch.randn(64, 48, device=device, dtype=dtype),
        ]
        with config.patch({option: True}):
            self.check(fn, args)

    @onlyCUDA
    @dtypes(torch.float32)
    def test_unsupported_epilogue_stays_triton(self, device, dtype):
        args = [
            torch.randn(32, 64, device=device, dtype=dtype),
            torch.randn(64, 48, device=device, dtype=dtype),
        ]
        source = self.check(lambda a, b: torch.sin(a @ b), args)
        self.assertIn("@triton.jit", source)
        self.assertNotIn("torch.sin(", source)

    @onlyCUDA
    @dtypes(torch.float32)
    @parametrize("shape", [(32, 64, 48), (1, 1, 1)])
    def test_dynamic_no_fallback(self, device, dtype, shape):
        m, k, n = shape
        args = [
            torch.randn(m, k, device=device, dtype=dtype),
            torch.randn(k, n, device=device, dtype=dtype),
        ]
        with self.assertRaisesRegex(Exception, "Helion native matmul.*dynamic"):
            torch.compile(lambda a, b: a @ b, dynamic=True, fullgraph=True)(*args)

    @onlyCUDA
    @dtypes(torch.bfloat16)
    def test_dynamic_input_policy(self, device, dtype):
        args = [torch.ones(1, 1, device=device, dtype=dtype) for _ in range(2)]
        with torch._dynamo.config.patch(assume_static_by_default=False):
            with self.assertRaisesRegex(
                Exception, "Helion native matmul.*dynamic input settings"
            ):
                torch.compile(lambda a, b: a @ b, fullgraph=True)(*args)

    @onlyCUDA
    @dtypes(torch.bfloat16)
    def test_marked_dynamic_input(self, device, dtype):
        args = [
            torch.randn(32, 64, device=device, dtype=dtype),
            torch.randn(64, 48, device=device, dtype=dtype),
        ]
        torch._dynamo.mark_dynamic(args[0], 0)
        with self.assertRaisesRegex(Exception, "Helion native matmul.*static shapes"):
            torch.compile(lambda a, b: a @ b, fullgraph=True)(*args)

    @onlyCUDA
    @dtypes(torch.bfloat16)
    def test_dynamic_non_matmul_input(self, device, dtype):
        args = [
            torch.randn(32, 64, device=device, dtype=dtype),
            torch.randn(64, 48, device=device, dtype=dtype),
            torch.randn(17, 3, device=device, dtype=dtype),
        ]
        torch._dynamo.mark_dynamic(args[2], 0)
        with self.assertRaisesRegex(Exception, "Helion native matmul.*static shapes"):
            torch.compile(lambda a, b, c: (a @ b, torch.sin(c)), fullgraph=True)(*args)

    @onlyCUDA
    @dtypes(torch.bfloat16)
    def test_automatic_dynamic_recompile(self, device, dtype):
        b = torch.randn(64, 48, device=device, dtype=dtype)
        with torch._dynamo.config.patch(automatic_dynamic_shapes=True):
            compiled = torch.compile(lambda a, b: a @ b, fullgraph=True)
            a = torch.randn(32, 64, device=device, dtype=dtype)
            self.assertEqual(compiled(a, b), a @ b)
            a = torch.randn(33, 64, device=device, dtype=dtype)
            with self.assertRaisesRegex(
                Exception, "Helion native matmul.*static shapes"
            ):
                compiled(a, b)

    @onlyCUDA
    @dtypes(torch.int8)
    def test_int_mm_no_fallback(self, device, dtype):
        args = [
            torch.ones(32, 64, device=device, dtype=dtype),
            torch.ones(64, 48, device=device, dtype=dtype),
        ]
        with self.assertRaisesRegex(
            Exception, "Helion requires native matmul; unsupported operator"
        ):
            torch.compile(torch._int_mm, fullgraph=True)(*args)

    @onlyCUDA
    @dtypes(torch.float32)
    def test_runtime_settings_are_explicit(self, device, dtype):
        args = [
            torch.randn(32, 64, device=device, dtype=dtype),
            torch.randn(64, 48, device=device, dtype=dtype),
        ]
        with patch.dict(
            os.environ, {"HELION_BACKEND": "cute", "HELION_INTERPRET": "1"}
        ):
            self.check(lambda a, b: a @ b, args)


@unittest.skipUnless(
    HAS_CUDA_AND_TRITON and HAS_HELION and HAS_CUTEDSL,
    "requires CUDA, Triton, Helion and CuTeDSL",
)
class TestNativeMatmulHelionCuTeDSL(TestNativeMatmulHelion):
    helion_backend = "cutedsl"
    test_dynamic_no_fallback = TestNativeMatmulHelion.test_dynamic_no_fallback
    test_dynamic_input_policy = TestNativeMatmulHelion.test_dynamic_input_policy
    test_marked_dynamic_input = TestNativeMatmulHelion.test_marked_dynamic_input
    test_dynamic_non_matmul_input = TestNativeMatmulHelion.test_dynamic_non_matmul_input
    test_automatic_dynamic_recompile = (
        TestNativeMatmulHelion.test_automatic_dynamic_recompile
    )
    test_cudagraph_aliased_static_inputs = (
        TestNativeMatmulHelion.test_cudagraph_aliased_static_inputs
    )
    test_precision_environment = TestNativeMatmulHelion.test_precision_environment
    test_fused_matmul = TestNativeMatmulHelion.test_fused_matmul
    test_horizontal = TestNativeMatmulHelion.test_horizontal
    test_views = TestNativeMatmulHelion.test_views
    test_singleton = TestNativeMatmulHelion.test_singleton
    test_addmm_out_dtype = TestNativeMatmulHelion.test_addmm_out_dtype
    test_cudagraph_tree = TestNativeMatmulHelion.test_cudagraph_tree
    test_horizontal_outputs = TestNativeMatmulHelion.test_horizontal_outputs
    test_matmul_chain = TestNativeMatmulHelion.test_matmul_chain
    test_output_alias = TestNativeMatmulHelion.test_output_alias
    test_training_cudagraph = TestNativeMatmulHelion.test_training_cudagraph
    test_manual_cudagraph = TestNativeMatmulHelion.test_manual_cudagraph
    test_optional_triton_paths = TestNativeMatmulHelion.test_optional_triton_paths
    test_unsupported_epilogue_stays_triton = (
        TestNativeMatmulHelion.test_unsupported_epilogue_stays_triton
    )
    test_runtime_settings_are_explicit = (
        TestNativeMatmulHelion.test_runtime_settings_are_explicit
    )


class TestHelionKernelWrapper(TestCase):
    hw_classification = HardwareClassification.CPU

    def test_backend_defaults(self):
        self.assertEqual(config.helion_backend, "triton")
        self.assertEqual(config.helion_autotune_effort, "none")
        self.assertEqual(config.cuda_backend, "triton")
        self.assertFalse(config.native_matmul_helion)

    def test_prepared_device_and_alignment(self):
        from unittest.mock import Mock

        from torch._inductor.runtime.helion import HelionKernelWrapper

        kernel = Mock()
        kernel.specialization_key.return_value = ("same_shape",)
        wrapper = HelionKernelWrapper(kernel)
        arg = Mock(device=torch.device("cuda:0"))
        arg.data_ptr.return_value = 16
        with patch("torch.cuda.is_current_stream_capturing", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "warmed up"):
                wrapper(arg)
        kernel.assert_not_called()
        with patch("torch.cuda.is_current_stream_capturing", return_value=False):
            wrapper(arg)
        with patch("torch.cuda.is_current_stream_capturing", return_value=True):
            wrapper(arg)
            arg.device = torch.device("cuda:1")
            with self.assertRaisesRegex(RuntimeError, "warmed up"):
                wrapper(arg)
            arg.device = torch.device("cuda:0")
            arg.data_ptr.return_value = 17
            with self.assertRaisesRegex(RuntimeError, "warmed up"):
                wrapper(arg)
        self.assertEqual(kernel.call_count, 2)


instantiate_device_type_tests(
    TestNativeMatmulHelionCuTeDSL, globals(), only_for=("cuda",)
)
instantiate_device_type_tests(TestNativeMatmulHelion, globals(), only_for=("cuda",))

if __name__ == "__main__":
    run_tests()
