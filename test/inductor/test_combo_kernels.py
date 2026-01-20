# Owner(s): ["module: inductor"]

import contextlib
import sys
import unittest

import torch
import torch._inductor
from torch._inductor.utils import run_and_get_code
from torch.testing import FileCheck
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    TestCase,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_CPU, HAS_GPU_AND_TRITON
from torch.testing._internal.triton_utils import requires_gpu_and_triton


aten = torch.ops.aten

try:
    try:
        from .test_torchinductor import check_model, check_model_gpu
    except ImportError:
        from test_torchinductor import (  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
            check_model,
            check_model_gpu,
        )
except (unittest.SkipTest, ImportError) as e:
    sys.stderr.write(f"{type(e)}: {e}\n")
    if __name__ == "__main__":
        sys.exit(0)
    raise


@instantiate_parametrized_tests
class ComboKernelTests(TestCase):
    check_model_gpu = check_model_gpu
    check_model_cpu = check_model
    check_kernel_count = True

    def setUp(self):
        super().setUp()
        torch._inductor.metrics.reset()
        self._test_stack = contextlib.ExitStack()
        self._test_stack.enter_context(
            torch._inductor.config.patch(
                {
                    "combo_kernels": True,
                    "benchmark_combo_kernel": False,
                }
            )
        )

    def tearDown(self):
        self._test_stack.close()
        torch._inductor.metrics.reset()
        super().tearDown()

    @requires_gpu_and_triton
    def test_activation_functions(self):
        def test_activations(a, b, c):
            a1 = torch.nn.functional.relu(a)
            b1 = torch.nn.functional.sigmoid(b)
            c1 = torch.nn.functional.tanh(c)
            return a1, b1, c1

        inps = [
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(20, 20, device=GPU_TYPE),
            torch.rand(10, 10, device=GPU_TYPE),
        ]

        out_eager = test_activations(*inps)
        out_compiled = torch.compile(test_activations)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_gpu_and_triton
    def test_reduce_functions(self):
        def test_reduce(a, b, c, d):
            a1 = torch.sum(a, dim=0)
            b1 = torch.max(b, dim=0)
            c1 = torch.min(c, dim=0)
            d1 = torch.nn.functional.tanh(d)

            return a1, b1, c1, d1

        inps = [
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(20, 20, device=GPU_TYPE),
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(30, 8, device=GPU_TYPE),
        ]

        out_eager = test_reduce(*inps)
        out_compiled = torch.compile(test_reduce)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertTrue(torch._inductor.metrics.generated_kernel_count <= 2)

    @requires_gpu_and_triton
    def test_mutated_args(self):
        def test_mutated(a, b, c, d):
            a.add_(1)
            b.sigmoid_()
            c = torch.add(c, 5)
            d.tanh_()

            return a, b, c, d

        inps = [
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(20, 20, device=GPU_TYPE),
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(30, 8, device=GPU_TYPE),
        ]

        out_eager = test_mutated(*inps)
        out_compiled = torch.compile(test_mutated)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_gpu_and_triton
    def test_reduce_split(self):
        def fn(a, b):
            a1 = torch.linalg.vector_norm(a)
            b1 = torch.sum(b, dim=0)
            return a1, b1

        inps = [
            torch.rand(2048, 512, device=GPU_TYPE),
            torch.rand(20, 20, device=GPU_TYPE),
        ]
        out_eager = fn(*inps)
        out_compiled = torch.compile(fn)(*inps)

        self.assertEqual(out_eager, out_compiled)

    @requires_gpu_and_triton
    def test_2d_blocking_partitioning(self):
        def fn(a0, a1, a2, b0, b1, b2):
            c0 = torch.add(a0, b0)
            c1 = torch.add(a1, b1)
            c2 = torch.add(a2, b2)
            return c0, c1, c2

        self.check_model_gpu(
            fn,
            (
                torch.rand(30, 20, device=GPU_TYPE),
                torch.rand(40, 30, device=GPU_TYPE),
                torch.rand(36, 40, device=GPU_TYPE),
                torch.rand(30, 20, device=GPU_TYPE),
                torch.rand(30, 40, device=GPU_TYPE).t(),
                torch.rand(40, 36, device=GPU_TYPE).t(),
            ),
        )

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 2)

    @requires_gpu_and_triton
    def test_persistent_reduction_size_hint(self):
        def fn(x, y):
            return x.max(1), y.min(1)

        inps = (
            torch.rand(768, 16, device=GPU_TYPE),
            torch.rand(768, 32, device=GPU_TYPE),
        )

        out_eager = fn(*inps)
        fn_c = torch.compile(fn)
        out_compiled, code = run_and_get_code(fn_c, *inps)
        FileCheck().check("triton_heuristics.persistent_reduction").check(
            "size_hints={'x': 1024, 'r0_': 32}"
        ).run(code[0])
        self.assertEqual(out_eager, out_compiled)

    @requires_gpu_and_triton
    def test_fuse_mix_order_reductions_combo_kernels(self):
        def fn(x, y, z):
            # FusedMixOrderReductions produces row_sum (buf0)
            row_sum = x.sum(dim=1)
            col_sum = x.sum(dim=0)

            # consumer of row_sum - excluded from combo kernels
            row_sum_reduced = row_sum.sum()  # reads buf0

            # independent reductions - combo-kerneled
            y_sum = y.sum()
            z_sum = z.sum()

            return row_sum_reduced, col_sum, y_sum, z_sum

        inps = [
            torch.rand(8192, 1024, device=GPU_TYPE),
            torch.rand(2048, device=GPU_TYPE),
            torch.rand(2048, device=GPU_TYPE),
        ]
        out_eager = fn(*inps)
        fn_c = torch.compile(fn)
        out_compiled, code = run_and_get_code(fn_c, *inps)
        self.assertEqual(out_eager, out_compiled)
        # [row_sum, col_sum] will became 1 kernel MixOrderReductionGrid
        # [row_sum_reduced] will become a separate kernel due to the consumer
        # [y_sum, z_sum] will become a combo kernel
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 3)


@instantiate_parametrized_tests
class ComboKernelBenchmarkTests(TestCase):
    check_model_gpu = check_model_gpu
    check_model_cpu = check_model
    check_kernel_count = True

    def setUp(self):
        super().setUp()
        torch._inductor.metrics.reset()
        self._test_stack = contextlib.ExitStack()
        self._test_stack.enter_context(
            torch._inductor.config.patch(
                {
                    "combo_kernels": True,
                    "benchmark_combo_kernel": True,
                }
            )
        )

    def tearDown(self):
        self._test_stack.close()
        torch._inductor.metrics.reset()
        super().tearDown()

    @requires_gpu_and_triton
    def test_activation_benchmark(self):
        def test_activations(a, b, c):
            a1 = torch.nn.functional.relu(a)
            b1 = torch.nn.functional.sigmoid(b)
            c1 = torch.nn.functional.tanh(c)
            return a1, b1, c1

        inps = [
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(20, 20, device=GPU_TYPE),
            torch.rand(10, 10, device=GPU_TYPE),
        ]

        out_eager = test_activations(*inps)
        out_compiled = torch.compile(test_activations)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 5)

    @requires_gpu_and_triton
    def test_reduce_benchmark(self):
        def test_reduce(a, b, c, d):
            a1 = torch.sum(a, dim=0)
            b1 = torch.max(b, dim=0)
            c1 = torch.min(c, dim=0)
            d1 = torch.nn.functional.tanh(d)

            return a1, b1, c1, d1

        inps = [
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(20, 20, device=GPU_TYPE),
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(30, 8, device=GPU_TYPE),
        ]

        out_eager = test_reduce(*inps)
        out_compiled = torch.compile(test_reduce)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertTrue(4 < torch._inductor.metrics.generated_kernel_count <= 10)

    @requires_gpu_and_triton
    def test_mutated_benchmark(self):
        def test_mutated(a, b, c, d):
            a.add_(1)
            b.sigmoid_()
            c = torch.add(c, 5)
            d.tanh_()

            return a, b, c, d

        inps = [
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(20, 20, device=GPU_TYPE),
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(30, 8, device=GPU_TYPE),
        ]

        out_eager = test_mutated(*inps)
        out_compiled = torch.compile(test_mutated)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertTrue(torch._inductor.metrics.generated_kernel_count in [6, 9])

    @requires_gpu_and_triton
    def test_round_robin_dispatch(self):
        # combo kernel dispatch strategy: round robin
        def test_mutated(a, b, c, d):
            a.add_(1)
            b.sigmoid_()
            c = torch.add(c, 5)
            d.tanh_()

            return a, b, c, d

        inps = [
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(20, 5, device=GPU_TYPE),
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(5, 18, device=GPU_TYPE),
        ]

        out_eager = test_mutated(*inps)
        out_compiled = torch.compile(test_mutated)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 6)

    @requires_gpu_and_triton
    def test_2d_blocking_benchmark(self):
        def fn(a0, a1, a2, b0, b1, b2):
            c0 = torch.add(a0, b0)
            c1 = torch.add(a1, b1)
            c2 = torch.add(a2, b2)
            return c0, c1, c2

        self.check_model_gpu(
            fn,
            (
                torch.rand(30, 20, device=GPU_TYPE),
                torch.rand(40, 30, device=GPU_TYPE),
                torch.rand(36, 40, device=GPU_TYPE),
                torch.rand(30, 20, device=GPU_TYPE),
                torch.rand(30, 40, device=GPU_TYPE).t(),
                torch.rand(40, 36, device=GPU_TYPE).t(),
            ),
        )

        self.assertTrue(7 <= torch._inductor.metrics.generated_kernel_count <= 8)

    @requires_gpu_and_triton
    def test_persistent_reduction_no_x_dim(self):
        def fn(x, y):
            return x.sum(1), y.sum(1)

        inps = (
            torch.rand(16, 256, device=GPU_TYPE),
            torch.rand(32, 256, device=GPU_TYPE),
        )
        torch._dynamo.mark_dynamic(inps[0], 0, min=1, max=256)
        torch._dynamo.mark_dynamic(inps[1], 0, min=1, max=256)
        out_eager = fn(*inps)
        out_compiled = torch.compile(fn)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 4)


@instantiate_parametrized_tests
class ComboKernelDynamicShapesTests(TestCase):
    check_model_gpu = check_model_gpu
    check_model_cpu = check_model
    check_kernel_count = True

    def setUp(self):
        super().setUp()
        torch._inductor.metrics.reset()
        self._test_stack = contextlib.ExitStack()
        self._test_stack.enter_context(
            torch._inductor.config.patch(
                {
                    "combo_kernels": True,
                    "benchmark_combo_kernel": True,
                }
            )
        )
        self._test_stack.enter_context(
            torch._dynamo.config.patch(
                {
                    "automatic_dynamic_shapes": False,
                    "assume_static_by_default": False,
                }
            )
        )

    def tearDown(self):
        self._test_stack.close()
        torch._inductor.metrics.reset()
        super().tearDown()

    @requires_gpu_and_triton
    def test_dynamic_shapes_activations(self):
        def test_activations(a, b, c):
            a1 = torch.nn.functional.relu(a)
            b1 = torch.nn.functional.sigmoid(b)
            c1 = torch.nn.functional.tanh(c)
            return a1, b1, c1

        inps = [
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(20, 20, device=GPU_TYPE),
            torch.rand(10, 10, device=GPU_TYPE),
        ]

        out_eager = test_activations(*inps)
        out_compiled = torch.compile(test_activations)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 5)

    @requires_gpu_and_triton
    def test_dynamic_shapes_2d_blocking(self):
        def fn(a0, a1, a2, b0, b1, b2):
            c0 = torch.add(a0, b0)
            c1 = torch.add(a1, b1)
            c2 = torch.add(a2, b2)
            return c0, c1, c2

        self.check_model_gpu(
            fn,
            (
                torch.rand(30, 20, device=GPU_TYPE),
                torch.rand(40, 30, device=GPU_TYPE),
                torch.rand(36, 40, device=GPU_TYPE),
                torch.rand(30, 20, device=GPU_TYPE),
                torch.rand(30, 40, device=GPU_TYPE).t(),
                torch.rand(40, 36, device=GPU_TYPE).t(),
            ),
        )

        self.assertTrue(7 <= torch._inductor.metrics.generated_kernel_count <= 8)

    @requires_gpu_and_triton
    def test_dynamic_shapes_reduce(self):
        def test_reduce(a, b, c, d):
            a1 = torch.sum(a, dim=0)
            b1 = torch.max(b, dim=0)
            c1 = torch.min(c, dim=0)
            d1 = torch.nn.functional.tanh(d)

            return a1, b1, c1, d1

        inps = [
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(20, 20, device=GPU_TYPE),
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(30, 8, device=GPU_TYPE),
        ]

        out_eager = test_reduce(*inps)
        out_compiled = torch.compile(test_reduce)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertTrue(4 < torch._inductor.metrics.generated_kernel_count <= 10)

    @requires_gpu_and_triton
    def test_dynamic_shapes_mutated(self):
        # combo kernel dispatch strategy: round robin
        def test_mutated(a, b, c, d):
            a.add_(1)
            b.sigmoid_()
            c = torch.add(c, 5)
            d.tanh_()

            return a, b, c, d

        inps = [
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(20, 5, device=GPU_TYPE),
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(5, 18, device=GPU_TYPE),
        ]

        out_eager = test_mutated(*inps)
        out_compiled = torch.compile(test_mutated)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 6)

    @requires_gpu_and_triton
    @torch._inductor.config.patch("combo_kernels_autotune", 0)
    def test_dynamic_shapes_activations_no_autotune(self):
        def test_activations(a, b, c):
            a1 = torch.nn.functional.relu(a)
            b1 = torch.nn.functional.sigmoid(b)
            c1 = torch.nn.functional.tanh(c)
            return a1, b1, c1

        inps = [
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(20, 20, device=GPU_TYPE),
            torch.rand(10, 10, device=GPU_TYPE),
        ]

        out_eager = test_activations(*inps)
        out_compiled = torch.compile(test_activations)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 5)

    @requires_gpu_and_triton
    @torch._dynamo.config.patch("automatic_dynamic_shapes", True)
    @torch._dynamo.config.patch("assume_static_by_default", True)
    def test_dynamic_shapes_persistent_reduction_no_x_dim(self):
        def fn(x, y):
            return x.sum(1), y.sum(1)

        inps = (
            torch.rand(16, 256, device=GPU_TYPE),
            torch.rand(32, 256, device=GPU_TYPE),
        )
        torch._dynamo.mark_dynamic(inps[0], 0, min=1, max=256)
        torch._dynamo.mark_dynamic(inps[1], 0, min=1, max=256)
        out_eager = fn(*inps)
        out_compiled = torch.compile(fn)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 4)

    @requires_gpu_and_triton
    @torch._dynamo.config.patch("automatic_dynamic_shapes", True)
    @torch._dynamo.config.patch("assume_static_by_default", True)
    def test_dynamic_shapes_persistent_reduction_no_x_dim_2(self):
        def fn(x, y):
            return x.sum(2), y.sum(2)

        inps = (
            torch.rand(8, 16, 256, device=GPU_TYPE),
            torch.rand(8, 32, 256, device=GPU_TYPE),
        )
        torch._dynamo.mark_dynamic(inps[0], (0, 1), min=1, max=256)
        torch._dynamo.mark_dynamic(inps[1], (0, 1), min=1, max=256)
        out_eager = fn(*inps)
        out_compiled = torch.compile(fn)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 4)

    @requires_gpu_and_triton
    @torch._dynamo.config.patch("automatic_dynamic_shapes", True)
    @torch._dynamo.config.patch("assume_static_by_default", True)
    def test_dynamic_shapes_2d_blocking_round_robin(self):
        def fn(a0, a1, a2, b0, b1, b2):
            c0 = torch.add(a0, b0)
            c1 = torch.add(a1, b1)
            c2 = torch.add(a2, b2)
            return c0, c1, c2

        inps = (
            torch.rand(20, 30, device=GPU_TYPE),
            torch.rand(30, 30, device=GPU_TYPE),
            torch.rand(40, 32, device=GPU_TYPE),
            torch.rand(30, 20, device=GPU_TYPE).t(),
            torch.rand(30, 30, device=GPU_TYPE).t(),
            torch.rand(32, 40, device=GPU_TYPE).t(),
        )

        out_eager = fn(*inps)
        compiled = torch.compile(fn)
        out_compiled = compiled(*inps)
        self.assertEqual(out_eager, out_compiled)
        self.assertTrue(5 <= torch._inductor.metrics.generated_kernel_count <= 6)
        torch._inductor.metrics.reset()

        inps = (
            torch.rand(24, 30, device=GPU_TYPE),
            torch.rand(32, 30, device=GPU_TYPE),
            torch.rand(48, 32, device=GPU_TYPE),
            torch.rand(30, 24, device=GPU_TYPE).t(),
            torch.rand(30, 32, device=GPU_TYPE).t(),
            torch.rand(32, 48, device=GPU_TYPE).t(),
        )
        out_compiled = compiled(*inps)
        out_eager = fn(*inps)
        self.assertEqual(out_eager, out_compiled)
        self.assertTrue(5 <= torch._inductor.metrics.generated_kernel_count <= 6)

    @requires_gpu_and_triton
    @torch._dynamo.config.patch("automatic_dynamic_shapes", True)
    @torch._dynamo.config.patch("assume_static_by_default", True)
    @torch._inductor.config.patch("triton.autotune_at_compile_time", True)
    def test_dynamic_shapes_persistent_reduction_mixed_x_dim_cuda(self):
        def fn(x, y, z):
            return x.sum(1), y.mean(1), z.max(1)

        inps = (
            torch.rand(16, 128, device=GPU_TYPE),
            torch.rand(32, 128, device=GPU_TYPE),
            torch.rand(32, 256, device=GPU_TYPE),
        )
        torch._dynamo.mark_dynamic(inps[0], 0, min=1, max=256)
        torch._dynamo.mark_dynamic(inps[1], 0, min=1, max=256)
        torch._dynamo.mark_dynamic(inps[2], 0, min=1, max=256)
        out_eager = fn(*inps)
        out_compiled = torch.compile(fn)(*inps)

        self.assertEqual(out_eager, out_compiled)

    @requires_gpu_and_triton
    def test_helper_fn_defined(self):
        def fn(x, y, z):
            return x.sum(1), y.mean(1), z.cumsum(1)

        inps = (
            torch.rand(16, 128, device=GPU_TYPE),
            torch.rand(32, 128, device=GPU_TYPE),
            torch.rand(32, 256, device=GPU_TYPE),
        )

        out_eager = fn(*inps)
        fn_c = torch.compile(fn)
        out_compiled, code = run_and_get_code(fn_c, *inps)
        code = " ".join(code)
        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(code.count("def _triton_helper_fn_add0(arg0_0, arg1_0):"), 1)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if HAS_CPU or HAS_GPU_AND_TRITON:
        run_tests(needs="filelock")
