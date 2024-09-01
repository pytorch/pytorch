# Owner(s): ["module: inductor"]

import sys
import unittest

import torch
import torch._inductor
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    TestCase,
)
from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA
from torch.testing._internal.triton_utils import requires_cuda


aten = torch.ops.aten

try:
    try:
        from .test_torchinductor import check_model, check_model_cuda
    except ImportError:
        from test_torchinductor import check_model, check_model_cuda
except (unittest.SkipTest, ImportError) as e:
    sys.stderr.write(f"{type(e)}: {e}\n")
    if __name__ == "__main__":
        sys.exit(0)
    raise


@instantiate_parametrized_tests
class ComboKernelTests(TestCase):
    check_model_cuda = check_model_cuda
    check_model_cpu = check_model
    check_kernel_count = True

    def setUp(self):
        super().setUp()
        torch._inductor.metrics.reset()
        torch._inductor.config.combo_kernels = True
        torch._inductor.config.benchmark_combo_kernel = False

    def tearDown(self):
        super().tearDown()
        torch._inductor.metrics.reset()

    @requires_cuda
    def test_activation_functions(self):
        def test_activations(a, b, c):
            a1 = torch.nn.functional.relu(a)
            b1 = torch.nn.functional.sigmoid(b)
            c1 = torch.nn.functional.tanh(c)
            return a1, b1, c1

        inps = [
            torch.rand(10, 10, device="cuda"),
            torch.rand(20, 20, device="cuda"),
            torch.rand(10, 10, device="cuda"),
        ]

        out_eager = test_activations(*inps)
        out_compiled = torch.compile(test_activations)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_cuda
    def test_reduce_functions(self):
        def test_reduce(a, b, c, d):
            a1 = torch.sum(a, dim=0)
            b1 = torch.max(b, dim=0)
            c1 = torch.min(c, dim=0)
            d1 = torch.nn.functional.tanh(d)

            return a1, b1, c1, d1

        inps = [
            torch.rand(10, 10, device="cuda"),
            torch.rand(20, 20, device="cuda"),
            torch.rand(10, 10, device="cuda"),
            torch.rand(30, 8, device="cuda"),
        ]

        out_eager = test_reduce(*inps)
        out_compiled = torch.compile(test_reduce)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertTrue(torch._inductor.metrics.generated_kernel_count <= 2)

    @requires_cuda
    def test_mutated_args(self):
        def test_mutated(a, b, c, d):
            a.add_(1)
            b.sigmoid_()
            c = torch.add(c, 5)
            d.tanh_()

            return a, b, c, d

        inps = [
            torch.rand(10, 10, device="cuda"),
            torch.rand(20, 20, device="cuda"),
            torch.rand(10, 10, device="cuda"),
            torch.rand(30, 8, device="cuda"),
        ]

        out_eager = test_mutated(*inps)
        out_compiled = torch.compile(test_mutated)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_cuda
    def test_reduce_split(self):
        def fn(a, b):
            a1 = torch.linalg.vector_norm(a)
            b1 = torch.sum(b, dim=0)
            return a1, b1

        inps = [
            torch.rand(2048, 512, device="cuda"),
            torch.rand(20, 20, device="cuda"),
        ]
        out_eager = fn(*inps)
        out_compiled = torch.compile(fn)(*inps)

        self.assertEqual(out_eager, out_compiled)

    @requires_cuda
    def test_2d_blocking_partitioning(self):
        def fn(a0, a1, a2, b0, b1, b2):
            c0 = torch.add(a0, b0)
            c1 = torch.add(a1, b1)
            c2 = torch.add(a2, b2)
            return c0, c1, c2

        self.check_model_cuda(
            fn,
            (
                torch.rand(30, 20, device="cuda"),
                torch.rand(40, 30, device="cuda"),
                torch.rand(36, 40, device="cuda"),
                torch.rand(30, 20, device="cuda"),
                torch.rand(30, 40, device="cuda").t(),
                torch.rand(40, 36, device="cuda").t(),
            ),
        )

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 2)


@instantiate_parametrized_tests
class ComboKernelBenchmarkTests(TestCase):
    check_model_cuda = check_model_cuda
    check_model_cpu = check_model
    check_kernel_count = True

    def setUp(self):
        super().setUp()
        torch._inductor.metrics.reset()
        torch._inductor.config.combo_kernels = True
        torch._inductor.config.benchmark_combo_kernel = True

    def tearDown(self):
        super().tearDown()
        torch._inductor.metrics.reset()

    @requires_cuda
    def test_activation_benchmark(self):
        def test_activations(a, b, c):
            a1 = torch.nn.functional.relu(a)
            b1 = torch.nn.functional.sigmoid(b)
            c1 = torch.nn.functional.tanh(c)
            return a1, b1, c1

        inps = [
            torch.rand(10, 10, device="cuda"),
            torch.rand(20, 20, device="cuda"),
            torch.rand(10, 10, device="cuda"),
        ]

        out_eager = test_activations(*inps)
        out_compiled = torch.compile(test_activations)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 5)

    @requires_cuda
    def test_reduce_benchmark(self):
        def test_reduce(a, b, c, d):
            a1 = torch.sum(a, dim=0)
            b1 = torch.max(b, dim=0)
            c1 = torch.min(c, dim=0)
            d1 = torch.nn.functional.tanh(d)

            return a1, b1, c1, d1

        inps = [
            torch.rand(10, 10, device="cuda"),
            torch.rand(20, 20, device="cuda"),
            torch.rand(10, 10, device="cuda"),
            torch.rand(30, 8, device="cuda"),
        ]

        out_eager = test_reduce(*inps)
        out_compiled = torch.compile(test_reduce)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertTrue(4 < torch._inductor.metrics.generated_kernel_count <= 10)

    @requires_cuda
    def test_mutated_benchmark(self):
        def test_mutated(a, b, c, d):
            a.add_(1)
            b.sigmoid_()
            c = torch.add(c, 5)
            d.tanh_()

            return a, b, c, d

        inps = [
            torch.rand(10, 10, device="cuda"),
            torch.rand(20, 20, device="cuda"),
            torch.rand(10, 10, device="cuda"),
            torch.rand(30, 8, device="cuda"),
        ]

        out_eager = test_mutated(*inps)
        out_compiled = torch.compile(test_mutated)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertTrue(torch._inductor.metrics.generated_kernel_count in [6, 9])

    @requires_cuda
    def test_round_robin_dispatch(self):
        # combo kernel dispatch strategy: round robin
        def test_mutated(a, b, c, d):
            a.add_(1)
            b.sigmoid_()
            c = torch.add(c, 5)
            d.tanh_()

            return a, b, c, d

        inps = [
            torch.rand(10, 10, device="cuda"),
            torch.rand(20, 5, device="cuda"),
            torch.rand(10, 10, device="cuda"),
            torch.rand(5, 18, device="cuda"),
        ]

        out_eager = test_mutated(*inps)
        out_compiled = torch.compile(test_mutated)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 6)

    @requires_cuda
    def test_2d_blocking_benchmark(self):
        def fn(a0, a1, a2, b0, b1, b2):
            c0 = torch.add(a0, b0)
            c1 = torch.add(a1, b1)
            c2 = torch.add(a2, b2)
            return c0, c1, c2

        self.check_model_cuda(
            fn,
            (
                torch.rand(30, 20, device="cuda"),
                torch.rand(40, 30, device="cuda"),
                torch.rand(36, 40, device="cuda"),
                torch.rand(30, 20, device="cuda"),
                torch.rand(30, 40, device="cuda").t(),
                torch.rand(40, 36, device="cuda").t(),
            ),
        )

        self.assertTrue(7 <= torch._inductor.metrics.generated_kernel_count <= 8)

    @requires_cuda
    def test_persistent_reduction_no_x_dim(self):
        def fn(x, y):
            return x.sum(1), y.sum(1)

        inps = (
            torch.rand(16, 256, device="cuda"),
            torch.rand(32, 256, device="cuda"),
        )
        torch._dynamo.mark_dynamic(inps[0], 0, min=1, max=256)
        torch._dynamo.mark_dynamic(inps[1], 0, min=1, max=256)
        out_eager = fn(*inps)
        out_compiled = torch.compile(fn)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 4)


@instantiate_parametrized_tests
class ComboKernelDynamicShapesTests(TestCase):
    check_model_cuda = check_model_cuda
    check_model_cpu = check_model
    check_kernel_count = True

    def setUp(self):
        super().setUp()
        torch._inductor.metrics.reset()
        torch._inductor.config.combo_kernels = True
        torch._inductor.config.benchmark_combo_kernel = True
        torch._dynamo.config.automatic_dynamic_shapes = False
        torch._dynamo.config.assume_static_by_default = False

    def tearDown(self):
        super().tearDown()
        torch._inductor.metrics.reset()

    @requires_cuda
    def test_dynamic_shapes_activations(self):
        def test_activations(a, b, c):
            a1 = torch.nn.functional.relu(a)
            b1 = torch.nn.functional.sigmoid(b)
            c1 = torch.nn.functional.tanh(c)
            return a1, b1, c1

        inps = [
            torch.rand(10, 10, device="cuda"),
            torch.rand(20, 20, device="cuda"),
            torch.rand(10, 10, device="cuda"),
        ]

        out_eager = test_activations(*inps)
        out_compiled = torch.compile(test_activations)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 5)

    @requires_cuda
    def test_dynamic_shapes_2d_blocking(self):
        def fn(a0, a1, a2, b0, b1, b2):
            c0 = torch.add(a0, b0)
            c1 = torch.add(a1, b1)
            c2 = torch.add(a2, b2)
            return c0, c1, c2

        self.check_model_cuda(
            fn,
            (
                torch.rand(30, 20, device="cuda"),
                torch.rand(40, 30, device="cuda"),
                torch.rand(36, 40, device="cuda"),
                torch.rand(30, 20, device="cuda"),
                torch.rand(30, 40, device="cuda").t(),
                torch.rand(40, 36, device="cuda").t(),
            ),
        )

        self.assertTrue(7 <= torch._inductor.metrics.generated_kernel_count <= 8)

    @requires_cuda
    def test_dynamic_shapes_reduce(self):
        def test_reduce(a, b, c, d):
            a1 = torch.sum(a, dim=0)
            b1 = torch.max(b, dim=0)
            c1 = torch.min(c, dim=0)
            d1 = torch.nn.functional.tanh(d)

            return a1, b1, c1, d1

        inps = [
            torch.rand(10, 10, device="cuda"),
            torch.rand(20, 20, device="cuda"),
            torch.rand(10, 10, device="cuda"),
            torch.rand(30, 8, device="cuda"),
        ]

        out_eager = test_reduce(*inps)
        out_compiled = torch.compile(test_reduce)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertTrue(4 < torch._inductor.metrics.generated_kernel_count <= 10)

    @requires_cuda
    def test_dynamic_shapes_mutated(self):
        # combo kernel dispatch strategy: round robin
        def test_mutated(a, b, c, d):
            a.add_(1)
            b.sigmoid_()
            c = torch.add(c, 5)
            d.tanh_()

            return a, b, c, d

        inps = [
            torch.rand(10, 10, device="cuda"),
            torch.rand(20, 5, device="cuda"),
            torch.rand(10, 10, device="cuda"),
            torch.rand(5, 18, device="cuda"),
        ]

        out_eager = test_mutated(*inps)
        out_compiled = torch.compile(test_mutated)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 6)

    @requires_cuda
    @torch._inductor.config.patch("combo_kernels_autotune", 0)
    def test_dynamic_shapes_activations_no_autotune(self):
        def test_activations(a, b, c):
            a1 = torch.nn.functional.relu(a)
            b1 = torch.nn.functional.sigmoid(b)
            c1 = torch.nn.functional.tanh(c)
            return a1, b1, c1

        inps = [
            torch.rand(10, 10, device="cuda"),
            torch.rand(20, 20, device="cuda"),
            torch.rand(10, 10, device="cuda"),
        ]

        out_eager = test_activations(*inps)
        out_compiled = torch.compile(test_activations)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 5)

    @requires_cuda
    @torch._dynamo.config.patch("automatic_dynamic_shapes", True)
    @torch._dynamo.config.patch("assume_static_by_default", True)
    def test_dynamic_shapes_persistent_reduction_no_x_dim(self):
        def fn(x, y):
            return x.sum(1), y.sum(1)

        inps = (
            torch.rand(16, 256, device="cuda"),
            torch.rand(32, 256, device="cuda"),
        )
        torch._dynamo.mark_dynamic(inps[0], 0, min=1, max=256)
        torch._dynamo.mark_dynamic(inps[1], 0, min=1, max=256)
        out_eager = fn(*inps)
        out_compiled = torch.compile(fn)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 4)

    @requires_cuda
    @torch._dynamo.config.patch("automatic_dynamic_shapes", True)
    @torch._dynamo.config.patch("assume_static_by_default", True)
    def test_dynamic_shapes_2d_blocking_round_robin(self):
        def fn(a0, a1, a2, b0, b1, b2):
            c0 = torch.add(a0, b0)
            c1 = torch.add(a1, b1)
            c2 = torch.add(a2, b2)
            return c0, c1, c2

        inps = (
            torch.rand(20, 30, device="cuda"),
            torch.rand(30, 30, device="cuda"),
            torch.rand(40, 32, device="cuda"),
            torch.rand(30, 20, device="cuda").t(),
            torch.rand(30, 30, device="cuda").t(),
            torch.rand(32, 40, device="cuda").t(),
        )

        out_eager = fn(*inps)
        compiled = torch.compile(fn)
        out_compiled = compiled(*inps)
        self.assertEqual(out_eager, out_compiled)
        self.assertTrue(5 <= torch._inductor.metrics.generated_kernel_count <= 6)
        torch._inductor.metrics.reset()

        inps = (
            torch.rand(24, 30, device="cuda"),
            torch.rand(32, 30, device="cuda"),
            torch.rand(48, 32, device="cuda"),
            torch.rand(30, 24, device="cuda").t(),
            torch.rand(30, 32, device="cuda").t(),
            torch.rand(32, 48, device="cuda").t(),
        )
        out_compiled = compiled(*inps)
        out_eager = fn(*inps)
        self.assertEqual(out_eager, out_compiled)
        self.assertTrue(5 <= torch._inductor.metrics.generated_kernel_count <= 6)

    @requires_cuda
    @torch._dynamo.config.patch("automatic_dynamic_shapes", True)
    @torch._dynamo.config.patch("assume_static_by_default", True)
    @torch._inductor.config.patch("triton.autotune_at_compile_time", True)
    def test_dynamic_shapes_persistent_reduction_mixed_x_dim_cuda(self):
        def fn(x, y, z):
            return x.sum(1), y.mean(1), z.max(1)

        inps = (
            torch.rand(16, 128, device="cuda"),
            torch.rand(32, 128, device="cuda"),
            torch.rand(32, 256, device="cuda"),
        )
        torch._dynamo.mark_dynamic(inps[0], 0, min=1, max=256)
        torch._dynamo.mark_dynamic(inps[1], 0, min=1, max=256)
        torch._dynamo.mark_dynamic(inps[2], 0, min=1, max=256)
        out_eager = fn(*inps)
        out_compiled = torch.compile(fn)(*inps)

        self.assertEqual(out_eager, out_compiled)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if HAS_CPU or HAS_CUDA:
        run_tests(needs="filelock")
