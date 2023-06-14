# Owner(s): ["module: inductor"]

import sys
import unittest

import torch

import torch._inductor

from torch.testing._internal.common_utils import TEST_WITH_ROCM, TestCase

from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA

try:
    try:
        from .test_torchinductor import check_model, check_model_cuda, requires_cuda
    except ImportError:
        from test_torchinductor import check_model, check_model_cuda, requires_cuda
except (unittest.SkipTest, ImportError) as e:
    sys.stderr.write(f"{type(e)}: {e}\n")
    if __name__ == "__main__":
        sys.exit(0)
    raise


class ForeachTests(TestCase):
    check_model_cuda = check_model_cuda
    check_model_cpu = check_model
    check_kernel_count = True

    def setUp(self):
        super().setUp()
        torch._inductor.metrics.reset()

    def tearDown(self):
        super().tearDown()
        torch._inductor.metrics.reset()

    def _test_single(self):
        def fn(a0, a1, b0, b1):
            return torch._foreach_add([a0, a1], [b0, b1])

        self.check_model_cuda(
            fn,
            (
                torch.rand(10, 10, device="cuda:0"),
                torch.rand(20, 20, device="cuda:0"),
                torch.rand(10, 10, device="cuda:0"),
                torch.rand(20, 20, device="cuda:0"),
            ),
        )

    # called in test_cpp_wrapper.py
    test_foreach_cpp_wrapper = requires_cuda()(_test_single)

    @requires_cuda()
    def test_single(self):
        self._test_single()
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_cuda()
    def test_scheduler_fusion(self):
        def fn(a0, a1, b0, b1):
            c = torch._foreach_add([a0, a1], [b0, b1])
            return c, torch._foreach_add([a0, a1], c)

        self.check_model_cuda(
            fn,
            (
                torch.rand(10, 10, device="cuda:0"),
                torch.rand(20, 20, device="cuda:0"),
                torch.rand(10, 10, device="cuda:0"),
                torch.rand(20, 20, device="cuda:0"),
            ),
        )

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_cuda()
    def test_broadcasting(self):
        def fn(a0, a1, b0, b1):
            return torch._foreach_add([a0, a1], [b0, b1])

        fn_opt = torch._dynamo.optimize()(fn)

        inputs = (
            torch.rand(10, 1, device="cuda:0"),
            torch.rand(20, 20, device="cuda:0"),
            torch.rand(1, 10, device="cuda:0"),
            torch.rand(20, 20, device="cuda:0"),
        )
        actual = fn_opt(*inputs)
        expected = fn(*inputs)
        self.assertEqual(actual, expected)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_cuda()
    def test_singleton_lists(self):
        def fn(a0, b0):
            return torch._foreach_add([a0], [b0])

        self.check_model_cuda(
            fn,
            (
                torch.rand(10, 10, device="cuda:0"),
                torch.rand(10, 10, device="cuda:0"),
            ),
        )

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_cuda()
    def test_type_promotion(self):
        def fn(a0, a1, b0, b1):
            return torch._foreach_add([a0, a1], [b0, b1])

        fn_opt = torch._dynamo.optimize()(fn)

        max32 = torch.iinfo(torch.int32).max
        max64 = torch.iinfo(torch.int64).max
        inputs = (
            torch.randint(max32, (10, 10), device="cuda:0", dtype=torch.int32),
            torch.randint(max32, (20, 20), device="cuda:0", dtype=torch.int32),
            torch.randint(max32, (10, 10), device="cuda:0", dtype=torch.int32),
            torch.randint(max64, (20, 20), device="cuda:0", dtype=torch.int64),
        )
        actual = fn_opt(*inputs)
        expected = fn(*inputs)
        self.assertEqual(actual, expected)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 2)

    @requires_cuda()
    def test_kernel_split_arg_limit(self):
        def fn(a, b):
            return torch._foreach_add(a, b)

        fn_opt = torch._dynamo.optimize()(fn)

        max_args = 370
        max_list_len = (max_args // 3) + 1
        inputs = (
            [torch.rand(10, 10, device="cuda:0") for _ in range(max_list_len)],
            [torch.rand(10, 10, device="cuda:0") for _ in range(max_list_len)],
        )

        actual = fn_opt(*inputs)
        expected = fn(*inputs)
        self.assertEqual(actual, expected)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 2)

    @requires_cuda()
    def test_fusion_duplicate_buffer(self):
        def fn(a0, a1, b0, b1):
            c = torch._foreach_add([a0, a1], [b0, b1])
            return torch._foreach_add([a0, b0], [c[0], c[0]])

        self.check_model_cuda(
            fn,
            (
                torch.rand(10, 10, device="cuda:0"),
                torch.rand(20, 20, device="cuda:0"),
                torch.rand(10, 10, device="cuda:0"),
                torch.rand(20, 20, device="cuda:0"),
            ),
        )

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 2)

    @requires_cuda()
    def test_non_foreach_consumer(self):
        def fn(a0, a1, b0, b1):
            c = torch._foreach_add([a0, a1], [b0, b1])
            return torch.mul(c[0], a0)

        self.check_model_cuda(
            fn,
            (
                torch.rand(10, 10, device="cuda:0"),
                torch.rand(20, 20, device="cuda:0"),
                torch.rand(10, 10, device="cuda:0"),
                torch.rand(20, 20, device="cuda:0"),
            ),
        )

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 2)

    @requires_cuda()
    def test_non_foreach_producer(self):
        def fn(a0, a1, b0, b1):
            c0 = torch.mul(a0, b0)
            c1 = torch.mul(a1, b1)
            return torch._foreach_add([a0, a1], [c0, c1])

        self.check_model_cuda(
            fn,
            (
                torch.rand(10, 10, device="cuda:0"),
                torch.rand(20, 20, device="cuda:0"),
                torch.rand(10, 10, device="cuda:0"),
                torch.rand(20, 20, device="cuda:0"),
            ),
        )

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 3)

    @requires_cuda()
    def test_non_foreach_consumer_producer(self):
        def fn(a0, a1, b0, b1):
            c0 = torch.mul(a0, b0)
            c1 = torch.mul(a1, b1)
            d = torch._foreach_add([a0, a1], [c0, c1])
            e0 = torch.mul(d[0], a0)
            e1 = torch.mul(d[1], a1)
            return [e0, e1]

        self.check_model_cuda(
            fn,
            (
                torch.rand(10, 10, device="cuda:0"),
                torch.rand(20, 20, device="cuda:0"),
                torch.rand(10, 10, device="cuda:0"),
                torch.rand(20, 20, device="cuda:0"),
            ),
        )

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 5)

    @requires_cuda()
    @torch._dynamo.config.patch("dynamic_shapes", True)
    @torch._dynamo.config.patch("automatic_dynamic_shapes", False)
    @torch._dynamo.config.patch("assume_static_by_default", False)
    def test_dynamic_shapes_fallback(self):
        def fn(a0, a1, b0, b1):
            return torch._foreach_add([a0, a1], [b0, b1])

        inputs = (
            torch.rand(10, 10, device="cuda:0"),
            torch.rand(20, 20, device="cuda:0"),
            torch.rand(10, 10, device="cuda:0"),
            torch.rand(20, 20, device="cuda:0"),
        )

        self.check_model_cuda(fn, inputs)

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 2)

    def test_cpu_cpp_fallback(self):
        def fn(a0, a1, b0, b1):
            return torch._foreach_add([a0, a1], [b0, b1])

        inputs = (
            torch.rand(10, 10, device="cpu"),
            torch.rand(20, 20, device="cpu"),
            torch.rand(10, 10, device="cpu"),
            torch.rand(20, 20, device="cpu"),
        )

        self.check_model_cpu(fn, inputs)

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 2)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if (HAS_CPU or HAS_CUDA) and not TEST_WITH_ROCM:
        run_tests(needs="filelock")
