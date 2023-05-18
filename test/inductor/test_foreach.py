# Owner(s): ["module: inductor"]

import sys
import unittest

import torch

import torch._inductor

from torch.testing._internal.common_utils import TEST_WITH_ROCM, TestCase

from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA

try:
    try:
        from .test_torchinductor import check_model_cuda, requires_cuda
    except ImportError:
        from test_torchinductor import check_model_cuda, requires_cuda
except (unittest.SkipTest, ImportError) as e:
    sys.stderr.write(f"{type(e)}: {e}\n")
    if __name__ == "__main__":
        sys.exit(0)
    raise


class ForeachTests(TestCase):
    check_model = check_model_cuda

    def setUp(self):
        super().setUp()
        torch._inductor.metrics.reset()

    def tearDown(self):
        super().tearDown()
        torch._inductor.metrics.reset()

    def test_single(self):
        def fn(a0, a1, b0, b1):
            return torch._foreach_add([a0, a1], [b0, b1])

        self.check_model(
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
    def test_scheduler_fusion(self):
        def fn(a0, a1, b0, b1):
            c = torch._foreach_add([a0, a1], [b0, b1])
            return c, torch._foreach_add([a0, a1], c)

        self.check_model(
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
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 0)

    @requires_cuda()
    def test_kernel_split_arg_limit(self):
        pass

    @requires_cuda()
    def test_non_foreach_consumer(self):
        pass

    @requires_cuda()
    def test_non_foreach_producer(self):
        pass

    @requires_cuda()
    def test_non_foreach_consumer_producer(self):
        pass

    @requires_cuda()
    def test_scalar_overloads(self):
        pass


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if (HAS_CPU or HAS_CUDA) and not TEST_WITH_ROCM:
        run_tests(needs="filelock")
