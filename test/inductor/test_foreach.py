# Owner(s): ["module: inductor"]

import sys
import unittest

import torch

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

    @requires_cuda()
    def test_single_foreach(self):
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

    @requires_cuda()
    def test_foreach_ir_fusion(self):
        def fn(a0, a1, b0, b1):
            c = torch._foreach_add([a0, a1], [b0, b1])
            return torch._foreach_add([a0, a1], c)

        self.check_model(
            fn,
            (
                torch.rand(10, 10, device="cuda:0"),
                torch.rand(20, 20, device="cuda:0"),
                torch.rand(10, 10, device="cuda:0"),
                torch.rand(20, 20, device="cuda:0"),
            ),
        )

    @requires_cuda()
    def test_foreach_scheduler_fusion(self):
        def fn(a0, a1, b0, b1):
            c = torch._foreach_add([a0, a1], [b0, b1])
            return c, torch._foreach_add(
                [a0, a1], c
            )  # return c forces it to be realized

        self.check_model(
            fn,
            (
                torch.rand(10, 10, device="cuda:0"),
                torch.rand(20, 20, device="cuda:0"),
                torch.rand(10, 10, device="cuda:0"),
                torch.rand(20, 20, device="cuda:0"),
            ),
        )

    @requires_cuda()
    def test_foreach_broadcasting(self):
        pass

    @requires_cuda()
    def test_foreach_type_promotion(self):
        pass

    @requires_cuda()
    def test_kernel_split_arg_limit(self):
        pass

    @requires_cuda()
    def test_foreach_non_foreach_consumer(self):
        pass

    @requires_cuda()
    def test_foreach_non_foreach_producer(self):
        pass


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if (HAS_CPU or HAS_CUDA) and not TEST_WITH_ROCM:
        run_tests(needs="filelock")
