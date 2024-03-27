# Owner(s): ["module: inductor"]

import sys
import unittest

import torch

from torch.testing._internal.common_utils import IS_LINUX, skipIfRocm
from torch.testing._internal.inductor_utils import HAS_GPU

try:
    import triton  # noqa: F401
except ImportError:
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires triton")  # noqa: TRY200

from torch._dynamo.test_case import run_tests, TestCase
from torch._inductor import config
from torch._inductor.triton_heuristics import triton_config


class TestTritonHeuristics(TestCase):
    def test_triton_config(self):
        """
        Make sure block size does not exceed the maximum defined in inductor config.
        """
        cfg = triton_config([2048, 2], 64, 64)
        for label in "XYZ":
            key = f"{label}BLOCK"
            if key not in cfg.kwargs:
                continue
            self.assertTrue(cfg.kwargs[key] <= config.triton.max_block[label])

    def _test_artificial_zgrid(self):
        torch._inductor.config.cpp_wrapper = True

        def forward(primals_1, primals_2, primals_5):
            view = torch.ops.aten.reshape.default(primals_5, [-1, 4, 128])
            primals_5 = None
            permute = torch.ops.aten.permute.default(view, [0, 2, 1])
            clone = torch.ops.aten.clone.default(
                permute, memory_format=torch.contiguous_format
            )
            permute = None
            view_1 = torch.ops.aten.reshape.default(clone, [-1, 4])
            clone = None
            permute_1 = torch.ops.aten.permute.default(primals_1, [1, 0])
            primals_1 = None
            addmm = torch.ops.aten.addmm.default(primals_2, view_1, permute_1)
            primals_2 = None
            return addmm

        s0 = 727828
        s1 = 512

        args = [
            torch.rand([2, 4], device="cuda"),
            torch.rand([2], device="cuda"),
            torch.rand([s0, s1], device="cuda"),
        ]
        torch._dynamo.mark_dynamic(args[-1], 0)
        foo_c = torch.compile(forward)

        self.assertEqual(forward(*args), foo_c(*args))

        args = [
            torch.rand([2, 4], device="cuda"),
            torch.rand([2], device="cuda"),
            torch.rand([s0, s1], device="cuda"),
        ]
        self.assertEqual(forward(*args), foo_c(*args))

    @skipIfRocm
    def test_artificial_zgrid(self):
        self._test_artificial_zgrid()

    @config.patch("cpp_wrapper", True)
    def test_artificial_grid_cpp_wrapper(self):
        self._test_artificial_zgrid()

    @config.patch("triton.max_tiles", 3)
    def test_artificial_grid_max_tiles(self):
        with self.assertRaisesRegex(Exception, "Generated y grid"):
            self._test_artificial_zgrid()


if __name__ == "__main__":
    if IS_LINUX and HAS_GPU:
        run_tests()
