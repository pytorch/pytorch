# Owner(s): ["module: intel"]

import unittest

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import IS_WINDOWS, run_tests, TestCase


# not torch.backends.xpu.is_avaliable
@unittest.skipIf(IS_WINDOWS, "XPU operators testing is disabled")
class TestXPUOperators(TestCase):
    def test_empty_tensor(self):
        # empty
        dtypes = [torch.float, torch.bfloat16]
        shapes = [(1, 3, 3, 3)]
        mformats = [torch.channels_last, torch.contiguous_format]

        def test_empty_tensor_(dtype, shape, mformat):
            xpu_tensor = torch.empty(
                shape, dtype=dtype, device=torch.device("xpu"), memory_format=mformat
            )
            cpu_tensor = torch.empty(shape, dtype=dtype, memory_format=mformat)
            self.assertEqual(xpu_tensor.device(), torch.device("xpu"))
            self.assertEqual(xpu_tensor.size(), cpu_tensor.size())
            self.assertEqual(xpu_tensor.stride(), cpu_tensor.stride())
            self.assertEqual(xpu_tensor.dtype(), cpu_tensor.dtype())

        for dtype in dtypes:
            for shape in shapes:
                for mformat in mformats:
                    test_empty_tensor_(dtype, shape, mformat)

    def test_empty_strided(self):
        # empty_strided
        dtypes = [torch.float, torch.bfloat16]
        shapes = [(1, 3, 3, 3)]
        strides = [(9, 9, 3, 1), (1, 1, 1, 1)]

        def test_empty_strided_(dtype, shape, stride):
            xpu_tensor = torch.empty_strided(
                shape, stride, dtype=dtype, device=torch.device("xpu")
            )
            cpu_tensor = torch.empty_strided(shape, stride, dtype=dtype)
            self.assertEqual(xpu_tensor.device(), torch.device("xpu"))
            self.assertEqual(xpu_tensor.size(), cpu_tensor.size())
            self.assertEqual(xpu_tensor.stride(), cpu_tensor.stride())
            self.assertEqual(xpu_tensor.dtype(), cpu_tensor.dtype())

        for dtype in dtypes:
            for shape in shapes:
                for stride in strides:
                    test_empty_strided_(dtype, shape, stride)


instantiate_device_type_tests(TestXPUOperators, globals(), only_for=("xpu",))

if __name__ == "__main__":
    run_tests()
