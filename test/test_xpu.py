# Owner(s): ["module: xpu"]

import unittest

import torch
from torch.testing._internal.common_utils import TestCase, \
    run_tests, IS_WINDOWS
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
)

# not torch.backends.xpu.is_avaliable
@unittest.skipIf(IS_WINDOWS, "XPU operators testing is disabled")
class TestXPUOperators(TestCase):
    def test_tensor_factory(self):
        # empty
        dtypes = [torch.float, torch.bfloat16]
        shapes = [(1, 3, 3, 3)]
        mformats = [torch.channels_last, torch.contiguous_format]

        def test_empty(dtype, shape, mformat):
            xpu_tensor = torch.empty(shape, dtype=dtype, device= torch.device('xpu'), memory_format=mformat)
            cpu_tensor = torch.empty(shape, dtype=dtype, memory_format=mformat)
            self.assertEqual(xpu_tensor.device(), torch.device('xpu'))
            self.assertEqual(xpu_tensor.size(), cpu_tensor.size())
            self.assertEqual(xpu_tensor.stride(), cpu_tensor.stride())
            self.assertEqual(xpu_tensor.dtype(), cpu_tensor.dtype())

        for dtype in dtypes:
            for shape in shapes:
                for mformat in memory_formats:
                    test_empty(dtype, shape, mformat)

        # empty_strided
        dtypes = [torch.float, torch.bfloat16]
        shapes = [(1, 3, 3, 3)]
        strides = [(9, 9, 3, 1), (1, 1, 1, 1)]

        def test_empty_strided(dtype, shape, stride):
            xpu_tensor = torch.empty_strided(shape, stride, dtype=dtype, device= torch.device('xpu'))
            cpu_tensor = torch.empty_strided(shape, stride, dtype=dtype)
            self.assertEqual(xpu_tensor.device(), torch.device('xpu'))
            self.assertEqual(xpu_tensor.size(), cpu_tensor.size())
            self.assertEqual(xpu_tensor.stride(), cpu_tensor.stride())
            self.assertEqual(xpu_tensor.dtype(), cpu_tensor.dtype())

        for dtype in dtypes:
            for shape in shapes:
                for stride in strides:
                    test_empty_strided(dtype, shape, stride)

instantiate_device_type_tests(TestXPUOperators, globals(), only_for=('xpu',))

if __name__ == '__main__':
    run_tests()
