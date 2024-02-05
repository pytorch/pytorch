# Owner(s): ["module: intel"]

import sys
import unittest

import torch
from torch.testing._internal.common_utils import IS_WINDOWS, NoTest, run_tests, TEST_XPU, TestCase
from torch.testing._internal.common_device_type import instantiate_device_type_tests

if not TEST_XPU:
    print("XPU not available, skipping tests", file=sys.stderr)
    TestCase = NoTest  # noqa: F811

TEST_MULTIXPU = torch.xpu.device_count() > 1


class TestXpu(TestCase):
    def test_device_behavior(self):
        current_device = torch.xpu.current_device()
        torch.xpu.set_device(current_device)
        self.assertEqual(current_device, torch.xpu.current_device())

    @unittest.skipIf(not TEST_MULTIXPU, "only one GPU detected")
    def test_multi_device_behavior(self):
        current_device = torch.xpu.current_device()
        target_device = (current_device + 1) % torch.xpu.device_count()

        with torch.xpu.device(target_device):
            self.assertEqual(target_device, torch.xpu.current_device())
        self.assertEqual(current_device, torch.xpu.current_device())

        with torch.xpu._DeviceGuard(target_device):
            self.assertEqual(target_device, torch.xpu.current_device())
        self.assertEqual(current_device, torch.xpu.current_device())

    def test_get_device_properties(self):
        current_device = torch.xpu.current_device()
        device_properties = torch.xpu.get_device_properties(current_device)
        self.assertEqual(device_properties, torch.xpu.get_device_properties(None))
        self.assertEqual(device_properties, torch.xpu.get_device_properties())

        device_name = torch.xpu.get_device_name(current_device)
        self.assertEqual(device_name, torch.xpu.get_device_name(None))
        self.assertEqual(device_name, torch.xpu.get_device_name())

        device_capability = torch.xpu.get_device_capability(current_device)
        self.assertTrue(device_capability["max_work_group_size"] > 0)
        self.assertTrue(device_capability["max_num_sub_groups"] > 0)


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
