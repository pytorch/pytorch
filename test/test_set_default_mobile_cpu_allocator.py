# Owner(s): ["oncall: mobile"]

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

class TestSetDefaultMobileCPUAllocator(TestCase):
    def test_no_exception(self):
        torch._C._set_default_mobile_cpu_allocator()
        torch._C._unset_default_mobile_cpu_allocator()

    def test_exception(self):
        with self.assertRaises(Exception):
            torch._C._unset_default_mobile_cpu_allocator()

        with self.assertRaises(Exception):
            torch._C._set_default_mobile_cpu_allocator()
            torch._C._set_default_mobile_cpu_allocator()

        # Must reset to good state
        # For next test.
        torch._C._unset_default_mobile_cpu_allocator()

        with self.assertRaises(Exception):
            torch._C._set_default_mobile_cpu_allocator()
            torch._C._unset_default_mobile_cpu_allocator()
            torch._C._unset_default_mobile_cpu_allocator()

if __name__ == '__main__':
    run_tests()
