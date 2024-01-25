import torch
import unittest
from torch.testing._internal.common_utils import TestCase, run_tests

class TestSetDefaultPinMemoryDevice(TestCase):

    def test_cpu_empty_pinned_memory_default(self):
        device_name = torch._C._get_default_pin_memory_device()
        self.assertEqual(device_name, 'cuda')

    def test_no_exception(self):
        torch._C._set_default_pin_memory_device('cuda')
        torch._C._get_default_pin_memory_device()

    def test_exception(self):
        with self.assertRaises(Exception):
            # Setting an unregistered device type.
            torch._C._set_default_pin_memory_device('abc')
            # Empty_input
            torch._C._set_default_pin_memory_device()

    @unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
    def test_cpu_empty_pinned_memory_on_cuda(self):
        torch._C._set_default_pin_memory_device('cuda')
        dummy = torch.empty((1), pin_memory=True)
        self.assertTrue(dummy.is_pinned())

    def test_with_privateuse1_rename(self):
        torch.utils.rename_privateuse1_backend('foo')
        torch._C._set_default_pin_memory_device('foo')
        device_name = torch._C._get_default_pin_memory_device()
        self.assertEqual(device_name, 'foo')

    def test_cpu_empty_pinned_memory_on_privateuse1(self):
        torch.utils.rename_privateuse1_backend('foo')
        torch._C._set_default_pin_memory_device('foo')
        with self.assertRaisesRegex(RuntimeError, "Please register PrivateUse1HooksInterface"):
            torch.empty((1), pin_memory=True)

if __name__ == '__main__':
    run_tests()
