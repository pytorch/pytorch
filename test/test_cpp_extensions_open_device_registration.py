# Owner(s): ["module: cpp-extensions"]

import os
import unittest

import pytorch_openreg  # noqa: F401

import torch
import torch.testing._internal.common_utils as common
import torch.utils.cpp_extension


@unittest.skipIf(common.TEST_XPU, "XPU does not support cppextension currently")
@common.markDynamoStrictTest
class TestCppExtensionOpenRegistration(common.TestCase):
    """Tests Open Device Registration with C++ extensions."""

    module = None

    def setUp(self):
        super().setUp()

        # cpp extensions use relative paths. Those paths are relative to
        # this file, so we'll change the working directory temporarily
        self.old_working_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        assert self.module is not None

    def tearDown(self):
        super().tearDown()

        # return the working directory (see setUp)
        os.chdir(self.old_working_dir)

    @classmethod
    def setUpClass(cls):
        common.remove_cpp_extensions_build_root()

        cls.module = torch.utils.cpp_extension.load(
            name="custom_device_extension",
            sources=[
                "cpp_extensions/open_registration_extension.cpp",
            ],
            extra_include_paths=["cpp_extensions"],
            extra_cflags=["-g"],
            verbose=True,
        )

    def test_open_device_scalar_type_fallback(self):
        z_cpu = torch.Tensor([[0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]]).to(torch.int64)
        z = torch.triu_indices(3, 3, device="openreg")
        self.assertEqual(z_cpu, z)

    def test_open_device_tensor_type_fallback(self):
        # create tensors located in custom device
        x = torch.Tensor([[1, 2, 3], [2, 3, 4]]).to("openreg")
        y = torch.Tensor([1, 0, 2]).to("openreg")
        # create result tensor located in cpu
        z_cpu = torch.Tensor([[0, 2, 1], [1, 3, 2]])
        # Check that our device is correct.
        device = self.module.custom_device()
        self.assertTrue(x.device == device)
        self.assertFalse(x.is_cpu)

        # call sub op, which will fallback to cpu
        z = torch.sub(x, y)
        self.assertEqual(z_cpu, z)

        # call index op, which will fallback to cpu
        z_cpu = torch.Tensor([3, 1])
        y = torch.Tensor([1, 0]).long().to("openreg")
        z = x[y, y]
        self.assertEqual(z_cpu, z)

    def test_open_device_tensorlist_type_fallback(self):
        # create tensors located in custom device
        v_openreg = torch.Tensor([1, 2, 3]).to("openreg")
        # create result tensor located in cpu
        z_cpu = torch.Tensor([2, 4, 6])
        # create tensorlist for foreach_add op
        x = (v_openreg, v_openreg)
        y = (v_openreg, v_openreg)
        # Check that our device is correct.
        device = self.module.custom_device()
        self.assertTrue(v_openreg.device == device)
        self.assertFalse(v_openreg.is_cpu)

        # call _foreach_add op, which will fallback to cpu
        z = torch._foreach_add(x, y)
        self.assertEqual(z_cpu, z[0])
        self.assertEqual(z_cpu, z[1])

        # call _fused_adamw_ with undefined tensor.
        self.module.fallback_with_undefined_tensor()


if __name__ == "__main__":
    common.run_tests()
