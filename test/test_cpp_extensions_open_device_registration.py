# Owner(s): ["module: cpp-extensions"]

import os
import shutil
import sys
from typing import Union
import unittest

import torch.testing._internal.common_utils as common
from torch.testing._internal.common_utils import IS_ARM64
import torch
import torch.utils.cpp_extension
from torch.utils.cpp_extension import CUDA_HOME, ROCM_HOME


TEST_CUDA = torch.cuda.is_available() and CUDA_HOME is not None
TEST_CUDNN = False
TEST_ROCM = torch.cuda.is_available() and torch.version.hip is not None and ROCM_HOME is not None
if TEST_CUDA and torch.version.cuda is not None:  # the skip CUDNN test for ROCm
    CUDNN_HEADER_EXISTS = os.path.isfile(os.path.join(CUDA_HOME, "include/cudnn.h"))
    TEST_CUDNN = (
        TEST_CUDA and CUDNN_HEADER_EXISTS and torch.backends.cudnn.is_available()
    )


def remove_build_path():
    if sys.platform == "win32":
        # Not wiping extensions build folder because Windows
        return
    default_build_root = torch.utils.cpp_extension.get_default_build_root()
    if os.path.exists(default_build_root):
        shutil.rmtree(default_build_root, ignore_errors=True)


class DummyModule(object):

    @staticmethod
    def device_count() -> int:
        return 1

    @staticmethod
    def get_rng_state(device: Union[int, str, torch.device] = 'foo') -> torch.Tensor:
        # create a tensor using our custom device object.
        return torch.empty(4, 4, device="foo")

    @staticmethod
    def set_rng_state(new_state: torch.Tensor, device: Union[int, str, torch.device] = 'foo') -> None:
        pass


@unittest.skipIf(IS_ARM64, "Does not work on arm")
class TestCppExtensionOpenRgistration(common.TestCase):
    """Tests Open Device Registration with C++ extensions.
    """
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
        remove_build_path()
        cls.module = torch.utils.cpp_extension.load(
            name="custom_device_extension",
            sources=[
                "cpp_extensions/open_registration_extension.cpp",
            ],
            extra_include_paths=["cpp_extensions"],
            extra_cflags=["-g"],
            verbose=True,
        )

    @classmethod
    def tearDownClass(cls):
        remove_build_path()

    def test_open_device_registration(self):
        self.assertFalse(self.module.custom_add_called())

        # create a tensor using our custom device object.
        device = self.module.custom_device()

        x = torch.empty(4, 4, device=device)
        y = torch.empty(4, 4, device=device)

        # Check that our device is correct.
        self.assertTrue(x.device == device)
        self.assertFalse(x.is_cpu)

        self.assertFalse(self.module.custom_add_called())

        # calls out custom add kernel, registered to the dispatcher
        z = x + y

        # check that it was called
        self.assertTrue(self.module.custom_add_called())

        z_cpu = z.to(device='cpu')

        # Check that our cross-device copy correctly copied the data to cpu
        self.assertTrue(z_cpu.is_cpu)
        self.assertFalse(z.is_cpu)
        self.assertTrue(z.device == device)
        self.assertEqual(z, z_cpu)

        z2 = z_cpu + z_cpu

        # None of our CPU operations should call the custom add function.
        self.assertFalse(self.module.custom_add_called())

        # check generator registered befor use
        with self.assertRaisesRegex(RuntimeError,
                                    "Please register a generator to the PrivateUse1 dispatch key"):
            gen_ = torch.Generator(device=device)

        self.module.register_generator()

        gen = torch.Generator(device=device)
        self.assertTrue(gen.device == device)

        # generator can be registered only once
        with self.assertRaisesRegex(RuntimeError,
                                    "Only can register a generator to the PrivateUse1 dispatch key once"):
            self.module.register_generator()

        # check whether print tensor.type() meets the expectation
        torch.utils.rename_privateuse1_backend('foo')
        dtypes = {
            torch.bool: 'torch.foo.BoolTensor',
            torch.double: 'torch.foo.DoubleTensor',
            torch.float32: 'torch.foo.FloatTensor',
            torch.half: 'torch.foo.HalfTensor',
            torch.int32: 'torch.foo.IntTensor',
            torch.int64: 'torch.foo.LongTensor',
            torch.int8: 'torch.foo.CharTensor',
            torch.short: 'torch.foo.ShortTensor',
            torch.uint8: 'torch.foo.ByteTensor',
        }
        for tt, dt in dtypes.items():
            test_tensor = torch.empty(4, 4, dtype=tt, device=device)
            self.assertTrue(test_tensor.type() == dt)

    def test_open_device_random(self):
        torch.utils.rename_privateuse1_backend('foo')
        with self.assertRaisesRegex(RuntimeError, "Expected one of cpu"):
            torch._register_device_module('xxx', DummyModule)

        with self.assertRaisesRegex(RuntimeError, "torch has no module of"):
            with torch.random.fork_rng(device_type="foo"):
                pass
        torch._register_device_module('foo', DummyModule)

        with torch.random.fork_rng(device_type="foo"):
            pass

if __name__ == "__main__":
    common.run_tests()
