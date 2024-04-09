# Owner(s): ["module: unknown"]

import os
import shutil
import sys
from typing import Union
import tempfile
import unittest

import torch.testing._internal.common_utils as common
from torch.testing._internal.common_utils import IS_ARM64, TEST_CUDA
import torch
import torch.utils.cpp_extension
from torch.utils.cpp_extension import CUDA_HOME, ROCM_HOME


TEST_CUDA = TEST_CUDA and CUDA_HOME is not None
TEST_ROCM = TEST_CUDA and torch.version.hip is not None and ROCM_HOME is not None

def remove_build_path():
    if sys.platform == "win32":
        # Not wiping extensions build folder because Windows
        return
    default_build_root = torch.utils.cpp_extension.get_default_build_root()
    if os.path.exists(default_build_root):
        shutil.rmtree(default_build_root, ignore_errors=True)

@unittest.skipIf(IS_ARM64, "Does not work on arm")
@torch.testing._internal.common_utils.markDynamoStrictTest
class TestCppExtensionOpenRgistration(common.TestCase):
    """Tests Stream and Event with C++ extensions.
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

        # Load the fake device guard impl.
        cls.module = torch.utils.cpp_extension.load(
            name="custom_device_extension",
            sources=[
                "cpp_extensions/mtia_extension.cpp",
            ],
            extra_include_paths=["cpp_extensions"],
            extra_cflags=["-g"],
            verbose=True,
        )

    def test_stream_creation(self):
        stream = torch.Stream("mtia")
        self.assertEqual(stream.device_type, torch._C._autograd.DeviceType.MTIA)


if __name__ == "__main__":
    common.run_tests()
