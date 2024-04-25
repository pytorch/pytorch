# Owner(s): ["module: mtia"]

import os
import shutil
import sys
import tempfile
import unittest

import torch
import torch.testing._internal.common_utils as common
import torch.utils.cpp_extension
from torch.testing._internal.common_utils import (
    IS_ARM64,
    IS_LINUX,
    skipIfTorchDynamo,
    TEST_CUDA,
    TEST_PRIVATEUSE1,
)
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


@unittest.skipIf(
    IS_ARM64 or not IS_LINUX or TEST_CUDA or TEST_PRIVATEUSE1,
    "Only on linux platform and mutual exclusive to other backends",
)
@torch.testing._internal.common_utils.markDynamoStrictTest
class TestCppExtensionMTIABackend(common.TestCase):
    """Tests MTIA backend with C++ extensions."""

    module = None

    def setUp(self):
        super().setUp()
        # cpp extensions use relative paths. Those paths are relative to
        # this file, so we'll change the working directory temporarily
        self.old_working_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

    def tearDown(self):
        super().tearDown()
        # return the working directory (see setUp)
        os.chdir(self.old_working_dir)

    @classmethod
    def tearDownClass(cls):
        remove_build_path()

    @classmethod
    def setUpClass(cls):
        remove_build_path()
        build_dir = tempfile.mkdtemp()
        # Load the fake device guard impl.
        cls.module = torch.utils.cpp_extension.load(
            name="mtia_extension",
            sources=["cpp_extensions/mtia_extension.cpp"],
            build_directory=build_dir,
            extra_include_paths=[
                "cpp_extensions",
                "path / with spaces in it",
                "path with quote'",
            ],
            is_python_module=False,
            verbose=True,
        )

    @skipIfTorchDynamo("Not a TorchDynamo suitable test")
    def test_get_device_module(self):
        device = torch.device("mtia:0")
        default_stream = torch.get_device_module(device).current_stream()
        self.assertEqual(
            default_stream.device_type, int(torch._C._autograd.DeviceType.MTIA)
        )
        print(torch._C.Stream.__mro__)
        print(torch.cuda.Stream.__mro__)

    @skipIfTorchDynamo("Not a TorchDynamo suitable test")
    def test_stream_basic(self):
        default_stream = torch.mtia.current_stream()
        user_stream = torch.mtia.Stream()
        self.assertEqual(torch.mtia.current_stream(), default_stream)
        self.assertNotEqual(default_stream, user_stream)
        # Check mtia_extension.cpp, default stream id starts from 0.
        self.assertEqual(default_stream.stream_id, 0)
        self.assertNotEqual(user_stream.stream_id, 0)
        with torch.mtia.stream(user_stream):
            self.assertEqual(torch.mtia.current_stream(), user_stream)
        self.assertTrue(user_stream.query())
        default_stream.synchronize()
        self.assertTrue(default_stream.query())

    @skipIfTorchDynamo("Not a TorchDynamo suitable test")
    def test_stream_context(self):
        mtia_stream_0 = torch.mtia.Stream(device="mtia:0")
        mtia_stream_1 = torch.mtia.Stream(device="mtia:0")
        print(mtia_stream_0)
        print(mtia_stream_1)
        with torch.mtia.stream(mtia_stream_0):
            current_stream = torch.mtia.current_stream()
            msg = f"current_stream {current_stream} should be {mtia_stream_0}"
            self.assertTrue(current_stream == mtia_stream_0, msg=msg)

        with torch.mtia.stream(mtia_stream_1):
            current_stream = torch.mtia.current_stream()
            msg = f"current_stream {current_stream} should be {mtia_stream_1}"
            self.assertTrue(current_stream == mtia_stream_1, msg=msg)

    @skipIfTorchDynamo("Not a TorchDynamo suitable test")
    def test_stream_context_different_device(self):
        device_0 = torch.device("mtia:0")
        device_1 = torch.device("mtia:1")
        mtia_stream_0 = torch.mtia.Stream(device=device_0)
        mtia_stream_1 = torch.mtia.Stream(device=device_1)
        print(mtia_stream_0)
        print(mtia_stream_1)
        orig_current_device = torch.mtia.current_device()
        with torch.mtia.stream(mtia_stream_0):
            current_stream = torch.mtia.current_stream()
            self.assertTrue(torch.mtia.current_device() == device_0.index)
            msg = f"current_stream {current_stream} should be {mtia_stream_0}"
            self.assertTrue(current_stream == mtia_stream_0, msg=msg)
        self.assertTrue(torch.mtia.current_device() == orig_current_device)
        with torch.mtia.stream(mtia_stream_1):
            current_stream = torch.mtia.current_stream()
            self.assertTrue(torch.mtia.current_device() == device_1.index)
            msg = f"current_stream {current_stream} should be {mtia_stream_1}"
            self.assertTrue(current_stream == mtia_stream_1, msg=msg)
        self.assertTrue(torch.mtia.current_device() == orig_current_device)

    @skipIfTorchDynamo("Not a TorchDynamo suitable test")
    def test_device_context(self):
        device_0 = torch.device("mtia:0")
        device_1 = torch.device("mtia:1")
        with torch.mtia.device(device_0):
            self.assertTrue(torch.mtia.current_device() == device_0.index)

        with torch.mtia.device(device_1):
            self.assertTrue(torch.mtia.current_device() == device_1.index)


if __name__ == "__main__":
    common.run_tests()
