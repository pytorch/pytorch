# Owner(s): ["module: mtia"]

import os
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
    TEST_XPU,
)
from torch.utils.cpp_extension import CUDA_HOME, ROCM_HOME


# define TEST_ROCM before changing TEST_CUDA
TEST_ROCM = TEST_CUDA and torch.version.hip is not None and ROCM_HOME is not None
TEST_CUDA = TEST_CUDA and CUDA_HOME is not None


@unittest.skipIf(
    IS_ARM64 or not IS_LINUX or TEST_CUDA or TEST_PRIVATEUSE1 or TEST_ROCM or TEST_XPU,
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
        torch.testing._internal.common_utils.remove_cpp_extensions_build_root()

    @classmethod
    def setUpClass(cls):
        torch.testing._internal.common_utils.remove_cpp_extensions_build_root()
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

    @skipIfTorchDynamo("Not a TorchDynamo suitable test")
    def test_default_generators(self):
        # Trigger lazy initialization first by calling current_stream()
        torch.mtia.current_stream()
        device_count = torch.mtia.device_count()

        # Verify the interface exists and is properly initialized
        self.assertTrue(hasattr(torch.mtia, "default_generators"))
        self.assertIsInstance(torch.mtia.default_generators, tuple)
        self.assertEqual(len(torch.mtia.default_generators), device_count)

        # Verify we can access generators by device index
        gen_0 = torch.mtia.default_generators[0]
        gen_1 = torch.mtia.default_generators[1]
        self.assertIsInstance(gen_0, torch.Generator)
        self.assertIsInstance(gen_1, torch.Generator)
        # Different devices should have different generator objects
        self.assertIsNot(gen_0, gen_1)

    @skipIfTorchDynamo("Not a TorchDynamo suitable test")
    def test_new_generator(self):
        # Verify we can create a generator via the hooks interface
        gen = torch.Generator(device="mtia:0")
        self.assertIsInstance(gen, torch.Generator)


if __name__ == "__main__":
    common.run_tests()
