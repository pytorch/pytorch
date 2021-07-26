from torch.testing._internal.jit_utils import JitTestCase
import ctypes
import os
import sys
import unittest

import torch
import torch._C
from pathlib import Path

from torch.testing._internal.common_utils import (
    TEST_WITH_ROCM,
    skipIfRocm,
    TEST_WITH_ASAN
)
# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )

"""
Unit Tests for Nnapi backend with delegate
"""
# This is needed for IS_WINDOWS or IS_MACOS to skip the tests.
@unittest.skipIf(not os.path.exists(Path(__file__).resolve().parent.parent.parent
                 / 'build' / 'lib' / 'libnnapi_backend.so'),
                 "Skipping the test as libnnapi_backend.so was not found")
class NnapiBackendPReLUTest(JitTestCase):
    """
    Test lowering a simple PRelU module to Nnapi backend.
    TODO: After Nnapi delegate is finished T91991928,
    add tests for running the module (currently only lowers the model)
    """
    def setUp(self):
        super().setUp()
        # Load nnapi delegate library
        torch_root = Path(__file__).resolve().parent.parent.parent
        p = torch_root / 'build' / 'lib' / 'libnnapi_backend.so'
        torch.ops.load_library(str(p))

        # Nnapi set up
        # Avoid saturation in fbgemm
        torch.backends.quantized.engine = 'qnnpack'

        libneuralnetworks_path = os.environ.get("LIBNEURALNETWORKS_PATH")
        if libneuralnetworks_path:
            ctypes.cdll.LoadLibrary(libneuralnetworks_path)
            print("Will attempt to run NNAPI models.")
            self.can_run_nnapi = True
        else:
            self.can_run_nnapi = False

    def test_conversion(self):
        # Save default dtype
        module = torch.nn.PReLU()
        default_dtype = module.weight.dtype
        # Change dtype to float32 (since a different unit test changed dtype to float64,
        # which is not supported by the Android NNAPI delegate)
        # Float32 should typically be the default in other files.
        torch.set_default_dtype(torch.float32)
        module = torch.nn.PReLU()
        args = torch.tensor([[1.0, -1.0, 2.0, -2.0]]).unsqueeze(-1).unsqueeze(-1)

        # Trace and lower PreLu module
        traced = torch.jit.trace(module, args)
        compile_spec = {"forward": {"inputs": args}}
        nnapi_model = torch._C._jit_to_backend("nnapi", traced, compile_spec)

        # Change dtype back to the default
        torch.set_default_dtype(default_dtype)

# First skip is needed for IS_WINDOWS or IS_MACOS to skip the tests.
# Second skip is because ASAN is currently causing an error.
# It is still unclear how to resolve this. T95764916
@unittest.skipIf(not os.path.exists(Path(__file__).resolve().parent.parent.parent
                 / 'build' / 'lib' / 'libnnapi_backend.so'),
                 "Skipping the test as libnnapi_backend.so was not found")
@unittest.skipIf(TEST_WITH_ASAN, "Unresolved bug with ASAN")
class TestNnapiBackend(JitTestCase):
    """
    This class wraps all Nnapi test classes
    TODO: Add more comprehensive Nnapi tests (currently only a basic one)
    """
    def __init__(self, name):
        super().__init__(name)
        self.prelu_test = NnapiBackendPReLUTest(name)

    def setUp(self):
        if not TEST_WITH_ROCM:
            self.prelu_test.setUp()

    @skipIfRocm
    def test_conversion(self):
        self.prelu_test.test_conversion()
