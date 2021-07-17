import os
import sys
import unittest

import torch
import torch._C
from pathlib import Path
from test_nnapi import TestNNAPI

from torch.testing._internal.common_utils import (
    IS_FBCODE,
    IS_MACOS,
    IS_SANDCASTLE,
    IS_WINDOWS,
    TEST_WITH_ROCM,
    skipIfRocm,
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
Inherits most tests from TestNNAPI, which loads Android NNAPI models
without the delegate API.
"""
# This is needed for IS_WINDOWS or IS_MACOS to skip the tests.
@unittest.skipIf(TEST_WITH_ROCM or IS_SANDCASTLE or IS_WINDOWS or IS_MACOS or IS_FBCODE,
                 "Non-portable load_library call used in test")
class TestNnapiBackend(TestNNAPI):
    def setUp(self):
        super().setUp()

        # Save default dtype
        module = torch.nn.PReLU()
        self.default_dtype = module.weight.dtype
        # Change dtype to float32 (since a different unit test changed dtype to float64,
        # which is not supported by the Android NNAPI delegate)
        # Float32 should typically be the default in other files.
        torch.set_default_dtype(torch.float32)

        # Load nnapi delegate library
        torch_root = Path(__file__).resolve().parent.parent.parent
        p = torch_root / 'build' / 'lib' / 'libnnapi_backend.so'
        torch.ops.load_library(str(p))

        # Disable execution tests, only test lowering modules
        # TODO: Re-enable execution tests after the Nnapi delegate is complete
        super().set_can_run_nnapi(False)

    # Override
    def call_convert_to_nnapi(self, traced_module, args):
        compile_spec = {"forward": {"inputs": args}}
        return torch._C._jit_to_backend("nnapi", traced_module, compile_spec)

    def test_tensor_input(self):
        # Lower a simple module
        args = torch.tensor([[1.0, -1.0, 2.0, -2.0]]).unsqueeze(-1).unsqueeze(-1)
        module = torch.nn.PReLU()
        traced = torch.jit.trace(module, args)

        # Argument input is a single Tensor
        self.call_convert_to_nnapi(traced, args)
        # Argument input is a Tensor in a list
        self.call_convert_to_nnapi(traced, [args])

    def tearDown(self):
        # Change dtype back to default (Otherwise, other unit tests will complain)
        torch.set_default_dtype(self.default_dtype)
