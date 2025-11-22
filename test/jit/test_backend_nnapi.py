# Owner(s): ["oncall: jit"]

import os
import sys
import unittest
from pathlib import Path

import torch
import torch._C
from torch.testing._internal.common_utils import (
    IS_FBCODE,
    raise_on_run_directly,
    skipIfTorchDynamo,
)


# hacky way to skip these tests in fbcode:
# during test execution in fbcode, test_nnapi is available during test discovery,
# but not during test execution. So we can't try-catch here, otherwise it'll think
# it sees tests but then fails when it tries to actually run them.
if not IS_FBCODE:
    from test_nnapi import TestNNAPI

    HAS_TEST_NNAPI = True
else:
    from torch.testing._internal.common_utils import TestCase as TestNNAPI

    HAS_TEST_NNAPI = False


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

"""
Unit Tests for Nnapi backend with delegate
Inherits most tests from TestNNAPI, which loads Android NNAPI models
without the delegate API.
"""
# First skip is needed for IS_WINDOWS or IS_MACOS to skip the tests.
torch_root = Path(__file__).resolve().parents[2]
lib_path = torch_root / "build" / "lib" / "libnnapi_backend.so"


@skipIfTorchDynamo("weird py38 failures")
@unittest.skipIf(
    not os.path.exists(lib_path),
    "Skipping the test as libnnapi_backend.so was not found",
)
@unittest.skipIf(IS_FBCODE, "test_nnapi.py not found")
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
        torch.ops.load_library(str(lib_path))

    # Override
    def call_lowering_to_nnapi(self, traced_module, args):
        compile_spec = {"forward": {"inputs": args}}
        return torch._C._jit_to_backend("nnapi", traced_module, compile_spec)

    def test_tensor_input(self):
        # Lower a simple module
        args = torch.tensor([[1.0, -1.0, 2.0, -2.0]]).unsqueeze(-1).unsqueeze(-1)
        module = torch.nn.PReLU()
        traced = torch.jit.trace(module, args)

        # Argument input is a single Tensor
        self.call_lowering_to_nnapi(traced, args)
        # Argument input is a Tensor in a list
        self.call_lowering_to_nnapi(traced, [args])

    # Test exceptions for incorrect compile specs
    def test_compile_spec_santiy(self):
        args = torch.tensor([[1.0, -1.0, 2.0, -2.0]]).unsqueeze(-1).unsqueeze(-1)
        module = torch.nn.PReLU()
        traced = torch.jit.trace(module, args)

        errorMsgTail = r"""
method_compile_spec should contain a Tensor or Tensor List which bundles input parameters: shape, dtype, quantization, and dimorder.
For input shapes, use 0 for run/load time flexible input.
method_compile_spec must use the following format:
{"forward": {"inputs": at::Tensor}} OR {"forward": {"inputs": c10::List<at::Tensor>}}"""

        # No forward key
        compile_spec = {"backward": {"inputs": args}}
        with self.assertRaisesRegex(
            RuntimeError,
            'method_compile_spec does not contain the "forward" key.' + errorMsgTail,
        ):
            torch._C._jit_to_backend("nnapi", traced, compile_spec)

        # No dictionary under the forward key
        compile_spec = {"forward": 1}
        with self.assertRaisesRegex(
            RuntimeError,
            'method_compile_spec does not contain a dictionary with an "inputs" key, '
            'under it\'s "forward" key.' + errorMsgTail,
        ):
            torch._C._jit_to_backend("nnapi", traced, compile_spec)

        # No inputs key (in the dictionary under the forward key)
        compile_spec = {"forward": {"not inputs": args}}
        with self.assertRaisesRegex(
            RuntimeError,
            'method_compile_spec does not contain a dictionary with an "inputs" key, '
            'under it\'s "forward" key.' + errorMsgTail,
        ):
            torch._C._jit_to_backend("nnapi", traced, compile_spec)

        # No Tensor or TensorList under the inputs key
        compile_spec = {"forward": {"inputs": 1}}
        with self.assertRaisesRegex(
            RuntimeError,
            'method_compile_spec does not contain either a Tensor or TensorList, under it\'s "inputs" key.'
            + errorMsgTail,
        ):
            torch._C._jit_to_backend("nnapi", traced, compile_spec)
        compile_spec = {"forward": {"inputs": [1]}}
        with self.assertRaisesRegex(
            RuntimeError,
            'method_compile_spec does not contain either a Tensor or TensorList, under it\'s "inputs" key.'
            + errorMsgTail,
        ):
            torch._C._jit_to_backend("nnapi", traced, compile_spec)

    def tearDown(self):
        # Change dtype back to default (Otherwise, other unit tests will complain)
        torch.set_default_dtype(self.default_dtype)


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
