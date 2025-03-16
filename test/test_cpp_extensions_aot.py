# Owner(s): ["module: cpp-extensions"]

import os
import re
import subprocess
import sys
import unittest
from itertools import repeat
from pathlib import Path
from typing import get_args, get_origin, Union

import torch
import torch.backends.cudnn
import torch.testing._internal.common_utils as common
import torch.utils.cpp_extension
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import (
    IS_WINDOWS,
    shell,
    skipIfTorchDynamo,
    TEST_XPU,
    xfailIfTorchDynamo,
)


try:
    import pytest

    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

# TODO: Rewrite these tests so that they can be collected via pytest without
# using run_test.py
try:
    if HAS_PYTEST:
        cpp_extension = pytest.importorskip("torch_test_cpp_extension.cpp")
        maia_extension = pytest.importorskip("torch_test_cpp_extension.maia")
        rng_extension = pytest.importorskip("torch_test_cpp_extension.rng")
    else:
        import torch_test_cpp_extension.cpp as cpp_extension
        import torch_test_cpp_extension.maia as maia_extension
        import torch_test_cpp_extension.rng as rng_extension
except ImportError as e:
    raise RuntimeError(
        "test_cpp_extensions_aot.py cannot be invoked directly. Run "
        "`python run_test.py -i test_cpp_extensions_aot_ninja` instead."
    ) from e


@torch.testing._internal.common_utils.markDynamoStrictTest
@torch.testing._internal.common_utils.markDynamoStrictTest
class TestMAIATensor(common.TestCase):
    # For all tests inside this class, check
    # pytorch/test/cpp_extensions/maia_extension.cpp
    # when seeing C++ side errors. Ignore
    # pytorch/aten/src/ATen/test/extension_backend_test.cpp.

    def test_unregistered(self):
        torch.arange(0, 10, device="cpu")
        with self.assertRaisesRegex(RuntimeError, "Could not run"):
            torch.arange(0, 10, device="maia")

    @skipIfTorchDynamo("dynamo cannot model maia device")
    def test_zeros(self):
        a = torch.empty(5, 5, device="cpu")
        self.assertEqual(a.device, torch.device("cpu"))

        b = torch.empty(5, 5, device="maia")
        self.assertEqual(b.device, torch.device("maia", 0))
        self.assertEqual(maia_extension.get_test_int(), 0)
        self.assertEqual(torch.get_default_dtype(), b.dtype)

        c = torch.empty((5, 5), dtype=torch.int64, device="maia")
        self.assertEqual(maia_extension.get_test_int(), 0)
        self.assertEqual(torch.int64, c.dtype)

    def test_add(self):
        a = torch.empty(5, 5, device="maia", requires_grad=True)
        self.assertEqual(maia_extension.get_test_int(), 0)

        b = torch.empty(5, 5, device="maia")
        self.assertEqual(maia_extension.get_test_int(), 0)

        a + b
        self.assertEqual(maia_extension.get_test_int(), 1)

    def test_conv_backend_override(self):
        # To simplify tests, we use 4d input here to avoid doing view4d( which
        # needs more overrides) in _convolution.
        input = torch.empty(2, 4, 10, 2, device="maia", requires_grad=True)
        weight = torch.empty(6, 4, 2, 2, device="maia", requires_grad=True)
        bias = torch.empty(6, device="maia")

        # Make sure forward is overriden
        out = torch.nn.functional.conv2d(input, weight, bias, 2, 0, 1, 1)
        self.assertEqual(maia_extension.get_test_int(), 2)
        self.assertEqual(out.shape[0], input.shape[0])
        self.assertEqual(out.shape[1], weight.shape[0])

        # Make sure backward is overriden
        # Double backward is dispatched to _convolution_double_backward.
        # It is not tested here as it involves more computation/overrides.
        grad = torch.autograd.grad(out, input, out, create_graph=True)
        self.assertEqual(maia_extension.get_test_int(), 3)
        self.assertEqual(grad[0].shape, input.shape)

    #def test_matmul_autocast_default(self):
    #    # Use default lower precision dtype.
    #    x = torch.empty((2, 4), dtype=torch.float, device="maia")
    #    w = torch.empty((4, 2), dtype=torch.float, device="maia")
    #    with torch.autocast(device_type="maia"):
    #        self.assertTrue(torch.is_autocast_enabled("maia"))
    #        y = torch.ops.aten.matmul(x, w)
    #        self.assertEqual(y.dtype, torch.bfloat16)
    #        self.assertEqual(y.shape, (2, 2))

    def test_matmul_autocast_float16(self):
        # Ensure we can change low precision dtype.
        x = torch.empty((2, 4), dtype=torch.float, device="maia")
        w = torch.empty((4, 2), dtype=torch.float, device="maia")
        with torch.autocast(device_type="maia", dtype=torch.float16):
            self.assertTrue(torch.is_autocast_enabled("maia"))
            y = torch.ops.aten.matmul(x, w)
            self.assertEqual(y.dtype, torch.float16)
            self.assertEqual(y.shape, (2, 2))


if __name__ == "__main__":
    common.run_tests()
