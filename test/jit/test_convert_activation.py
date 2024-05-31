# Owner(s): ["oncall: jit"]

import os
import sys
import unittest

from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing import FileCheck

try:
    import torchvision

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase

if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )

activations = [
    F.celu,
    F.elu,
    F.hardsigmoid,
    F.hardswish,
    F.hardtanh,
    F.leaky_relu,
    F.relu,
    F.relu6,
    F.rrelu,
    F.selu,
    F.silu,
]


class TestFunctionalToInplaceActivation(JitTestCase):
    def test_check_no_type_promotion(self):
        dtypes = [
            torch.bool,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.float32,
            torch.float64,
        ]
        # restore_mutation.h contains a mapping from activation operators
        # to whether they allow type conversion. Use this checking to
        # guard the mapping, and if any later change breaks the assumption
        # we need to update the mapping correspondingly.
        for activation, dtype in product(activations, dtypes):
            inp = torch.normal(0, 5, size=(4, 4)).to(dtype)
            try:
                out = activation(inp)
                self.assertEqual(dtype, out.dtype)
            except RuntimeError:
                # Skip the not implemented error
                pass

    def test_functional_to_inplace_activation(self):
        for activation in activations:

            def test_basic(x):
                y = x + 1
                z = activation(y)
                return z

            fn = torch.jit.script(test_basic)
            self.run_pass("inline", fn.graph)
            self.run_pass("constant_propagation", fn.graph)
            FileCheck().check(f"aten::{activation.__name__}(").run(fn.graph)
            self.run_pass("functional_to_inplace_activation", fn.graph)
            FileCheck().check_not(f"aten::{activation.__name__}(").run(fn.graph)
            FileCheck().check(f"aten::{activation.__name__}_").run(fn.graph)
            inp = torch.rand([2, 2])
            self.assertEqual(fn(inp), test_basic(inp))

    def test_no_functional_to_inplace(self):
        # inplace conversion should not happen because sigmoid may
        # perform type conversion
        def test1():
            y = torch.ones([2, 2])
            z = torch.sigmoid(y)
            return z

        fn = torch.jit.script(test1)
        self.run_pass("functional_to_inplace_activation", fn.graph)
        FileCheck().check_not("aten::sigmoid_").run(fn.graph)

        # inplace conversion should not happen because y is alias
        # the input x
        def test2(x):
            y = x[0]
            z = torch.relu(y)
            return z

        fn = torch.jit.script(test2)
        self.run_pass("functional_to_inplace_activation", fn.graph)
        FileCheck().check_not("aten::relu_").run(fn.graph)

        # inplace conversion should not happen because self.x is
        # at the global scope
        class Test3(nn.Module):
            def __init__(self, x):
                super().__init__()
                self.x = x

            def forward(self):
                y = torch.relu(self.x)
                return y

        fn = torch.jit.script(Test3(torch.rand([2, 2])).eval())
        self.run_pass("functional_to_inplace_activation", fn.graph)
        FileCheck().check_not("aten::relu_").run(fn.graph)

    @skipIfNoTorchVision
    def test_resnet18_correctness(self):
        model = torchvision.models.resnet18()
        frozen_model = torch.jit.freeze(torch.jit.script(model.eval()))
        (
            N,
            C,
            H,
            W,
        ) = (
            10,
            3,
            224,
            224,
        )
        inp = torch.randn(N, C, H, W)
        self.run_pass("functional_to_inplace_activation", frozen_model.graph)
        self.assertEqual(model(inp), frozen_model(inp))


class TestInplaceToFunctionalActivation(JitTestCase):
    def test_inplace_to_functional_activation(self):
        for activation in activations:

            def test_basic(x):
                y = x + 1
                activation(y, inplace=True)
                return y

            fn = torch.jit.script(test_basic)
            self.run_pass("inline", fn.graph)
            self.run_pass("constant_propagation", fn.graph)
            FileCheck().check(f"aten::{activation.__name__}_").run(fn.graph)
            self.run_pass("inplace_to_functional_activation", fn.graph)
            FileCheck().check_not(f"aten::{activation.__name__}_").run(fn.graph)
            FileCheck().check(f"aten::{activation.__name__}(").run(fn.graph)

        for activation in [
            torch.relu_,
            torch.sigmoid_,
            torch.tanh_,
        ]:

            def test_basic(x):
                y = x + 1
                activation(y)
                return y

            fn = torch.jit.script(test_basic)
            self.run_pass("inline", fn.graph)
            self.run_pass("constant_propagation", fn.graph)
            FileCheck().check(f"aten::{activation.__name__}").run(fn.graph)
            self.run_pass("inplace_to_functional_activation", fn.graph)
            FileCheck().check_not(f"aten::{activation.__name__}").run(fn.graph)
            FileCheck().check(f"aten::{activation.__name__[:-1]}(").run(fn.graph)

            inp = torch.rand([2, 2])
            self.assertEqual(fn(inp), test_basic(inp))

    @skipIfNoTorchVision
    def test_resnet18_correctness(self):
        model = torchvision.models.resnet18()
        frozen_model = torch.jit.freeze(torch.jit.script(model.eval()))
        (
            N,
            C,
            H,
            W,
        ) = (
            10,
            3,
            224,
            224,
        )
        inp = torch.randn(N, C, H, W)
        self.run_pass("inplace_to_functional_activation", frozen_model.graph)
        self.assertEqual(model(inp), frozen_model(inp))
