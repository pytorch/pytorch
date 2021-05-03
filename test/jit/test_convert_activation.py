import os
import sys

import torch
import torchvision
from torch.testing import FileCheck

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase

if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_jit.py TESTNAME\n\n"
                       "instead.")

class TestFunctionalToInplaceActivation(JitTestCase):
    def test_functional_to_inplace_activation(self):
        for activation in [
            torch.nn.functional.hardsigmoid,
            torch.nn.functional.hardtanh,
            torch.nn.functional.hardswish,
            torch.nn.functional.relu,
            torch.nn.functional.relu6,
            torch.sigmoid,
            torch.tanh,
        ]:
            def test_basic(x):
                y = x + 1
                z = activation(y)
                return z

            fn = torch.jit.script(test_basic)
            self.run_pass("inline", fn.graph)
            self.run_pass("constant_propagation", fn.graph)
            FileCheck().check(f"aten::{activation.__name__}(").run(fn.graph)
            self.run_pass('functional_to_inplace_activation', fn.graph)
            FileCheck().check_not(f"aten::{activation.__name__}(").run(fn.graph)
            FileCheck().check(f"aten::{activation.__name__}_").run(fn.graph)

            inp = torch.rand([2, 2])
            self.assertEqual(fn(inp), test_basic(inp))

        model = torchvision.models.resnet18()
        frozen_model = torch.jit.freeze(torch.jit.script(model.eval()))
        N, C, H, W, = 10, 3, 224, 224
        inp = torch.randn(N, C, H, W)
        self.run_pass('functional_to_inplace_activation', frozen_model.graph)
        self.assertEqual(model(inp), frozen_model(inp))

class TestInplaceToFunctionalActivation(JitTestCase):
    def test_inplace_to_functional_activation(self):
        for activation in [
            torch.nn.functional.hardsigmoid,
            torch.nn.functional.hardtanh,
            torch.nn.functional.hardswish,
            torch.nn.functional.relu,
            torch.nn.functional.relu6,
        ]:
            def test_basic(x):
                y = x + 1
                activation(y, inplace=True)
                return y

            fn = torch.jit.script(test_basic)
            self.run_pass("inline", fn.graph)
            self.run_pass("constant_propagation", fn.graph)
            FileCheck().check(f"aten::{activation.__name__}_").run(fn.graph)
            self.run_pass('inplace_to_functional_activation', fn.graph)
            FileCheck().check_not(f"aten::{activation.__name__}_").run(fn.graph)
            FileCheck().check(f"aten::{activation.__name__}(").run(fn.graph)

            inp = torch.rand([2, 2])
            self.assertEqual(fn(inp), test_basic(inp))

        model = torchvision.models.resnet18()
        frozen_model = torch.jit.freeze(torch.jit.script(model.eval()))
        N, C, H, W, = 10, 3, 224, 224
        inp = torch.randn(N, C, H, W)
        self.run_pass('inplace_to_functional_activation', frozen_model.graph)
        self.assertEqual(model(inp), frozen_model(inp))
