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
        for activation in [torch.relu, torch.sigmoid]:
            def test_basic():
                x = torch.ones([2, 2])
                y = activation(x)
                return y

            fn = torch.jit.script(test_basic)
            graph = fn.graph
            self.run_pass('functional_to_inplace_activation', graph)
            FileCheck().check(f"aten::{activation.__name__}_").run(graph)
            FileCheck().check_not(f"aten::{activation.__name__}(").run(graph)
            self.assertEqual(fn(), test_basic())

            model = torchvision.models.resnet18()
            frozen_model = torch.jit.freeze(torch.jit.script(model.eval()))
            N, C, H, W, = 10, 3, 224, 224
            inp = torch.randn(N, C, H, W)
            self.run_pass('inplace_to_functional_activation', frozen_model.graph)
            #self.run_pass('functional_to_inplace_activation', frozen_model.graph)
            print(frozen_model.graph)
            self.assertEqual(model(inp), frozen_model(inp))

class TestInplaceToFunctionalActivation(JitTestCase):
    def test_inplace_to_functional_activation(self):
        def test_relu():
            x = torch.tensor([2, 2])
            torch.relu_(x)
            return x

        fn = torch.jit.script(test_relu)
        graph = fn.graph
        self.run_pass('inplace_to_functional_activation', graph)
        FileCheck().check("aten::relu(").run(graph)
        FileCheck().check_not("aten::relu_").run(graph)
        self.assertEqual(fn(), test_relu())
