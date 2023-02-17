from torch.testing._internal.common_utils import run_tests, TestCase
from functorch.experimental.control_flow import cond
from torch._export import export
import torch
import unittest

class TestExport(TestCase):
    @unittest.skip("dynamo failure -> RuntimeError: Could not infer dtype of SymBool")
    def test_export_cond(self):
        def true_fn(x):
            return x.sin()
        def false_fn(x):
            return x.cos()
        def foo(x):
            return cond(torch.tensor(x.shape[0] > 4), true_fn, false_fn, [x])

        exported_program = export(foo, (torch.ones(6, 4, requires_grad=True),))
        print(exported_program.graph_module.graph)

    def test_export_simple_model_with_attr(self):
        class Foo(torch.nn.Module):
            def __init__(self, float_val):
                super().__init__()
                self.float_val = float_val

            def forward(self, x):
                y = x + self.float_val
                return y.cos()

        inp = (torch.ones(6, 4, requires_grad=True),)
        mod = Foo(0.5)

        exported_program = export(mod, inp)
        self.assertEqual(exported_program.fw_module(*inp)[0], mod(*inp))

    def test_export_simple_model(self):
        class Foo(torch.nn.Module):
            def __init__(self, float_val):
                super().__init__()
                self.float_val = float_val

            def forward(self, x):
                return x.cos()

        inp = (torch.ones(6, 4, requires_grad=True),)
        mod = Foo(0.5)

        exported_program = export(mod, inp)
        self.assertEqual(exported_program.fw_module(*inp)[0], mod(*inp))

if __name__ == '__main__':
    run_tests()
