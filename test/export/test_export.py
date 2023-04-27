# Owner(s): ["module: dynamo"]
import unittest

import torch
import torch._dynamo as torchdynamo
from torch._dynamo.eval_frame import is_dynamo_supported
from torch._export import _export
from torch._export.constraints import constrain_as_size
from torch._export.trace import do_not_use_experimental_export
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import run_tests, TestCase


class TestExperimentalExport(TestCase):
    @unittest.skip("TypeError: <lambda>() missing 1 required positional argument")
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

        exported_program = do_not_use_experimental_export(mod, inp)
        self.assertEqual(exported_program.fw_module(*inp)[0], mod(*inp))

    @unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo doesn't support")
    def test_export_simple_model(self):
        class Foo(torch.nn.Module):
            def __init__(self, float_val):
                super().__init__()
                self.float_val = float_val

            def forward(self, x):
                return x.cos()

        inp = (torch.ones(6, 4, requires_grad=True),)
        mod = Foo(0.5)

        exported_program = do_not_use_experimental_export(mod, inp)
        self.assertEqual(exported_program.fw_module(*inp)[0], mod(*inp))

    @unittest.skip("TypeError: <lambda>() missing 1 required positional argument")
    def test_export_simple_model_buffer_mutation(self):
        class Foo(torch.nn.Module):
            def __init__(self, float_val):
                super().__init__()
                self.register_buffer("buffer1", torch.ones(6, 1))

            def forward(self, x):
                self.buffer1.add_(2)
                return x.cos() + self.buffer1.sin()

        inp = (torch.ones(6, 4, requires_grad=True),)
        mod = Foo(0.5)

        exported_program = do_not_use_experimental_export(mod, inp)
        mutated_buffer, output = exported_program.fw_module(*inp)
        # TODO (tmanlaibaatar) enable this once we figure out
        # how to do buffer mutation
        # self.assertEqual(mutated_buffer.sum().item(), 30)
        self.assertEqual(output, mod(*inp))


class TestExport(TestCase):
    @unittest.skipIf(not is_dynamo_supported(), "Dynamo not supported")
    def test_export_constraints(self):

        def f(x):
            b = x.item()
            constrain_as_size(b, min=2, max=5)
            return torch.full((b, 1), 1)

        inp = (torch.tensor([3]),)
        ref = f(*inp)

        gm = _export(f, inp)
        res = gm(*inp)

        self.assertTrue(torchdynamo.utils.same(ref, res))

        gm = make_fx(f, tracing_mode="symbolic")(*inp)
        res = gm(*inp)
        self.assertTrue(torchdynamo.utils.same(ref, res))

    @unittest.skipIf(not is_dynamo_supported(), "Dynamo not supported")
    def test_export_constraints_error(self):
        def invalid_size(x):
            b = x.item()
            constrain_as_size(b, min=0, max=5)
            return torch.full((b, 1), 1)

        inp = (torch.tensor([3]),)
        with self.assertRaisesRegex(torchdynamo.exc.UserError, "Unable to set min size"):
            _export(invalid_size, inp)

        def invalid_input(x):
            b = x.item()
            constrain_as_size(b, min=2, max=5)
            return torch.full((b, 1), 1)

        inp = (torch.tensor([6]),)

        with self.assertRaisesRegex(torch.utils._sympy.value_ranges.ValueRangeError, "Invalid value 6 for range"):
            _export(invalid_input, inp)

        def conflicting_constraints(x):
            b = x.item()
            constrain_as_size(b, min=2, max=3)
            constrain_as_size(b, min=4, max=5)
            return torch.full((b, 1), 1)

        inp = (torch.tensor([3]),)

        with self.assertRaisesRegex(torchdynamo.exc.UserError, "Invalid ranges"):
            _export(conflicting_constraints, inp)

if __name__ == '__main__':
    run_tests()
