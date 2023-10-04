# Owner(s): ["module: unknown"]

import os.path
import sys
import tempfile

import torch
from torch import ops

from model import Model, get_custom_op_library_path
from torch.testing._internal.common_utils import TestCase, run_tests


class TestCustomOperators(TestCase):
    def setUp(self):
        self.library_path = get_custom_op_library_path()
        ops.load_library(self.library_path)

    def test_custom_library_is_loaded(self):
        self.assertIn(self.library_path, ops.loaded_libraries)

    def test_abstract_impl_pystub_faketensor(self):
        from functorch import make_fx
        x = torch.randn(3, device='cpu')
        self.assertNotIn("my_custom_ops", sys.modules.keys())

        with self.assertRaises(torch._subclasses.fake_tensor.UnsupportedOperatorException):
            gm = make_fx(torch.ops.custom.nonzero.default, tracing_mode="symbolic")(x)

        torch.ops.import_module("my_custom_ops")
        gm = make_fx(torch.ops.custom.nonzero.default, tracing_mode="symbolic")(x)
        self.assertExpectedInline("""\
def forward(self, arg0_1):
    nonzero = torch.ops.custom.nonzero.default(arg0_1);  arg0_1 = None
    return nonzero
""".strip(), gm.code.strip())

    def test_abstract_impl_pystub_meta(self):
        x = torch.randn(3, device="meta")
        self.assertNotIn("my_custom_ops2", sys.modules.keys())
        with self.assertRaisesRegex(NotImplementedError, r"import the 'my_custom_ops2'"):
            y = torch.ops.custom.sin.default(x)
        torch.ops.import_module("my_custom_ops2")
        y = torch.ops.custom.sin.default(x)

    def test_calling_custom_op_string(self):
        output = ops.custom.op2("abc", "def")
        self.assertLess(output, 0)
        output = ops.custom.op2("abc", "abc")
        self.assertEqual(output, 0)

    def test_calling_custom_op(self):
        output = ops.custom.op(torch.ones(5), 2.0, 3)
        self.assertEqual(type(output), list)
        self.assertEqual(len(output), 3)
        for tensor in output:
            self.assertTrue(tensor.allclose(torch.ones(5) * 2))

        output = ops.custom.op_with_defaults(torch.ones(5))
        self.assertEqual(type(output), list)
        self.assertEqual(len(output), 1)
        self.assertTrue(output[0].allclose(torch.ones(5)))

    def test_calling_custom_op_with_autograd(self):
        x = torch.randn((5, 5), requires_grad=True)
        y = torch.randn((5, 5), requires_grad=True)
        output = ops.custom.op_with_autograd(x, 2, y)
        self.assertTrue(output.allclose(x + 2 * y + x * y))

        go = torch.ones((), requires_grad=True)
        output.sum().backward(go, False, True)
        grad = torch.ones(5, 5)

        self.assertEqual(x.grad, y + grad)
        self.assertEqual(y.grad, x + grad * 2)

        # Test with optional arg.
        x.grad.zero_()
        y.grad.zero_()
        z = torch.randn((5, 5), requires_grad=True)
        output = ops.custom.op_with_autograd(x, 2, y, z)
        self.assertTrue(output.allclose(x + 2 * y + x * y + z))

        go = torch.ones((), requires_grad=True)
        output.sum().backward(go, False, True)
        self.assertEqual(x.grad, y + grad)
        self.assertEqual(y.grad, x + grad * 2)
        self.assertEqual(z.grad, grad)

    def test_calling_custom_op_with_autograd_in_nograd_mode(self):
        with torch.no_grad():
            x = torch.randn((5, 5), requires_grad=True)
            y = torch.randn((5, 5), requires_grad=True)
            output = ops.custom.op_with_autograd(x, 2, y)
            self.assertTrue(output.allclose(x + 2 * y + x * y))

    def test_calling_custom_op_inside_script_module(self):
        model = Model()
        output = model.forward(torch.ones(5))
        self.assertTrue(output.allclose(torch.ones(5) + 1))

    def test_saving_and_loading_script_module_with_custom_op(self):
        model = Model()
        # Ideally we would like to not have to manually delete the file, but NamedTemporaryFile
        # opens the file, and it cannot be opened multiple times in Windows. To support Windows,
        # close the file after creation and try to remove it manually.
        file = tempfile.NamedTemporaryFile(delete=False)
        try:
            file.close()
            model.save(file.name)
            loaded = torch.jit.load(file.name)
        finally:
            os.unlink(file.name)

        output = loaded.forward(torch.ones(5))
        self.assertTrue(output.allclose(torch.ones(5) + 1))


if __name__ == "__main__":
    run_tests()
