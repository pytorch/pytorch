# Owner(s): ["oncall: jit"]

import os
import sys
import torch

from torch.testing._internal.schema_check_mode import SchemaCheckMode
from torch.utils._python_dispatch import enable_torch_dispatch_mode
from torch.testing._internal.jit_utils import JitTestCase

pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_jit.py TESTNAME\n\n"
                       "instead.")

# Tests various schema checking functionalities.
class TestSchemaCheck(JitTestCase):
    # Tests that SchemaCheckMode records operator order with grad
    def test_schema_check_mode_operator_order(self):
        schema_check = SchemaCheckMode()
        with enable_torch_dispatch_mode(schema_check):
            x = torch.rand((3, 3), requires_grad=True)
            x.relu().sin()
            self.assertEqual(["aten::rand", "aten::relu", "aten::sin"], schema_check.ops)

    # Tests that SchemaCheckMode records operator order without grad
    def test_schema_check_tensor_operator_order_without_grad(self):
        schema_check = SchemaCheckMode()
        with enable_torch_dispatch_mode(schema_check):
            x = torch.rand((3, 3), requires_grad=False)
            x.relu().sin()
            self.assertEqual(["aten::rand", "aten::relu", "aten::sin"], schema_check.ops)

    # Tests that SchemaCheckMode wraps torch.Tensor
    def test_schema_check_tensor_functionality(self):
        x = torch.rand((3, 3), requires_grad=True)
        expected = x.relu().sin()
        with enable_torch_dispatch_mode(SchemaCheckMode()):
            actual = x.relu().sin()
        self.assertEqual(expected, actual)

    # Tests that SchemaCheckMode wraps torch.Tensor when an argument's default is overriden
    def test_schema_check_tensor_functionality_default_replaced(self):
        x = torch.rand((3, 3), requires_grad=True)
        expected = x.add(x, alpha=2)
        with enable_torch_dispatch_mode(SchemaCheckMode()):
            actual = x.add(x, alpha=2)
        self.assertEqual(expected, actual)

    # Tests that SchemaCheckMode wraps torch.Tensor when there is a Tensor[] argument
    def test_schema_check_tensor_functionality_list_input(self):
        a = torch.rand((3, 3))
        b = torch.rand((3, 3))
        c = torch.rand((3, 3))
        expected = torch.linalg.multi_dot([a, b, c])
        with enable_torch_dispatch_mode(SchemaCheckMode()):
            actual = torch.linalg.multi_dot([a, b, c])
        self.assertEqual(expected, actual)

    # Tests that SchemaCheckMode wraps torch.Tensor when there is a kwarg tensor input
    def test_schema_check_tensor_functionality_kwarg_tensor(self):
        x = torch.rand((3, 5))
        w = torch.rand((4))
        expected = torch.stft(x, 4, win_length=4, window=w, return_complex=True)
        with enable_torch_dispatch_mode(SchemaCheckMode()):
            actual = torch.stft(x, 4, win_length=4, window=w, return_complex=True)
        self.assertEqual(expected, actual)

    # Tests that SchemaCheckMode wraps torch.Tensor with a mutable op
    def test_schema_check_tensor_functionality_mutable_inputs(self):
        expected = torch.rand((3, 3), requires_grad=False)
        actual = torch.clone(expected)
        expected.sinh_()
        with enable_torch_dispatch_mode(SchemaCheckMode()):
            actual.sinh_()
        self.assertEqual(expected, actual)

    # Tests that an exception is raised for a mismatching mutation
    def test_mutation_check_fail(self):
        with self.assertRaises(RuntimeError):
            x = torch.rand((3, 3), requires_grad=True)
            batch = torch.nn.BatchNorm1d(3, track_running_stats=True)
            with enable_torch_dispatch_mode(SchemaCheckMode()):
                batch(x)

    # Tests that an exception is raised for a mismatching mutation over multiple ops
    def test_mutation_check_fail_multiple_operators(self):
        with self.assertRaises(RuntimeError):
            x = torch.rand((3, 3), requires_grad=True)
            batch = torch.nn.BatchNorm1d(3, track_running_stats=True)
            with enable_torch_dispatch_mode(SchemaCheckMode()):
                x.sinh_()
                x.tanh_()
                x.relu_()
                batch(SchemaCheckMode(x))
