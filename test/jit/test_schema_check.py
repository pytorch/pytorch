# Owner(s): ["oncall: jit"]

import os
import sys
import torch

from torch.testing._internal.schema_check_tensor import SchemaCheckTensor

pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase

if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_jit.py TESTNAME\n\n"
                       "instead.")

# Tests various schema checking functionalities.
class TestSchemaCheck(JitTestCase):
    def setUp(self):
        SchemaCheckTensor.reset_cache()

    # Tests that SchemaCheckTensor records operator order with grad
    def test_schema_check_tensor_operator_order_grad(self):
        x = torch.rand((3, 3), requires_grad=True)
        SchemaCheckTensor(x).relu().sin()
        self.assertEqual(["aten::relu", "aten::detach", "aten::sin"], SchemaCheckTensor.recorded_ops)

    # Tests that SchemaCheckTensor records operator order without grad
    def test_schema_check_tensor_operator_order_no_grad(self):
        x = torch.rand((3, 3), requires_grad=False)
        SchemaCheckTensor(x).relu().sin()
        self.assertEqual(["aten::relu", "aten::sin"], SchemaCheckTensor.recorded_ops)

    # Tests that SchemaCheckTensor wraps torch.Tensor
    def test_schema_check_tensor_functionality(self):
        x = torch.rand((3, 3), requires_grad=True)
        self.assertEqual(x.relu().sin(), SchemaCheckTensor(x).relu().sin().elem)

    # Tests that SchemaCheckTensor wraps torch.Tensor when an argument's default is overriden
    def test_schema_check_tensor_functionality_default_replaced(self):
        x = torch.rand((3, 3), requires_grad=True)
        self.assertEqual(x.add(x, alpha=2), SchemaCheckTensor(x).add(SchemaCheckTensor(x), alpha=2).elem)

    # Tests that SchemaCheckTensor wraps torch.Tensorwith a mutable op
    def test_schema_check_tensor_functionality_mutable_inputs(self):
        x = torch.rand((3, 3), requires_grad=False)
        y = torch.clone(x)
        x.sinh_()
        SchemaCheckTensor(y).sinh_()
        self.assertEqual(x, y)

    # Tests that an exception is raised for a mismatching mutation
    def test_mutation_check_fail(self):
        with self.assertRaises(RuntimeError):
            x = torch.rand((3, 3), requires_grad=True)
            batch = torch.nn.BatchNorm1d(3, track_running_stats=True)
            batch(SchemaCheckTensor(x))

    # Tests that an exception is raised for a mismatching mutation over multiple ops
    def test_mutation_check_fail_multiple_operators(self):
        with self.assertRaises(RuntimeError):
            x = torch.rand((3, 3), requires_grad=True)
            x.sinh_()
            x.tanh_()
            x.relu_()
            batch = torch.nn.BatchNorm1d(3, track_running_stats=True)
            batch(SchemaCheckTensor(x))
