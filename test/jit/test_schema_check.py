import os
import sys
import unittest

import torch

from torch.testing._internal import schema_check_tensor

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
        schema_check_tensor.start_recording()

    # Tests that SchemaCheckTensor records operator order with grad
    def test_schema_check_tensor_operator_order_grad(self):
        x = torch.rand((3, 3), requires_grad=True)
        schema_check_tensor.SchemaCheckTensor(x).relu().sin()
        self.assertEqual(["relu.default", "detach.default", "sin.default"], schema_check_tensor.ops)

    # Tests that SchemaCheckTensor records operator order without grad
    def test_schema_check_tensor_operator_order_no_grad(self):
        x = torch.rand((3, 3), requires_grad=False)
        schema_check_tensor.SchemaCheckTensor(x).relu().sin()
        self.assertEqual(["relu.default", "sin.default"], schema_check_tensor.ops)

    # Tests that SchemaCheckTensor wraps torch.Tensor
    def test_schema_check_tensor_functionality(self):
        x = torch.rand((3, 3), requires_grad=True)
        self.assertEqual(x.relu().sin(), schema_check_tensor.SchemaCheckTensor(x).relu().sin().elem)
