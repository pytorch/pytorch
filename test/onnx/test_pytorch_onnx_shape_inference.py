import unittest
import torch
import numpy as np

import copy

def expect_tensor(scalar_type, shape=None, dynamic_shape=None):
    def verify(actual_type):
        np.testing.assert_equal(actual_type.scalarType(), scalar_type)
        if shape is not None:
            np.testing.assert_equal(actual_type.sizes(), shape)
        if dynamic_shape is not None:
            np.testing.assert_equal(actual_type.varyingSizes(), dynamic_shape)
    return verify

class TestONNXShapeInference(unittest.TestCase):
    from torch.onnx.symbolic_helper import _onnx_main_opset
    opset_version = _onnx_main_opset

    def run_test(self, g, n, expected_types):
        if not isinstance(expected_types, list):
            expected_types = [expected_types]

        torch._C._jit_pass_onnx_graph_shape_type_inference(g, {}, self.opset_version)
        for out, expected_type in zip(n.outputs(), expected_types):
            expected_type(out.type())

    def create_empty_graph(self):
        return torch._C.Graph()

    def insert_tensor_constant(self, g, tensor):
        return g.op("Constant", value_t=tensor)

    def test_cast(self):
        # Test cast with input of unknown scalar type.
        g = self.create_empty_graph()
        input = g.addInput()
        cast_out = g.op("Cast", input, to_i=1)
        self.run_test(g, cast_out.node(), expect_tensor('Float'))

    def test_constant_of_shape(self):
        # Test ConstantOfShape with input of onnx::Shape node.
        g = self.create_empty_graph()
        constant = self.insert_tensor_constant(g, torch.ones(1, 2, 3, 4))
        shape = g.op("Shape", constant)
        constant_of_shape = g.op("ConstantOfShape", shape, value_t=torch.tensor([2.0]))
        self.run_test(g, constant_of_shape.node(), expect_tensor('Float', (1, 2, 3, 4)))

        # Test ConstantOfShape with input of prim::ListConstruct of static tensor
        rank = 4
        g = self.create_empty_graph()
        constants = [self.insert_tensor_constant(g, torch.tensor(i+1)) for i in range(rank)]
        shape = g.op("prim::ListConstruct", *constants)
        shape.setType(torch._C.ListType.ofInts())
        constant_of_shape = g.op("ConstantOfShape", shape, value_t=torch.tensor([2.0]))
        self.run_test(g, constant_of_shape.node(), expect_tensor('Float', (1, 2, 3, 4)))

        # Test ConstantOfShape with input of prim::ListConstruct of dynamic tensor
        rank = 4
        g = self.create_empty_graph()
        inputs = [g.addInput() for i in range(rank)]
        shape = g.op("prim::ListConstruct", *inputs)
        shape.setType(torch._C.ListType.ofInts())
        constant_of_shape = g.op("ConstantOfShape", shape, value_t=torch.tensor([2.0]))
        self.run_test(g, constant_of_shape.node(), expect_tensor('Float', None, (None, None, None, None)))

if __name__ == '__main__':
    unittest.main()
