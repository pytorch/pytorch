# Owner(s): ["module: onnx"]

import numpy as np

import torch
from pytorch_test_common import skipIfUnsupportedMinOpsetVersion
from torch.onnx import _constants, symbolic_helper
from torch.testing._internal import common_utils


def expect_tensor(scalar_type, shape=None):
    def verify(actual_type):
        np.testing.assert_equal(actual_type.scalarType(), scalar_type)
        # if shape is not None:
        #     np.testing.assert_equal(actual_type.sizes(), shape)
        if shape is not None:
            np.testing.assert_equal(actual_type.varyingSizes(), shape)

    return verify


class TestONNXShapeInference(common_utils.TestCase):
    def setUp(self):
        self.opset_version = _constants.ONNX_MAX_OPSET
        symbolic_helper._set_onnx_shape_inference(True)
        symbolic_helper._set_opset_version(self.opset_version)

    def run_test(self, g, n, type_assertion_funcs):
        if not isinstance(type_assertion_funcs, list):
            type_assertion_funcs = [type_assertion_funcs]

        torch._C._jit_pass_onnx_graph_shape_type_inference(g, {}, self.opset_version)
        for out, type_assertion_func in zip(n.outputs(), type_assertion_funcs):
            type_assertion_func(out.type())

    def create_empty_graph(self):
        g = torch._C.Graph()
        # kick off initialization for ConstantMap.
        torch._C._jit_pass_onnx_graph_shape_type_inference(g, {}, self.opset_version)
        return g

    def insert_tensor_constant(self, g, tensor):
        return g.op("Constant", value_t=tensor)

    def test_cast(self):
        # Test cast with input of unknown scalar type.
        g = self.create_empty_graph()
        input = g.addInput()
        cast_out = g.op("Cast", input, to_i=1)
        self.run_test(g, cast_out.node(), expect_tensor("Float"))

    def test_constant_of_shape(self):
        # Test ConstantOfShape with input of onnx::Shape node.
        g = self.create_empty_graph()
        constant = self.insert_tensor_constant(g, torch.ones(1, 2, 3, 4))
        shape = g.op("Shape", constant)
        constant_of_shape = g.op("ConstantOfShape", shape, value_t=torch.tensor([2.0]))
        self.run_test(
            g, constant_of_shape.node(), expect_tensor("Float", shape=(1, 2, 3, 4))
        )

    def test_constant_of_shape_static(self):
        # Test ConstantOfShape with input of prim::ListConstruct of static tensor
        rank = 4
        g = self.create_empty_graph()
        constants = [
            self.insert_tensor_constant(g, torch.tensor(i + 1)) for i in range(rank)
        ]
        shape = g.op("prim::ListConstruct", *constants)
        shape.setType(torch._C.ListType.ofInts())
        constant_of_shape = g.op("ConstantOfShape", shape, value_t=torch.tensor([2.0]))
        self.run_test(
            g, constant_of_shape.node(), expect_tensor("Float", shape=(1, 2, 3, 4))
        )

    def test_constant_of_shape_dynamic(self):
        # Test ConstantOfShape with input of prim::ListConstruct of dynamic tensor
        rank = 4
        g = self.create_empty_graph()
        inputs = [g.addInput() for i in range(rank)]
        shape = g.op("prim::ListConstruct", *inputs)
        shape.setType(torch._C.ListType.ofInts())
        constant_of_shape = g.op("ConstantOfShape", shape, value_t=torch.tensor([2.0]))
        self.run_test(
            g,
            constant_of_shape.node(),
            expect_tensor("Float", shape=(None, None, None, None)),
        )

    def test_gather_dynamic_index(self):
        g = self.create_empty_graph()
        input = g.addInput()
        input.setType(
            input.type().with_dtype(torch.float).with_sizes([None, 3, 16, 16])
        )
        indices = g.addInput()
        indices.setType(indices.type().with_dtype(torch.int64).with_sizes([None]))
        output = g.op("Gather", input, indices, axis_i=1)
        self.run_test(
            g, output.node(), expect_tensor("Float", shape=([None, None, 16, 16]))
        )

    def test_gather_scalar_index(self):
        g = self.create_empty_graph()
        input = g.addInput()
        input.setType(
            input.type().with_dtype(torch.float).with_sizes([None, 3, 16, 16])
        )
        indices = self.insert_tensor_constant(g, torch.tensor(1))
        output = g.op("Gather", input, indices, axis_i=1)
        self.run_test(g, output.node(), expect_tensor("Float", shape=([None, 16, 16])))

    def test_reshape(self):
        g = self.create_empty_graph()
        constant = self.insert_tensor_constant(g, torch.ones(2, 16, 5, 5))
        constant_2 = self.insert_tensor_constant(g, torch.tensor([2, 0, -1]))
        shape = g.op("Reshape", constant, constant_2)
        self.run_test(g, shape.node(), expect_tensor("Float", shape=(2, 16, 25)))

        g = self.create_empty_graph()
        constant = self.insert_tensor_constant(g, torch.ones(2, 16, 5, 4))
        constant_2 = self.insert_tensor_constant(g, torch.tensor([-1, 0, 4]))
        shape = g.op("Reshape", constant, constant_2)
        self.run_test(g, shape.node(), expect_tensor("Float", shape=(10, 16, 4)))

        g = self.create_empty_graph()
        constant = self.insert_tensor_constant(g, torch.ones(2, 16, 5, 4))
        constant_2 = self.insert_tensor_constant(g, torch.tensor([-1, 0, 0]))
        shape = g.op("Reshape", constant, constant_2)
        self.run_test(g, shape.node(), expect_tensor("Float", shape=(8, 16, 5)))

    def test_reshape_symbolic(self):
        g = self.create_empty_graph()
        input = g.addInput()
        input.setType(input.type().with_sizes([None, None, 2, 8]))
        constant = self.insert_tensor_constant(g, torch.tensor([0, 0, -1]))
        output = g.op("Reshape", input, constant)
        self.run_test(g, output.node(), expect_tensor(None, shape=(None, None, 16)))

    @skipIfUnsupportedMinOpsetVersion(14)
    def test_reshape_allowzero(self):
        g = self.create_empty_graph()
        input = g.addInput()
        input.setType(input.type().with_sizes([3, 4, 0]))
        constant = self.insert_tensor_constant(g, torch.tensor([0, 4, 3]))
        output = g.op("Reshape", input, constant, allowzero_i=1)
        self.run_test(g, output.node(), expect_tensor(None, shape=(0, 4, 3)))

    def test_slice(self):
        g = self.create_empty_graph()
        input = g.addInput()
        input.setType(input.type().with_sizes([None, None]))
        start_input = g.addInput()
        start_input.setType(start_input.type().with_sizes([None]))
        end = self.insert_tensor_constant(g, torch.tensor([3]))
        axis = self.insert_tensor_constant(g, torch.tensor([0]))
        step = self.insert_tensor_constant(g, torch.tensor([1]))
        slice = g.op("Slice", input, start_input, end, axis, step)
        self.run_test(g, slice.node(), expect_tensor(None, shape=(None, None)))

    def test_broadcast_matmul(self):
        g = self.create_empty_graph()
        constant = self.insert_tensor_constant(g, torch.ones(5, 1, 2))
        constant_2 = self.insert_tensor_constant(g, torch.ones(3, 1, 2, 1))
        shape = g.op("MatMul", constant, constant_2)
        self.run_test(g, shape.node(), expect_tensor("Float", shape=(3, 5, 1, 1)))

        # test when first input is of rank 1
        g = self.create_empty_graph()
        constant = self.insert_tensor_constant(g, torch.ones(2))
        constant_2 = self.insert_tensor_constant(g, torch.ones(3, 1, 2, 1))
        shape = g.op("MatMul", constant, constant_2)
        self.run_test(g, shape.node(), expect_tensor("Float", shape=(3, 1, 1)))

        # test when second input is of rank 1
        g = self.create_empty_graph()
        constant = self.insert_tensor_constant(g, torch.ones(5, 1, 2))
        constant_2 = self.insert_tensor_constant(g, torch.ones(2))
        shape = g.op("MatMul", constant, constant_2)
        self.run_test(g, shape.node(), expect_tensor("Float", shape=(5, 1)))

        # test when both inputs are of rank 1
        g = self.create_empty_graph()
        constant = self.insert_tensor_constant(g, torch.ones(2))
        constant_2 = self.insert_tensor_constant(g, torch.ones(2))
        shape = g.op("MatMul", constant, constant_2)
        self.run_test(g, shape.node(), expect_tensor("Float", shape=()))

    def test_expand(self):
        g = self.create_empty_graph()
        input = g.addInput()
        constant = self.insert_tensor_constant(g, torch.ones(2, 4))
        input.setType(constant.type().with_sizes([None, None]))
        shape = g.op("Shape", input)
        expand = g.op("Expand", constant, shape)
        self.run_test(g, expand.node(), expect_tensor("Float", shape=(None, None)))

    def test_pad(self):
        g = self.create_empty_graph()
        input = g.addInput()
        input.setType(input.type().with_dtype(torch.float).with_sizes([3, 320, 100]))
        constant = self.insert_tensor_constant(g, torch.ones(6, dtype=torch.long))
        none = g.op("prim::Constant").setType(torch.NoneType.get())
        pad = g.op("Pad", input, constant, none, mode_s="constant")
        self.run_test(g, pad.node(), expect_tensor("Float", shape=(5, 322, 102)))

    def test_pad_with_dynamic_input_shape(self):
        g = self.create_empty_graph()
        input = g.addInput()
        input.setType(input.type().with_dtype(torch.float).with_sizes([3, None, None]))
        constant = self.insert_tensor_constant(g, torch.ones(6, dtype=torch.long))
        none = g.op("prim::Constant").setType(torch.NoneType.get())
        pad = g.op("Pad", input, constant, none, mode_s="constant")
        self.run_test(g, pad.node(), expect_tensor("Float", shape=(5, None, None)))

    def test_pad_with_dynamic_pad_size(self):
        g = self.create_empty_graph()
        input = g.addInput()
        input.setType(input.type().with_dtype(torch.float).with_sizes([3, 320, 100]))
        pad_size = g.addInput()
        pad_size.setType(pad_size.type().with_dtype(torch.long).with_sizes([6]))
        none = g.op("prim::Constant").setType(torch.NoneType.get())
        pad = g.op("Pad", input, pad_size, none, mode_s="constant")
        self.run_test(g, pad.node(), expect_tensor("Float", shape=(None, None, None)))

    def test_resize(self):
        g = self.create_empty_graph()
        input = g.addInput()
        input.setType(input.type().with_dtype(torch.float).with_sizes([4, 32, 64, 64]))
        none = g.op("prim::Constant").setType(torch.NoneType.get())
        scales = self.insert_tensor_constant(
            g, torch.tensor([1, 1, 2, 2], dtype=torch.float)
        )
        resize = g.op(
            "Resize",
            input,
            none,
            scales,
            coordinate_transformation_mode_s="align_corners",
            cubic_coeff_a_f=-0.75,
            mode_s="linear",
            nearest_mode_s="floor",
        )
        self.run_test(g, resize.node(), expect_tensor("Float", shape=(4, 32, 128, 128)))

    def test_resize_after_concat(self):
        g = self.create_empty_graph()
        input = g.addInput()
        input.setType(input.type().with_dtype(torch.float).with_sizes([4, 32, 64, 64]))
        none = g.op("prim::Constant").setType(torch.NoneType.get())
        scale_1 = self.insert_tensor_constant(
            g, torch.tensor([1, 1], dtype=torch.float)
        )
        scale_2 = self.insert_tensor_constant(
            g, torch.tensor([2, 2], dtype=torch.float)
        )
        # `scales` values should be statically known due to constant folding in shape inference.
        scales = g.op("Concat", scale_1, scale_2, axis_i=0)
        resize = g.op(
            "Resize",
            input,
            none,
            scales,
            coordinate_transformation_mode_s="align_corners",
            cubic_coeff_a_f=-0.75,
            mode_s="linear",
            nearest_mode_s="floor",
        )
        self.run_test(g, resize.node(), expect_tensor("Float", shape=(4, 32, 128, 128)))

    def test_reduce_prod_with_axes(self):
        g = self.create_empty_graph()
        input = g.addInput()
        input.setType(input.type().with_dtype(torch.long).with_sizes([2]))
        reduce_prod = g.op("ReduceProd", input, axes_i=[0])
        self.run_test(g, reduce_prod.node(), expect_tensor("Long", shape=(1,)))

    def test_reduce_prod_without_axes(self):
        g = self.create_empty_graph()
        input = g.addInput()
        input.setType(input.type().with_dtype(torch.long).with_sizes([2]))
        reduce_prod = g.op("ReduceProd", input)
        self.run_test(g, reduce_prod.node(), expect_tensor("Long", shape=(1,)))


if __name__ == "__main__":
    common_utils.run_tests()
