import unittest
import torch
import torch.utils.cpp_extension

import onnx
import caffe2.python.onnx.backend as c2

import numpy as np

from test_pytorch_onnx_caffe2 import do_export
from test_pytorch_onnx_onnxruntime import run_model_test
from torch.onnx.symbolic_helper import _unimplemented

class TestCustomOps(unittest.TestCase):

    def test_custom_add(self):
        op_source = """
        #include <torch/script.h>

        torch::Tensor custom_add(torch::Tensor self, torch::Tensor other) {
          return self + other;
        }

        static auto registry =
          torch::RegisterOperators("custom_namespace::custom_add", &custom_add);
        """

        torch.utils.cpp_extension.load_inline(
            name="custom_add",
            cpp_sources=op_source,
            is_python_module=False,
            verbose=True,
        )

        class CustomAddModel(torch.nn.Module):
            def forward(self, a, b):
                return torch.ops.custom_namespace.custom_add(a, b)

        def symbolic_custom_add(g, self, other):
            return g.op('Add', self, other)

        from torch.onnx import register_custom_op_symbolic
        register_custom_op_symbolic('custom_namespace::custom_add', symbolic_custom_add, 9)

        x = torch.randn(2, 3, 4, requires_grad=False)
        y = torch.randn(2, 3, 4, requires_grad=False)

        model = CustomAddModel()
        onnxir, _ = do_export(model, (x, y), opset_version=11)
        onnx_model = onnx.ModelProto.FromString(onnxir)
        prepared = c2.prepare(onnx_model)
        caffe2_out = prepared.run(inputs=[x.cpu().numpy(), y.cpu().numpy()])
        np.testing.assert_array_equal(caffe2_out[0], model(x, y).cpu().numpy())


class TestCustomAutogradFunction(unittest.TestCase):
    opset_version = 9
    keep_initializers_as_inputs = False
    onnx_shape_inference = True

    def test_symbolic(self):
        class MyClip(torch.autograd.Function):

            @staticmethod
            def forward(ctx, input, scalar):
                ctx.save_for_backward(input)
                return input.clamp(min=scalar)

            @staticmethod
            def symbolic(g, input, scalar):
                return g.op("Clip", input, min_f=scalar)

        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.clip = MyClip.apply

            def forward(self, x):
                h = self.clip(x, 2)
                return h

        x = torch.randn(2, 3, 4, requires_grad=True)
        model = MyModule()
        run_model_test(self, model, input=(x, ))

    def test_register_custom_op(self):
        class MyClip(torch.autograd.Function):

            @staticmethod
            def forward(ctx, input, scalar):
                ctx.save_for_backward(input)
                return input.clamp(min=scalar)

        class MyRelu(torch.autograd.Function):

            @staticmethod
            def forward(ctx, input):
                ctx.save_for_backward(input)
                return input.clamp(min=0)

        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.clip = MyClip.apply
                self.relu = MyRelu.apply

            def forward(self, x):
                h = self.clip(x, 2)
                h = self.relu(h)
                return h

        def symbolic_pythonop(g, n, *args, **kwargs):
            name = kwargs['name']
            if name == "MyClip":
                return g.op("Clip", args[0], min_f=args[1])
            elif name == "MyRelu":
                return g.op("Relu", args[0])
            else:
                return _unimplemented("prim::PythonOp", "unknown node kind: " + name)

        from torch.onnx import register_custom_op_symbolic
        register_custom_op_symbolic('::prim_PythonOp', symbolic_pythonop, 1)

        x = torch.randn(2, 3, 4, requires_grad=True)
        model = MyModule()
        run_model_test(self, model, input=(x, ))

if __name__ == '__main__':
    unittest.main()
