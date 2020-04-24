import unittest
import torch
import torch.utils.cpp_extension

import onnx
import caffe2.python.onnx.backend as c2

import numpy as np

from test_pytorch_onnx_caffe2 import do_export

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


if __name__ == '__main__':
    unittest.main()
