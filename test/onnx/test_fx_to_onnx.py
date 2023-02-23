# Owner(s): ["module: onnx"]
import unittest

import pytorch_test_common
import torch
from torch import nn
from torch.nn import functional as F
from torch.onnx._internal import fx as fx_onnx
from torch.testing._internal import common_utils


class TestFxToOnnx(pytorch_test_common.ExportTestCase):
    def setUp(self):
        super().setUp()
        self.opset_version = torch.onnx._constants.ONNX_DEFAULT_OPSET

    def test_simple_function(self):
        def func(x):
            y = x + 1
            z = y.relu()
            return (y, z)

        _ = fx_onnx.export(func, torch.randn(1, 1, 2), opset_version=self.opset_version)

    @unittest.skip(
        "Conv Op is not supported at the time. https://github.com/microsoft/onnx-script/issues/397"
    )
    def test_mnist(self):
        class MNISTModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, 1, bias=False)
                self.conv2 = nn.Conv2d(32, 64, 3, 1, bias=False)
                self.fc1 = nn.Linear(9216, 128, bias=False)
                self.fc2 = nn.Linear(128, 10, bias=False)

            def forward(self, tensor_x: torch.Tensor):
                tensor_x = self.conv1(tensor_x)
                tensor_x = F.sigmoid(tensor_x)
                tensor_x = self.conv2(tensor_x)
                tensor_x = F.sigmoid(tensor_x)
                tensor_x = F.max_pool2d(tensor_x, 2)
                tensor_x = torch.flatten(tensor_x, 1)
                tensor_x = self.fc1(tensor_x)
                tensor_x = F.sigmoid(tensor_x)
                tensor_x = self.fc2(tensor_x)
                output = F.log_softmax(tensor_x, dim=1)
                return output

        tensor_x = torch.rand((64, 1, 28, 28), dtype=torch.float32)
        _ = fx_onnx.export(MNISTModel(), tensor_x, opset_version=self.opset_version)

    def test_trace_only_op_with_evaluator(self):
        model_input = torch.tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 2.0]])

        class ArgminArgmaxModel(torch.nn.Module):
            def forward(self, input):
                return (
                    torch.argmin(input),
                    torch.argmax(input),
                    torch.argmin(input, keepdim=True),
                    torch.argmax(input, keepdim=True),
                    torch.argmin(input, dim=0, keepdim=True),
                    torch.argmax(input, dim=1, keepdim=True),
                )

        _ = fx_onnx.export(
            ArgminArgmaxModel(), model_input, opset_version=self.opset_version
        )

    def test_multiple_outputs_op_with_evaluator(self):
        class TopKModel(torch.nn.Module):
            def forward(self, x):
                return torch.topk(x, 3)

        x = torch.arange(1.0, 6.0, requires_grad=True)
        _ = fx_onnx.export(TopKModel(), x, opset_version=self.opset_version)


if __name__ == "__main__":
    common_utils.run_tests()
