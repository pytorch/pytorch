# Owner(s): ["module: onnx"]
import io
import unittest
from typing import Any, Sequence, Tuple, Union

import onnx_test_common
import onnxruntime  # type: ignore[import]
import torch
import transformers  # type: ignore[import]
from torch import nn
from torch.nn import functional as F
from torch.onnx._internal import fx as fx_onnx
from torch.testing._internal import common_utils
from torch.utils import _pytree as pytree


class TestFxToOnnxWithOnnxRuntime(onnx_test_common._TestONNXRuntime):
    def setUp(self):
        super().setUp()
        self.opset_version = torch.onnx._constants.ONNX_DEFAULT_OPSET

    def _run_ort(
        self, onnx_model: Union[str, io.BytesIO], pytorch_inputs: Tuple[Any, ...]
    ) -> Sequence[Any]:
        session = onnxruntime.InferenceSession(
            onnx_model, providers=["CPUExecutionProvider"]
        )
        input_names = [ort_input.name for ort_input in session.get_inputs()]
        return session.run(
            None, {k: v.cpu().numpy() for k, v in zip(input_names, pytorch_inputs)}
        )

    def test_simple_function(self):
        def func(x):
            y = x + 1
            z = y.relu()
            return (y, z)

        tensor_x = torch.randn(1, 1, 2, dtype=torch.float32)

        self.run_test_with_fx_to_onnx_exporter(func, (tensor_x,))

    @unittest.skip("TypeError: export() got an unexpected keyword argument 'b'")
    def test_func_with_args_and_kwargs(self):
        def func(x, b=1.0):
            y = x + b
            z = y.relu()
            return (y, z)

        tensor_x = torch.randn(1, 1, 2, dtype=torch.float32)

        self.run_test_with_fx_to_onnx_exporter(func, (tensor_x,), {"b": 500.0})

    def test_mnist(self):
        class MNISTModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, 1, bias=True)
                self.conv2 = nn.Conv2d(32, 64, 3, 1, bias=True)
                self.fc1 = nn.Linear(9216, 128, bias=True)
                self.fc2 = nn.Linear(128, 10, bias=True)

            def forward(self, tensor_x: torch.Tensor):
                tensor_x = self.conv1(tensor_x)
                tensor_x = torch.sigmoid(tensor_x)
                tensor_x = self.conv2(tensor_x)
                tensor_x = torch.sigmoid(tensor_x)
                tensor_x = F.max_pool2d(tensor_x, 2)
                tensor_x = torch.flatten(tensor_x, 1)
                tensor_x = self.fc1(tensor_x)
                tensor_x = torch.sigmoid(tensor_x)
                output = self.fc2(tensor_x)
                return output

        tensor_x = torch.rand((64, 1, 28, 28), dtype=torch.float32)
        self.run_test_with_fx_to_onnx_exporter(MNISTModel(), (tensor_x,))

    # test single op with no kwargs
    def test_sigmoid(self):
        x = torch.randn(1, 4, 2, 3)

        class SigmoidModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sigmoid = torch.nn.Sigmoid()

            def forward(self, x):
                return self.sigmoid(x)

        self.run_test_with_fx_to_onnx_exporter(SigmoidModel(), (x,))

    # test single op with no kwargs
    def test_sigmoid_add(self):
        x = torch.randn(1, 4, 2, 3)

        class SigmoidAddModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sigmoid = torch.nn.Sigmoid()

            def forward(self, x):
                x = torch.ops.aten.add(x, 1, alpha=2)
                return self.sigmoid(x)

        self.run_test_with_fx_to_onnx_exporter(SigmoidAddModel(), (x,))
    def test_gpt2_tiny(self):
        model_name = "sshleifer/tiny-gpt2"
        # Download pytorch model
        model = transformers.AutoModel.from_pretrained(model_name)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

        # Transform input tokens
        inputs = tokenizer("Hello world!", return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        onnx_model = fx_onnx.export_without_kwargs(
            model, self.opset_version, **inputs, use_binary_format=True
        )

        ref_outputs, _ = pytree.tree_flatten(model(**inputs, return_dict=False))
        ort_outputs = self._run_ort(onnx_model, (input_ids, attention_mask))
        assert len(ref_outputs) == len(ort_outputs)
        assert len(ref_outputs) == 5
        for ref_output, ort_output in zip(ref_outputs, ort_outputs):
            torch.testing.assert_allclose(ref_output, torch.tensor(ort_output))


if __name__ == "__main__":
    common_utils.run_tests()
