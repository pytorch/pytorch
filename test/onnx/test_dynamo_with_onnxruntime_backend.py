# Owner(s): ["module: onnx"]
from __future__ import annotations

import copy

import onnx_test_common
import torch
import torch.onnx
from torch import nn

from torch.onnx._backend.core import make_aot_ort
from torch.testing._internal import common_utils


class TestDynamoWithONNXRuntime(onnx_test_common._TestONNXRuntime):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()

    def tearDown(self):
        torch._dynamo.reset()
        super().tearDown()

    def test_simple_function(self):
        def func(w):
            x = w + w
            y = x.relu()
            z = y * y
            return x, y, z

        local_aot_ort, local_ort = make_aot_ort(dynamic=True)

        compiled_func = torch.compile(
            func,
            backend=local_aot_ort,
            dynamic=True,
        )

        feature_dimension = 4
        batch_dimensions_to_test = (2, 4, 6, 8)
        for batch in batch_dimensions_to_test:
            tensor_w = torch.randn(batch, feature_dimension, dtype=torch.float32)
            x_baseline, y_baseline, z_baseline = func(tensor_w)
            x, y, z = compiled_func(tensor_w)
            torch.testing.assert_allclose(x_baseline, x)
            torch.testing.assert_allclose(y_baseline, y)
            torch.testing.assert_allclose(z_baseline, z)

        # OrtBackend._ort_acclerated_call should have been called 5 times because
        # we have 5 different batch sizes to test.
        breakpoint()
        self.assertEqual(len(batch_dimensions_to_test), local_ort.execution_count)
        # Since this local_ort only compiled one function, there should be only one
        # GraphModule in its cached.
        self.assertEqual(
            len(local_ort._all_ort_execution_info.execution_info_per_graph_module), 1
        )
        # Since dynamic shape is enabled, we should only have one ONNX model
        # to support different batch sizes.
        for (
            onnx_info
        ) in local_ort._all_ort_execution_info.execution_info_per_graph_module.values():
            self.assertEqual(len(onnx_info), 1)

    def test_mlp(self):
        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(2, 4, bias=True)
                self.fc2 = nn.Linear(4, 2, bias=True)

            def forward(self, tensor_x: torch.Tensor):
                tensor_x = self.fc1(tensor_x)
                tensor_x = torch.sigmoid(tensor_x)
                tensor_x = self.fc2(tensor_x)
                tensor_x = torch.sigmoid(tensor_x)
                return tensor_x

        model = MLP()

        compiled_model = torch.compile(
            copy.deepcopy(model), backend="onnxrt", dynamic=True
        )

        for batch_size in (2, 4, 6, 8):
            tensor_x = torch.randn(batch_size, 2, dtype=torch.float32)
            y_baseline = model(tensor_x)
            y = compiled_model(tensor_x)
            torch.testing.assert_allclose(y_baseline, y)


if __name__ == "__main__":
    common_utils.run_tests()
