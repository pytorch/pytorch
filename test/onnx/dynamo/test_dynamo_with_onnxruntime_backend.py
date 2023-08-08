# Owner(s): ["module: onnx"]
from __future__ import annotations

import copy

import torch
import torch.onnx
from torch import nn

from torch.onnx._backend.core import make_aot_ort
from torch.testing._internal import common_utils

from .. import onnx_test_common


class TestDynamoWithONNXRuntime(onnx_test_common._TestONNXRuntime):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()

    def tearDown(self):
        torch._dynamo.reset()
        super().tearDown()

    def _elementwise_numerical_comparison(
        self, dynamo_backend, batch_dimensions_to_test
    ):
        """Run original and compiled elementwise models and compare the results.

        Args:
            dynamo_backend: The dynamo backend to use. Here we use string `onnxrt` or
                            the first returned value of `make_aot_ort(dynamic=True)`.
            batch_dimensions_to_test: A tuple of batch dimensions to test. E.g., (1, 2, 4, 6, 8).
        """

        def elementwise_model(w: torch.Tensor):
            x = w + w
            y = x.relu()
            z = y * y
            return x, y, z

        compiled_model = torch.compile(
            elementwise_model,
            backend=dynamo_backend,
            dynamic=True,
        )

        for batch in batch_dimensions_to_test:
            tensor_w = torch.randn(batch, dtype=torch.float32)
            x_baseline, y_baseline, z_baseline = elementwise_model(tensor_w)
            x, y, z = compiled_model(tensor_w)
            torch.testing.assert_allclose(x_baseline, x)
            torch.testing.assert_allclose(y_baseline, y)
            torch.testing.assert_allclose(z_baseline, z)

    def test_elementwise_function_with_local_backend(self):
        batch_dimensions_to_test = (2, 4, 6, 8)
        local_aot_ort, local_ort = make_aot_ort(dynamic=True)

        self._elementwise_numerical_comparison(local_aot_ort, batch_dimensions_to_test)

        # OrtBackend._ort_acclerated_call should have been called 5 times because
        # we have 5 different batch sizes to test.
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

    def _mlp_numerical_comparison(self, dynamo_backend, batch_dimensions_to_test):
        """Run original and compiled MLP models and compare the results.

        Args:
            dynamo_backend: The dynamo backend to use. Here we use string `onnxrt` or
                            the first returned value of `make_aot_ort(dynamic=True)`.
            batch_dimensions_to_test: A tuple of batch dimensions to test. E.g., (1, 2, 4, 6, 8).
        """

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
            copy.deepcopy(model), backend=dynamo_backend, dynamic=True
        )

        for batch_size in batch_dimensions_to_test:
            tensor_x = torch.randn(batch_size, 2, dtype=torch.float32)
            y_baseline = model(tensor_x)
            y = compiled_model(tensor_x)
            torch.testing.assert_allclose(y_baseline, y)

    def test_mlp_with_local_backend(self):
        batch_dimensions_to_test = (1, 2, 4, 6, 8)
        local_aot_ort, local_ort = make_aot_ort(dynamic=True)
        self._mlp_numerical_comparison(local_aot_ort, batch_dimensions_to_test)

        # OrtBackend._ort_acclerated_call should have been called 5 times because
        # we have 5 different batch sizes to test.
        self.assertEqual(len(batch_dimensions_to_test), local_ort.execution_count)
        # Since this local_ort only compiled one function, there should be only two
        # GraphModule's in its cached. One for batch sizes 2, 4, 6, 8 and the other
        # for batch size 1.
        self.assertEqual(
            len(local_ort._all_ort_execution_info.execution_info_per_graph_module), 2
        )
        # Since dynamic shape is enabled, we should only have one ONNX model
        # to support different batch sizes.
        for (
            onnx_info
        ) in local_ort._all_ort_execution_info.execution_info_per_graph_module.values():
            self.assertEqual(len(onnx_info), 1)


if __name__ == "__main__":
    common_utils.run_tests()
