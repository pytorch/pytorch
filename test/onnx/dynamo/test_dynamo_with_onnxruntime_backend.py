# Owner(s): ["module: onnx"]
from __future__ import annotations

import copy
import os
import sys
from typing import Tuple

import torch
import torch.onnx
from torch import nn

from torch.onnx._internal.onnxruntime import make_aot_ort, OrtBackend
from torch.testing._internal import common_utils

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import onnx_test_common


class TestDynamoWithONNXRuntime(onnx_test_common._TestONNXRuntime):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()

    def tearDown(self):
        super().tearDown()
        torch._dynamo.reset()

    def _test_model_numerically(
        self,
        model,
        dynamo_backend,
        example_args_collection,
    ):
        """Run original and compiled model and compare the results.

        Args:
            model: The model to test.
            dynamo_backend: The dynamo backend to use. Here we use string `onnxrt` or
              the first returned value of `make_aot_ort(dynamic=True)`.
            example_args_collection: A tuple of example arguments to test. E.g.,
                (
                  (torch.randn(2), torch.randn(2)),
                  (torch.randn(4), torch.randn(4)),
                )
              if you want to test
                model(torch.randn(2), torch.randn(2)) and
                model(torch.randn(4), torch.randn(4))
              .
        """
        compiled_model = torch.compile(
            model if not isinstance(model, torch.nn.Module) else copy.deepcopy(model),
            backend=dynamo_backend,
            dynamic=True,
        )

        for example_args in example_args_collection:
            baseline_result = model(*example_args)
            result = compiled_model(*example_args)
            if isinstance(baseline_result, torch.Tensor):
                torch.testing.assert_close(baseline_result, result)
            else:
                for baseline_elem, result_elem in zip(baseline_result, result):
                    torch.testing.assert_close(baseline_elem, result_elem)

    def _assert_counting_information(
        self,
        ort_backend: OrtBackend,
        # Number of session runs.
        # If there is no graph break, this should be the same as
        # total number of forward calls.
        expected_execution_count: int,
        # Number of GraphModule's cached.
        # With one graph break, a model will be mapped
        # to two GraphModule's.
        number_of_cached_graph_modules: int,
        # Number of ONNX models cached for each GraphModule,
        # number_of_exported_onnx_models[i] contains # of ONNX models exported from
        # the i-th element (type: torch.fx.GraphModule) in
        # OrtBackend._all_ort_execution_info.execution_info_per_graph_module.values().
        number_of_exported_onnx_models_for_all_graph_modules: Tuple[int, ...],
    ):
        self.assertEqual(expected_execution_count, ort_backend.execution_count)
        self.assertEqual(
            len(ort_backend._all_ort_execution_info.execution_info_per_graph_module),
            number_of_cached_graph_modules,
        )
        self.assertEqual(
            len(ort_backend._all_ort_execution_info.execution_info_per_graph_module),
            len(number_of_exported_onnx_models_for_all_graph_modules),
        )
        for (
            onnx_info,
            expected_number_of_onnx_models,
        ) in zip(
            ort_backend._all_ort_execution_info.execution_info_per_graph_module.values(),
            number_of_exported_onnx_models_for_all_graph_modules,
        ):
            self.assertEqual(len(onnx_info), expected_number_of_onnx_models)

    def test_elementwise_function_single_output(self):
        local_aot_ort, local_ort = make_aot_ort(dynamic=True)

        example_args_collection = tuple(
            (torch.randn(batch, dtype=torch.float32),) for batch in (2, 4, 6, 8, 10)
        )

        def elementwise_model(x: torch.Tensor):
            y = x.relu()
            z = y.sigmoid()
            return z

        self._test_model_numerically(
            elementwise_model,
            local_aot_ort,
            example_args_collection,
        )

        self._assert_counting_information(
            local_ort,
            # OrtBackend._ort_acclerated_call should have been called 5 times because
            # we have 5 different batch sizes to test.
            expected_execution_count=len(example_args_collection),
            # Since this local_ort only compiled one function,
            # there should be only one GraphModule in its cached.
            number_of_cached_graph_modules=1,
            # Since dynamic shape is enabled, we should only have one ONNX model
            # to support different batch sizes.
            number_of_exported_onnx_models_for_all_graph_modules=(1,),
        )

    def test_elementwise_function_multiple_output(self):
        local_aot_ort, local_ort = make_aot_ort(dynamic=True)

        example_args_collection = tuple(
            (torch.randn(batch, dtype=torch.float32),) for batch in (2, 4, 8)
        )

        def elementwise_model_with_multiple_outputs(w: torch.Tensor):
            x = w + w
            y = x.relu()
            z = y * y
            return x, y, z

        self._test_model_numerically(
            elementwise_model_with_multiple_outputs,
            local_aot_ort,
            example_args_collection,
        )

        self._assert_counting_information(
            local_ort,
            expected_execution_count=len(example_args_collection),
            number_of_cached_graph_modules=1,
            number_of_exported_onnx_models_for_all_graph_modules=(1,),
        )

    def test_mlp_with_local_backend(self):
        local_aot_ort, local_ort = make_aot_ort(dynamic=True)

        example_args_collection = tuple(
            (torch.randn(batch, 2, dtype=torch.float32),) for batch in (1, 2, 4, 6, 8)
        )

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

        self._test_model_numerically(
            MLP(),
            local_aot_ort,
            example_args_collection,
        )

        self._assert_counting_information(
            local_ort,
            # OrtBackend._ort_acclerated_call should have been called 5 times because
            # we have 5 different batch sizes to test.
            expected_execution_count=len(example_args_collection),
            # Since this local_ort only compiled one function, there should be only two
            # GraphModule's in its cached. One for batch sizes 2, 4, 6, 8 and the other
            # for batch size 1.
            number_of_cached_graph_modules=2,
            # Since dynamic shape is enabled, we should only have one ONNX model
            # to support different batch sizes.
            number_of_exported_onnx_models_for_all_graph_modules=(1, 1),
        )


if __name__ == "__main__":
    common_utils.run_tests()
