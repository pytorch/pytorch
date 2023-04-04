# Owner(s): ["module: onnx"]
import unittest

import pytorch_test_common
import torch
from torch import nn
from torch.nn import functional as F
from torch.onnx._internal import diagnostics, fx as fx_onnx
from torch.onnx._internal.diagnostics import infra
from torch.testing._internal import common_utils


def assert_has_diagnostics(
    engine: infra.DiagnosticEngine,
    rule: infra.Rule,
    level: infra.Level,
    expected_error_node: str,
    expected_error_message: str,
):
    rule_level_pairs = (rule.id, level.name.lower())
    sarif_log = engine.sarif_log()
    actual_results = []
    for run in sarif_log.runs:
        if run.results is None:
            continue
        for result in run.results:
            id_level_pair = (result.rule_id, result.level)
            actual_results.append(id_level_pair)
            if (
                result.message.text
                and result.message.markdown
                and expected_error_node in result.message.text
                and expected_error_message in result.message.markdown
            ):
                return

    raise AssertionError(
        f"Expected diagnostic results of rule id and level pair {rule_level_pairs} "
        f"not found with expected error node {expected_error_node} and "
        f"expected error message {expected_error_message}. "
        f"Actual diagnostic results: {actual_results}"
    )


class TestFxToOnnx(pytorch_test_common.ExportTestCase):
    def setUp(self):
        super().setUp()
        self.diag_ctx = diagnostics.engine.create_diagnostic_context(
            "test_fx_export", version=torch.__version__
        )
        self.opset_version = torch.onnx._constants.ONNX_DEFAULT_OPSET

    def tearDown(self):
        diagnostics.engine.dump(
            f"test_report_{self._testMethodName}.sarif", compress=False
        )
        super().tearDown()

    def test_simple_function(self):
        def func(x):
            y = x + 1
            z = y.relu()
            return (y, z)

        _ = fx_onnx.export(func, torch.randn(1, 1, 2), opset_version=self.opset_version)

    @unittest.skip("MaxPool2D Op is not supported at the time.")
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
                values, _ = torch.topk(x, 3)
                return torch.sum(values)

        x = torch.arange(1.0, 6.0, requires_grad=True)
        _ = fx_onnx.export(TopKModel(), x, opset_version=self.opset_version)

    def test_unsupported_indices_fake_tensor_generated_with_op_level_debug(self):
        class EmbedModelWithoutPaddingIdx(torch.nn.Module):
            def forward(self, input, emb):
                return torch.nn.functional.embedding(input, emb)

        model = EmbedModelWithoutPaddingIdx()
        x = torch.randint(4, (4, 3, 2))
        embedding_matrix = torch.rand(10, 3)

        _ = fx_onnx.export(model, x, embedding_matrix, op_level_debug=True)
        assert_has_diagnostics(
            diagnostics.engine,
            diagnostics.rules.fx_node_to_onnx,
            diagnostics.levels.WARNING,
            expected_error_node="aten.embedding.default",
            expected_error_message="IndexError: index out of range in self",
        )

    def test_unsupported_function_schema_with_op_level_debug(self):
        class TraceModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv2 = torch.nn.Conv2d(
                    16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1)
                )

            def forward(self, input):
                return self.conv2(input)

        x = torch.randn(20, 16, 50, 50)
        _ = fx_onnx.export(TraceModel(), x, op_level_debug=True)
        assert_has_diagnostics(
            diagnostics.engine,
            diagnostics.rules.fx_node_to_onnx,
            diagnostics.levels.WARNING,
            expected_error_node="aten.convolution.default",
            expected_error_message="ValueError: You passed in an iterable attribute but I cannot figure out its applicable type.",
        )


if __name__ == "__main__":
    common_utils.run_tests()
