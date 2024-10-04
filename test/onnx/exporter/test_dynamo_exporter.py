# Owner(s): ["module: onnx"]
"""Unit tests for the onnx dynamo exporter."""

from __future__ import annotations

import torch
from torch.onnx._internal.exporter import _testing as onnx_testing
from torch.testing._internal import common_utils


@common_utils.instantiate_parametrized_tests
class DynamoExporterTest(common_utils.TestCase):
    def test_insert_flatten_between_transpose_and_view(self):
        class DummyModel(torch.nn.Module):
            def __init__(self, enable_math: bool):
                super().__init__()
                self.enable_math = False

            def forward(self, query, key, value):
                res = torch.nn.functional.scaled_dot_product_attention(
                    query, key, value
                )
                rest = res.transpose(0, 1)
                final = rest.view(8, 32, 128 * 64)
                return final

        model = DummyModel(False)
        device = "cpu"

        query = torch.rand(32, 8, 128, 64, dtype=torch.float16, device=device)
        key = torch.rand(32, 8, 128, 64, dtype=torch.float16, device=device)
        value = torch.rand(32, 8, 128, 64, dtype=torch.float16, device=device)

        onnx_program = torch.onnx.export(
            model, (query, key, value), dynamo=True, fallback=False
        )
        onnx_testing.assert_onnx_program(onnx_program, atol=1e-3, rtol=1)


if __name__ == "__main__":
    common_utils.run_tests()
