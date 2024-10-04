# Owner(s): ["module: onnx"]
"""Unit tests for the onnx dynamo exporter."""

from __future__ import annotations

import numpy as np
import onnx.reference as rt

import torch
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
        expected = model(query, key, value)

        onnx_program = torch.onnx.export(
            model, (query, key, value), dynamo=True, fallback=False
        )
        proto = onnx_program.model_proto
        ref = rt.ReferenceEvaluator(proto)
        got = ref.run(
            None, {"query": query.numpy(), "key": key.numpy(), "value": value.numpy()}
        )
        np.testing.assert_allclose(expected.numpy(), got[0], atol=1e-3)


if __name__ == "__main__":
    common_utils.run_tests()
