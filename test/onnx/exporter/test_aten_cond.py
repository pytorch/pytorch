# Owner(s): ["module: onnx"]
"""Unit tests for the onnx dynamo exporter."""

from __future__ import annotations

import torch
from torch.onnx._internal.exporter import _testing as onnx_testing
from torch.testing._internal import common_utils


@common_utils.instantiate_parametrized_tests
class DynamoExporterTest(common_utils.TestCase):
    def test_onnx_export_controlflow(self):
        class Bad1Fixed(torch.nn.Module):
            def forward(self, x):
                def true_fn(x):
                    return torch.sin(x)

                def false_fn(x):
                    return torch.cos(x)

                return torch.cond(x.sum() > 0, true_fn, false_fn, [x])

        x = torch.rand(5, 3)
        model = Bad1Fixed()
        onnx_program = torch.onnx.export(
            model,
            (x,),
            input_names=["x"],
            opset_version=18,
            dynamo=True,
            fallback=False,
        )
        onnx_testing.assert_onnx_program(onnx_program, atol=1e-3, rtol=1)


if __name__ == "__main__":
    common_utils.run_tests()
