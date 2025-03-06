# Owner(s): ["module: onnx"]
"""Unit tests for the _capture_strategies module."""

from __future__ import annotations

import torch
from torch.onnx._internal.exporter import _capture_strategies
from torch.testing._internal import common_utils


@common_utils.instantiate_parametrized_tests
class ExportStrategiesTest(common_utils.TestCase):
    @common_utils.parametrize(
        "strategy_cls",
        [
            _capture_strategies.TorchExportStrategy,
            _capture_strategies.TorchExportNonStrictStrategy,
            _capture_strategies.JitTraceConvertStrategy,
        ],
        name_fn=lambda strategy_cls: strategy_cls.__name__,
    )
    def test_jit_isinstance(self, strategy_cls):
        class Model(torch.nn.Module):
            def forward(self, a, b):
                if torch.jit.isinstance(a, torch.Tensor):
                    return a.cos()
                return b.sin()

        model = Model()
        a = torch.tensor(0.0)
        b = torch.tensor(1.0)

        result = strategy_cls()(model, (a, b), kwargs=None, dynamic_shapes=None)
        if result.exception:
            raise result.exception
        ep = result.exported_program
        assert ep is not None
        torch.testing.assert_close(ep.module()(a, b), model(a, b))


    def test_jit_module_with_dynamic_shapes(self):
        class Model(torch.nn.Module):
            def forward(self, a, b):
                c = torch.relu(a)
                return c + b

        model = Model()
        a = torch.tensor([[0.0]])
        b = torch.tensor([[1.0]])

        strategy = _capture_strategies.JitTraceConvertStrategy()
        batch_dim = torch.export.Dim("batch_dim")
        dynamic_shapes = (0: batch_dim }, {0: batch_dim})
        result = strategy(model, (a, b), kwargs=None, dynamic_shapes=dynamic_shapes)
        if result.exception:
            raise result.exception
        ep = result.exported_program
        assert ep is not None
        torch.testing.assert_close(ep.module()(a, b), model(a, b))


if __name__ == "__main__":
    common_utils.run_tests()
