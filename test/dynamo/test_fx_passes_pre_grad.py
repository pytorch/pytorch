# Owner(s): ["module: dynamo"]
from unittest import mock

import torch

import torch._dynamo
import torch._dynamo.test_case
from torch._inductor.utils import pass_execution_and_save


class FxPassesPreGradTests(torch._dynamo.test_case.TestCase):
    @mock.patch("torch._inductor.utils.ShapeProp.propagate")
    def test_pass_execution_and_save(self, mock_shape_prop):
        class TestModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.param = torch.nn.Parameter(torch.ones(4, 4))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.param + x

        def fx_pass(graph: torch.fx.GraphModule) -> None:
            return

        sample_input = torch.randn(4, 4)
        m = TestModule()
        m(sample_input)
        exported_program = torch.export.export(m, (sample_input,))
        gm = exported_program.graph_module

        pass_execution_and_save(fx_pass, gm, sample_input, "Apply testing pass")
        mock_shape_prop.assert_called_once()


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
