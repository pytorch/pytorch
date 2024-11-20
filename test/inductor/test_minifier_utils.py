# Owner(s): ["module: inductor"]
import torch
from torch._dynamo.repro.aoti import export_for_aoti_minifier
from torch.testing._internal.common_utils import run_tests, TestCase


class MinifierUtilsTests(TestCase):
    def test_invalid_output(self):
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, x):
                # return a graph module
                return self.linear

        model = SimpleModel()
        # Here we obtained a graph with invalid output by symbolic_trace for simplicity,
        # it can also obtained from running functorch.compile.minifier on an exported graph.
        traced = torch.fx.symbolic_trace(model)
        gm = export_for_aoti_minifier(traced, (torch.randn(2, 2),))
        self.assertTrue(gm is None)


if __name__ == "__main__":
    run_tests()
