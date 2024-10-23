# Owner(s): ["oncall: distributed"]

import torch
import torch.nn as nn
from torch.distributed.tensor.debug._op_coverage import get_inductor_decomp_graphs
from torch.testing._internal.common_utils import run_tests, TestCase


class SimpleMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net1 = nn.Linear(50, 32)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(32, 8)

    def forward(self, x):
        return torch.sigmoid(self.net2(self.relu(self.net1(x))))


class TestOpCoverage(TestCase):
    def test_trace_with_inductor_decomp(self):
        model = SimpleMLP()
        args = (torch.randn(8, 50),)
        kwargs = {}
        graphs = get_inductor_decomp_graphs(model, args, kwargs)
        assert len(graphs) == 2, "Expect fwd + bwd graphs"
        self.assertIsInstance(graphs[0], torch.fx.GraphModule)
        self.assertIsInstance(graphs[1], torch.fx.GraphModule)


if __name__ == "__main__":
    run_tests()
