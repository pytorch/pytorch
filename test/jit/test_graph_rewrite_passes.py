# Owner(s): ["oncall: jit"]

from torch.testing._internal.jit_utils import JitTestCase
import torch
import torch._C
from torch.testing import FileCheck
from typing import Callable, Dict

class FunctionalConv2d(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor) -> None:
        super(FunctionalConv2d, self).__init__()
        self.weight = weight
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.conv2d(
            input=x.unsqueeze(dim=0),
            weight=self.weight,
            bias=self.bias,
        )
        return x

class FunctionalLinear(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor = None) -> None:
        super(FunctionalLinear, self).__init__()
        self.weight = weight
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = torch.matmul(x, self.weight.t())
        if self.bias is not None:
            res.add_(self.bias)
        return res

class FunctionalMatmul(torch.nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super(FunctionalMatmul, self).__init__()
        self.weight = weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight)

class TestGraphRewritePasses(JitTestCase):
    def check_single_replacement(
        self,
        old_kind: str,
        new_kind: str,
        jit_pass: Callable[[str], None],
        model: torch.jit.ScriptModule
    ) -> None:
        for node in model.graph.nodes():
            if node.kind() == old_kind:
                source_range_1 = node.sourceRange()
        jit_pass(model.graph)
        for node in model.graph.nodes():
            if node.kind() == new_kind:
                source_range_2 = node.sourceRange()
        self.assertTrue(source_range_1 == source_range_2)

    def check_op_presence(
        self,
        pattern_count_map: Dict[str, int],
        jit_pass: Callable[[str], None],
        model: torch.jit.ScriptModule
    ) -> None:
        jit_pass(model.graph)
        for pattern, v in pattern_count_map.items():
            FileCheck().check_count(pattern, v, exactly=True).run(model.graph)

    def test_fuse_linear(self) -> None:
        x_1 = torch.rand(3)
        w_1 = torch.rand(5, 3)
        b_1 = torch.rand(5)
        model_1 = torch.jit.trace(FunctionalLinear(w_1, b_1), [x_1])
        check_pattern_count_map_1 = {"aten::matmul": 0, "aten::addmm": 0, "aten::add_": 0, "aten::t(": 0}
        self.check_single_replacement("aten::matmul", "aten::linear", torch._C._jit_pass_fuse_linear, model_1)
        self.check_op_presence(check_pattern_count_map_1, torch._C._jit_pass_fuse_linear, model_1)
        model_1(x_1)  # make sure it runs

        model_2 = torch.jit.trace(FunctionalLinear(w_1, None), [x_1])
        self.check_single_replacement("aten::matmul", "aten::linear", torch._C._jit_pass_fuse_linear, model_2)
        self.check_op_presence(check_pattern_count_map_1, torch._C._jit_pass_fuse_linear, model_2)
        model_2(x_1)  # make sure it runs

        # check matmuls are not fused
        x_3 = torch.rand(5, 6, 5)
        w_3 = torch.rand(5, 5, 100)
        model_3 = torch.jit.trace(FunctionalMatmul(w_3), [x_3])
        check_pattern_count_map_3 = {"aten::linear": 0}
        self.check_single_replacement("aten::matmul", "aten::matmul", torch._C._jit_pass_fuse_linear, model_3)
        self.check_op_presence(check_pattern_count_map_3, torch._C._jit_pass_fuse_linear, model_3)
        model_3(x_3)  # make sure it runs
