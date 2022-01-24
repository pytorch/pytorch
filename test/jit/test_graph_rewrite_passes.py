# Owner(s): ["oncall: jit"]

from torch.testing._internal.jit_utils import JitTestCase
import torch
import torch._C
from torch.testing import FileCheck
from typing import Callable

class FunctionalConv2d(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor):
        super(FunctionalConv2d, self).__init__()
        self.weight = weight
        self.bias = bias

    def forward(self, x: torch.Tensor):
        x = torch.nn.functional.conv2d(
            input=x.unsqueeze(dim=0),
            weight=self.weight,
            bias=self.bias,
        )
        return x

class FunctionalLinear(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor = None):
        super(FunctionalLinear, self).__init__()
        self.weight = weight
        self.bias = bias

    def forward(self, x: torch.Tensor):
        res = torch.matmul(x, self.weight.t())
        if self.bias is not None:
            res.add_(self.bias)
        return res

class FunctionalMatmul(torch.nn.Module):
    def __init__(self, weight: torch.Tensor):
        super(FunctionalMatmul, self).__init__()
        self.weight = weight

    def forward(self, x: torch.Tensor):
        return torch.matmul(x, self.weight)

class TestGraphRewritePasses(JitTestCase):
    def check_single_replacement(
        self,
        old_kind: str,
        new_kind: str,
        jit_pass: Callable[[str], None],
        model: torch.jit.ScriptModule
    ):
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
        check_yes: list[str],
        check_not: list[str],
        jit_pass: Callable[[str], None],
        model: torch.jit.ScriptModule
    ):
        jit_pass(model.graph)
        for cn in check_not:
            FileCheck().check_not(cn).run(model.graph)
        for cy in check_yes:
            FileCheck().check(cy).run(model.graph)

    def test_fuse_linear(self):
        x1 = torch.rand(3)
        w1 = torch.rand(5, 3)
        b1 = torch.rand(5)
        model1 = torch.jit.trace(FunctionalLinear(w1, b1), [x1])
        check_not = ["aten::matmul", "aten::addmm", "aten::add_", "aten::t("]
        self.check_single_replacement("aten::matmul", "aten::linear", torch._C._jit_pass_fuse_linear, model1)
        self.check_op_presence([], check_not, torch._C._jit_pass_fuse_linear, model1)
        model1(x1)  # make sure it runs

        model2 = torch.jit.trace(FunctionalLinear(w1, None), [x1])
        self.check_single_replacement("aten::matmul", "aten::linear", torch._C._jit_pass_fuse_linear, model2)
        self.check_op_presence([], check_not, torch._C._jit_pass_fuse_linear, model2)
        model2(x1)  # make sure it runs

        # check matmuls are not fused
        x3 = torch.rand(5, 6, 5)
        w3 = torch.rand(5, 5, 100)
        model3 = torch.jit.trace(FunctionalMatmul(w3), [x3])
        check_not3 = ["aten::linear"]
        self.check_single_replacement("aten::matmul", "aten::matmul", torch._C._jit_pass_fuse_linear, model3)
        self.check_op_presence([], check_not3, torch._C._jit_pass_fuse_linear, model3)
        model3(x3)  # make sure it runs

    def test_vulkan_insert_pre_packed_ops(self):
        x1 = torch.rand(3)
        w1 = torch.rand(5, 3)
        b1 = torch.rand(5)
        model1 = torch.jit.trace(FunctionalLinear(w1, b1), [x1])
        check_not1 = ["aten::matmul", "aten::add_", "aten::t"]
        self.check_single_replacement(
            "aten::matmul", 
            "vulkan_prepack::linear_run", 
            torch._C._jit_pass_vulkan_insert_prepacked_ops, 
            model1
        )
        self.check_op_presence([], check_not1, torch._C._jit_pass_vulkan_insert_prepacked_ops, model1)
        model1(x1)  # make sure it runs

        conv2d_in_channels = 3
        conv2d_out_channels = 4
        conv2d_kernel = 2
        conv2d_weight = torch.rand(
            conv2d_out_channels,
            conv2d_in_channels,
            conv2d_kernel,
            conv2d_kernel,
        )
        conv2d_bias = torch.rand(conv2d_out_channels)
        x2_shape = (3, 2, 5)
        x2 = torch.rand(x2_shape)
        model2 = torch.jit.trace(FunctionalConv2d(conv2d_weight, conv2d_bias), [x2])
        check_not2 = ["aten::_convolution"]
        check_yes2 = ["vulkan_prepack::conv2d_clamp_run", "vulkan_prepack::conv2d_clamp_prepack"]
        self.check_single_replacement(
            "aten::_convolution", 
            "prim::Constant", 
            torch._C._jit_pass_vulkan_insert_prepacked_ops, 
            model2
        )
        self.check_op_presence(check_yes2, check_not2, torch._C._jit_pass_vulkan_insert_prepacked_ops, model2)
        model2(x2)  # make sure it runs
