# Owner(s): ["oncall: fx"]

import unittest
from torch.fx import GraphModule
from torch.fx.experimental.migrate_gradual_types.constraint_generator import ConstraintGenerator
from torch.fx.experimental.rewriter import RewritingTracer
from torch.fx.tensor_type import Dyn
import torch

class ConstraintGeneration(unittest.TestCase):

    def test_add_reshape(self):
        class BasicBlock(torch.nn.Module):
            def __init__(self):
                super(BasicBlock, self).__init__()

            def forward(self, x: Dyn, y: Dyn):
                return torch.add(torch.reshape(x, (1, 2)), torch.reshape(y, (2, 2)))

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        generator = ConstraintGenerator(traced)
        new_constraints, counter = generator.generate_constraints(0)
        assert len(new_constraints.conjucts) == 11


    def test_conv_reshape_add(self):
        class BasicBlock(torch.nn.Module):
            def __init__(self, in_planes, out_planes, kernel_size, stride, padding, groups, dilation):
                super(BasicBlock, self).__init__()
                self.conv1 = torch.nn.Conv2d(in_channels=in_planes, out_channels=out_planes,
                                             kernel_size=kernel_size, stride=stride,
                                             padding=padding, groups=groups, bias=False, dilation=dilation)

            def forward(self, x: Dyn, y: Dyn):
                return torch.add(self.conv1(torch.reshape(x, (1, 2, 10, 20))), y)

        B = BasicBlock(2, 2, 2, 3, 2, 2, 2)
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        generator = ConstraintGenerator(traced)
        new_constraints, counter = generator.generate_constraints(0)
        assert len(new_constraints.conjucts) == 16


if __name__ == '__main__':
    unittest.main()
