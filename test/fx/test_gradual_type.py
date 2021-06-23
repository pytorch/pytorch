import unittest
import torch
from torch.fx import symbolic_trace
from torch.fx.tensor_type import Tensor_Type, Dyn
from torch.fx.annotate import annotate


class AnnotationsTest(unittest.TestCase):

    def test_annotations(self):
        """
        Test type annotations in the forward function.
        The annoation should appear in the n.graph
        where n is the corresoinding node in the resulting graph.
        """
        class M(torch.nn.Module):
            def forward(self, x: Tensor_Type((1, 2, 3, Dyn)), y: Dyn):
                return torch.add(x, y)

        module = M()
        symbolic_traced : torch.fx.GraphModule = symbolic_trace(module)

        for n in symbolic_traced.graph.nodes:
            if n.name == 'x':
                assert n.type == Tensor_Type((1, 2, 3, Dyn))
            if n.name == 'y':
                assert n.type == Dyn

    def test_annotate(self):
        class M(torch.nn.Module):

            def forward(self, x):
                y = annotate(x, Tensor_Type((1, 2, 3, Dyn)))
                return torch.add(x, y)

        module = M()
        symbolic_traced : torch.fx.GraphModule = symbolic_trace(module)
        for n in symbolic_traced.graph.nodes:
            if n.name == 'x':
                assert n.type == Tensor_Type((1, 2, 3, Dyn))


if __name__ == '__main__':
    unittest.main()
