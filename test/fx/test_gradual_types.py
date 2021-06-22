import unittest

import torch
from torch.fx import symbolic_trace
from torch.fx.tensor_type import Tensor_Type, Dyn, consistency

class gradual_type_test(unittest.TestCase):
    def test_consistenty(self):
        """
        Test the consistency relation.
        """
        self.assertTrue(consistency(Tensor_Type((1, 2, 3)), Tensor_Type((1, Dyn, 3))))
        self.assertTrue(consistency(int, Dyn))
        self.assertTrue(consistency(int, int))
        self.assertFalse(consistency(Tensor_Type((1, 2, 3)), Tensor_Type((1, 2, 3, 5))))
        self.assertFalse(consistency(Tensor_Type((1, 2, 3)), int))

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


if __name__ == '__main__':
    unittest.main()
