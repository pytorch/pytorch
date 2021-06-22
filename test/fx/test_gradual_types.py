import unittest

import torch
from torch.fx import symbolic_trace
from torch.fx.tensor_type import Tensor_Type, Dyn, consistency
from torch.fx import annotate
from torch.fx.experimental.graph_gradual_typechecker import type_check

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


    def test_type_check_add_true(self):
        class M(torch.nn.Module):
            def forward(self, x:Tensor_Type((1, 2, 3, Dyn)), y: Tensor_Type((1, 2, 3))):
                return torch.add(x, y)
        module = M()
        symbolic_traced : torch.fx.GraphModule = symbolic_trace(module)
        self.assertFalse(type_check(symbolic_traced.graph))

    def test_type_check_add_false(self):
        class M(torch.nn.Module):
            def forward(self, x: Tensor_Type((1, 2, Dyn)), y: Tensor_Type((1, 2, 3))):
                return torch.add(x, y)
        module = M()
        symbolic_traced : torch.fx.GraphModule = symbolic_trace(module)
        self.assertTrue(type_check(symbolic_traced.graph))


        for n in symbolic_traced.graph.nodes:
            if n.name == 'x':
                assert n.type == Tensor_Type((1, 2, Dyn))
            if n.name == 'y':
                assert n.type == Tensor_Type((1, 2, 3))
            if n.name == 'output':
                assert n.type == Tensor_Type((1, 2, Dyn))

    def test_type_check_reshape_true(self):
        class M(torch.nn.Module):
            def forward(self, x:Tensor_Type((1,6))):
                return torch.reshape(x, [1,2,3])

        module = M()
        symbolic_traced : torch.fx.GraphModule = symbolic_trace(module)
        self.assertTrue(type_check(symbolic_traced.graph))


        for n in symbolic_traced.graph.nodes:
            if n.name == 'x':
                assert n.type == Tensor_Type((1, 6))

            if n.name == 'reshape':
                assert n.type == Tensor_Type((1, 2, 3))

            if n.name == 'output':
                assert n.type == Tensor_Type((1, 2, 3))

    def test_type_check_reshape_false(self):
        class M(torch.nn.Module):
            def forward(self, x:Tensor_Type((1, 5))):
                return torch.reshape(x, [1, 2, 3])

        module = M()
        symbolic_traced : torch.fx.GraphModule = symbolic_trace(module)
        self.assertFalse(type_check(symbolic_traced.graph))

    def test_type_check_transpose_true(self):
        class M(torch.nn.Module):
            def forward(self, x:Tensor_Type((1, 2, 3, 5))):
                return torch.transpose(x,0,1)

        module = M()
        symbolic_traced : torch.fx.GraphModule = symbolic_trace(module)
        self.assertTrue(type_check(symbolic_traced.graph))

        for n in symbolic_traced.graph.nodes:
            if n.name == 'transpose':
                assert n.type == Tensor_Type([2, 1, 3, 5])
            if n.name == 'output':
                assert n.type == Tensor_Type([2, 1, 3, 5])
            if n.name == 'x':
                assert n.type == Tensor_Type([1, 2, 3, 5])

    def test_type_check_transpose_False(self):
        class M(torch.nn.Module):
            def forward(self, x:Tensor_Type((1, 2, 3, 5))):
                return torch.transpose(x,0,10)

        module = M()
        symbolic_traced : torch.fx.GraphModule = symbolic_trace(module)
        self.assertFalse(type_check(symbolic_traced.graph))


if __name__ == '__main__':
    unittest.main()
