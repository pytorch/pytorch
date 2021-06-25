import unittest
import torch
from torch.fx import symbolic_trace
from torch.fx.tensor_type import TensorType, Dyn, is_consistent, is_more_precise
from torch.fx.annotate import annotate
from torch.fx.experimental.graph_gradual_typechecker import type_check


class AnnotationsTest(unittest.TestCase):

    def test_annotations(self):
        """
        Test type annotations in the forward function.
        The annoation should appear in the n.graph
        where n is the corresoinding node in the resulting graph.
        """
        class M(torch.nn.Module):
            def forward(self, x: TensorType((1, 2, 3, Dyn)), y: Dyn):
                return torch.add(x, y)

        module = M()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)

        expected_ph_types = [TensorType((1, 2, 3, Dyn)), Dyn]
        expected_iter = iter(expected_ph_types)

        for n in symbolic_traced.graph.nodes:
            if n.op == 'placeholder':
                assert n.type == next(expected_iter)

    def test_annotate(self):
        class M(torch.nn.Module):

            def forward(self, x):
                y = annotate(x, TensorType[1, 2, 3, Dyn])
                return torch.add(x, y)

        module = M()
        symbolic_traced : torch.fx.GraphModule = symbolic_trace(module)
        for n in symbolic_traced.graph.nodes:
            if n.op == 'placeholder':
                assert n.type == TensorType[1, 2, 3, Dyn]
#
    def test_consistency(self):
        """
        Test the consistency relation.
        """
        self.assertTrue(is_consistent(TensorType((1, 2, 3)), TensorType((1, Dyn, 3))))
        self.assertTrue(is_consistent(int, Dyn))
        self.assertTrue(is_consistent(int, int))
        self.assertFalse(is_consistent(TensorType((1, 2, 3)), TensorType((1, 2, 3, 5))))
        self.assertFalse(is_consistent(TensorType((1, 2, 3)), int))

    def test_precision(self):
        """
        Test the consistency relation.
        """
        self.assertTrue(is_more_precise(TensorType((1, 2, 3)), TensorType((1, Dyn, 3))))
        self.assertTrue(is_more_precise(int, Dyn))
        self.assertTrue(is_more_precise(int, int))
        self.assertFalse(is_more_precise(TensorType((1, 2, 3)), TensorType((1, 2, 3, 5))))
        self.assertFalse(is_more_precise(TensorType((1, 2, 3)), int))


class TypeCheckerTest(unittest.TestCase):

    def test_type_check_add_false(self):
        class M(torch.nn.Module):
            def forward(self, x: TensorType((1, 2, 3, Dyn)), y: TensorType((1, 2, 3))):
                return torch.add(x, y)
        module = M()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        self.assertFalse(type_check(symbolic_traced.graph))

    def test_type_check_add_true(self):
        class M(torch.nn.Module):
            def forward(self, x: TensorType((1, 2, Dyn)), y: TensorType((1, 2, 3))):
                return torch.add(x, y)
        module = M()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        self.assertTrue(type_check(symbolic_traced.graph))

        expected_ph_types = [TensorType((1, 2, Dyn)), TensorType((1, 2, 3))]
        expected_iter = iter(expected_ph_types)

        for n in symbolic_traced.graph.nodes:
            if n.op == 'placeholder':
                assert n.type == next(expected_iter)
            if n.op == 'output':
                assert n.type == TensorType((1, 2, 3))

    def test_type_check_reshape_true(self):
        class M(torch.nn.Module):
            def forward(self, x: TensorType((1, 6))):
                return torch.reshape(x, [1, 2, 3])

        module = M()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        self.assertTrue(type_check(symbolic_traced.graph))

        for n in symbolic_traced.graph.nodes:
            if n.op == 'placeholder':
                assert n.type == TensorType((1, 6))

            if n.op == 'callfunction':
                assert n.type == TensorType((1, 2, 3))

            if n.op == 'output':
                assert n.type == TensorType((1, 2, 3))

    def test_type_check_reshape_false(self):
        class M(torch.nn.Module):
            def forward(self, x: TensorType((1, 5))):
                return torch.reshape(x, [1, 2, 3])

        module = M()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        self.assertFalse(type_check(symbolic_traced.graph))

    def test_type_check_reshape_dyn_false(self):
        class M(torch.nn.Module):
            def forward(self, x: TensorType((1, 5))):
                return torch.reshape(x, [1, 2, -1])

        module = M()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        self.assertFalse(type_check(symbolic_traced.graph))

    def test_type_check_reshape_dyn_true(self):
        class M(torch.nn.Module):
            def forward(self, x: TensorType((1, 15))):
                return torch.reshape(x, [1, 5, -1])

        module = M()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        self.assertTrue(type_check(symbolic_traced.graph))

    def test_type_check_reshape_dyn_true_param_false(self):
        class M(torch.nn.Module):
            def forward(self, x: TensorType((Dyn, 5))):
                return torch.reshape(x, [1, 2, -1])

        module = M()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        self.assertFalse(type_check(symbolic_traced.graph))


    def test_type_check_transpose_true(self):
        class M(torch.nn.Module):
            def forward(self, x: TensorType((1, 2, 3, 5))):
                return torch.transpose(x, 0, 1)

        module = M()
        symbolic_traced : torch.fx.GraphModule = symbolic_trace(module)
        self.assertTrue(type_check(symbolic_traced.graph))

        for n in symbolic_traced.graph.nodes:
            if n.op == 'callfunction':
                assert n.type == TensorType([2, 1, 3, 5])
            if n.op == 'output':
                assert n.type == TensorType([2, 1, 3, 5])
            if n.op == 'x':
                assert n.placeholder == TensorType([1, 2, 3, 5])

    def test_type_check_transpose_False(self):
        class M(torch.nn.Module):
            def forward(self, x: TensorType((1, 2, 3, 5))):
                return torch.transpose(x, 0, 10)

        module = M()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        self.assertFalse(type_check(symbolic_traced.graph))


if __name__ == '__main__':
    unittest.main()
