import unittest
import torch
from torch.fx import symbolic_trace
from torch.fx.tensor_type import TensorType, Dyn, is_consistent, is_more_precise
from torch.fx.annotate import annotate
from torch.fx.experimental.graph_gradual_typechecker import GraphTypeChecker, broadcast_types


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
                y = annotate(x, TensorType((1, 2, 3, Dyn)))
                return torch.add(x, y)

        module = M()
        symbolic_traced : torch.fx.GraphModule = symbolic_trace(module)
        for n in symbolic_traced.graph.nodes:
            if n.op == 'placeholder':
                assert n.type == TensorType((1, 2, 3, Dyn))

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

    def test_broadcasting1(self):
        t1 = TensorType((1, 2, 3, 4))
        t2 = TensorType((1, 2, 1, 4))
        assert broadcast_types(t1, t2) == (TensorType((1, 2, 3, 4)), TensorType((1, 2, 3, 4)))

    def test_broadcasting2(self):
        t1 = TensorType((2, 3, 4))
        t2 = TensorType((1, 2, 1, 4))

        with self.assertRaises(TypeError):
            broadcast_types(t1, t2)

    def test_broadcasting3(self):
        t1 = TensorType((1, 2, 3, Dyn))
        t2 = TensorType((2, 3, 4))
        assert broadcast_types(t1, t2) == (TensorType((1, 2, 3, Dyn)), TensorType((1, 2, 3, 4)))


class TypeCheckerTest(unittest.TestCase):

    def test_type_check_add_with_broadcast(self):
        class M(torch.nn.Module):
            def forward(self, x: TensorType((1, 2, 3, Dyn)), y: TensorType((2, 3, 4))):
                return torch.add(x, y)
        module = M()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        tc = GraphTypeChecker({}, symbolic_traced)
        tc.type_check()
        expected_ph_types = [TensorType((1, 2, 3, Dyn)),
                             TensorType((1, 2, 3, 4)),
                             TensorType((1, 2, 3, Dyn)),
                             TensorType((1, 2, 3, Dyn))]
        expected_iter = iter(expected_ph_types)

        for n in symbolic_traced.graph.nodes:
            assert n.type == next(expected_iter)

    def test_type_check_add_with_scalar(self):
        class M(torch.nn.Module):
            def forward(self, x: int, y: TensorType((2, 3, 4))):
                return torch.add(x, y)
        module = M()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        tc = GraphTypeChecker({}, symbolic_traced)
        tc.type_check()
        expected_ph_types = [int,
                             TensorType((2, 3, 4)),
                             TensorType((2, 3, 4)),
                             TensorType((2, 3, 4))]
        expected_iter = iter(expected_ph_types)

        for n in symbolic_traced.graph.nodes:
            assert n.type == next(expected_iter)

    def test_type_check_add_false(self):
        class M(torch.nn.Module):
            def forward(self, x: TensorType((1, 2, 3, Dyn)), y: TensorType((1, 2, 3))):
                return torch.add(x, y)
        module = M()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        tc = GraphTypeChecker({}, symbolic_traced)
        with self.assertRaises(TypeError):
            tc.type_check()

    def test_type_check_add_true(self):
        class M(torch.nn.Module):
            def forward(self, x: TensorType((1, 2, Dyn)), y: TensorType((1, 2, 3))):
                return torch.add(x, y)
        module = M()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        tc = GraphTypeChecker({}, symbolic_traced)
        self.assertTrue(tc.type_check())

        expected_ph_types = [TensorType((1, 2, Dyn)), TensorType((1, 2, 3))]
        expected_iter = iter(expected_ph_types)

        for n in symbolic_traced.graph.nodes:
            if n.op == 'placeholder':
                assert n.type == next(expected_iter)
            if n.op == 'output':
                assert n.type == TensorType((1, 2, Dyn))

    def test_type_check_reshape_true(self):
        class M(torch.nn.Module):
            def forward(self, x: TensorType((1, 6))):
                return torch.reshape(x, [1, 2, 3])

        module = M()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        tc = GraphTypeChecker({}, symbolic_traced)
        self.assertTrue(tc.type_check())

        for n in symbolic_traced.graph.nodes:
            if n.op == 'placeholder':
                assert n.type == TensorType((1, 6))

            if n.op == 'call_function':
                assert n.type == TensorType((1, 2, 3))

            if n.op == 'output':
                assert n.type == TensorType((1, 2, 3))

    def test_type_check_reshape_false(self):
        class M(torch.nn.Module):
            def forward(self, x: TensorType((1, 5))):
                return torch.reshape(x, [1, 2, 3])

        module = M()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        tc = GraphTypeChecker({}, symbolic_traced)
        with self.assertRaises(TypeError):
            tc.type_check()

    def test_type_check_reshape_dyn_false(self):
        class M(torch.nn.Module):
            def forward(self, x: TensorType((1, 5))):
                return torch.reshape(x, [1, 2, -1])

        module = M()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        tc = GraphTypeChecker({}, symbolic_traced)
        with self.assertRaises(TypeError):
            tc.type_check()

    def test_type_check_reshape_dyn_true(self):
        class M(torch.nn.Module):
            def forward(self, x: TensorType((1, 15))):
                return torch.reshape(x, [1, 5, -1])

        module = M()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        tc = GraphTypeChecker({}, symbolic_traced)
        self.assertTrue(tc.type_check())

    def test_type_check_reshape_dyn_true_param_false(self):
        class M(torch.nn.Module):
            def forward(self, x: TensorType((Dyn, 5))):
                return torch.reshape(x, [1, 2, -1])

        module = M()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        tc = GraphTypeChecker({}, symbolic_traced)
        with self.assertRaises(TypeError):
            tc.type_check()

    def test_type_check_transpose_true(self):
        class M(torch.nn.Module):
            def forward(self, x: TensorType((1, 2, 3, 5))):
                return torch.transpose(x, 0, 1)

        module = M()
        symbolic_traced : torch.fx.GraphModule = symbolic_trace(module)
        tc = GraphTypeChecker({}, symbolic_traced)
        self.assertTrue(tc.type_check())

        for n in symbolic_traced.graph.nodes:
            if n.op == 'call_function':
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
        tc = GraphTypeChecker({}, symbolic_traced)
        with self.assertRaises(TypeError):
            tc.type_check()


if __name__ == '__main__':
    unittest.main()
