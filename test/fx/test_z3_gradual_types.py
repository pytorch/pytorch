# Owner(s): ["module: fx"]
import operator
import unittest
from torch.fx import GraphModule, symbolic_trace
from torch.fx.experimental.meta_tracer import symbolic_trace as meta_symbolic_trace
from torch.fx.experimental.migrate_gradual_types.constraint import BinConstraintT, DVar, TVar, T
from torch.fx.experimental.migrate_gradual_types.constraint_generator import ConstraintGenerator
from torch.fx.experimental.migrate_gradual_types.constraint_transformation import transform_constraint
from torch.fx.experimental.migrate_gradual_types.operation import op_precision, op_matching, op_consistency
from torch.fx.experimental.migrate_gradual_types.transform_to_z3 import transform_all_constraints, \
    evaluate_conditional_with_constraints
from torch.fx.experimental.migrate_gradual_types.z3_types import tensor_type, D, z3_dyn
from torch.fx.experimental.rewriter import RewritingTracer
from torch.fx.tensor_type import Dyn, TensorType
import torch


try:
    import z3  # type: ignore[import]
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False


try:
    from torchvision import models
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")

class TorchDynamoUseCases(unittest.TestCase):

    def test_dim(self):
        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([1, 2])):
                y = x.dim()
                return y

        symbolic_traced: torch.fx.GraphModule = symbolic_trace(BasicBlock())
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        y_res = z3.z3.Int(2)
        self.assertEqual(s.model()[y_res], 2)


    def test_reshape(self):
        """
        In this example, we prove that some nodes must
        always have a fixed shape regardless of the input
        """

        class BasicBlock(torch.nn.Module):
            def forward(self, x: Dyn):
                y = x.view(100)
                tmp = y.size()[0]
                return tmp

        symbolic_traced: torch.fx.GraphModule = symbolic_trace(BasicBlock())
        transformed = transform_all_constraints(symbolic_traced, counter=0)

        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        dim = z3.Int(4)
        self.assertEqual(s.model()[dim], 100)
        # print(s.model()[dim])


class HFOperations(unittest.TestCase):

    def test_eq_dim(self):
        """
        test dimensions and equalities
        """
        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([32, 4, 4])):
                eq = x.dim() == 3
                return eq

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())

        # The node we are considering is the gt node
        for n in graph.nodes:
            if n.target == operator.eq:
                node = n

        positive, negative = evaluate_conditional_with_constraints(ast_rewriter.root, graph, node)
        self.assertEqual(positive, z3.sat)
        self.assertEqual(negative, z3.unsat)

    def test_conditional_ne_1(self):
        """
        This test case is for the HFmodels interface.
        A function takes a node and a graph and considers
        the conditional the node represents and its negation
        and solves each formula with the remaining sets of constraints
        Returns:

        """
        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([32, 4, 4]), y: TensorType([32, 4, 4])):
                size_5 = x.size()
                getitem_7 = size_5[0]
                getitem_8 = size_5[1]
                getitem_9 = size_5[2]
                ne_1 = y != (getitem_7, getitem_8, getitem_9)
                return ne_1

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())

        # The node we are considering is the gt node
        for n in graph.nodes:
            if n.target == operator.ne:
                node = n

        # since x and y are equal, the requirement that x != y cannot be true, so we should get unsat
        # for the positive condition and sat for the negative condition
        positive, negative = evaluate_conditional_with_constraints(ast_rewriter.root, graph, node)
        self.assertEqual(positive, z3.unsat)
        self.assertEqual(negative, z3.sat)

    def test_bmm(self):
        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([Dyn, 2, 3]), y: TensorType([1, 3, 2])):
                bmm = torch.bmm(x, y)
                return bmm

        symbolic_traced: torch.fx.GraphModule = symbolic_trace(BasicBlock())
        b = BasicBlock().forward(torch.rand(1, 2, 3), torch.rand(1, 3, 2))
        transformed = transform_all_constraints(symbolic_traced, counter=0)

        s = z3.Solver()
        s.add(transformed)

        output = z3.Const(3, tensor_type)
        self.assertEqual(s.check(), z3.sat)
        self.assertEqual(s.model()[output].arg(0).arg(1), b.shape[0])
        self.assertEqual(s.model()[output].arg(1).arg(1), b.shape[1])
        self.assertEqual(s.model()[output].arg(2).arg(1), b.shape[2])


    def test_bmm2(self):
        class BasicBlock(torch.nn.Module):
            def forward(self, x: Dyn, y: TensorType([1, 3, 2])):
                bmm = torch.bmm(x, y)
                return bmm

        symbolic_traced: torch.fx.GraphModule = symbolic_trace(BasicBlock())
        b = BasicBlock().forward(torch.rand(1, 2, 3), torch.rand(1, 3, 2))
        transformed = transform_all_constraints(symbolic_traced, counter=0)

        s = z3.Solver()
        s.add(transformed)

        output = z3.Const(3, tensor_type)
        self.assertEqual(s.check(), z3.sat)
        self.assertEqual(s.model()[output].arg(0).arg(1), b.shape[0])
        self.assertEqual(s.model()[output].arg(1).arg(0), 0)
        self.assertEqual(s.model()[output].arg(2).arg(1), b.shape[2])

    def test_bmm3(self):
        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([2, 3, 3]), y: TensorType([1, 3, 2])):
                bmm = torch.bmm(x, y)
                return bmm

        symbolic_traced: torch.fx.GraphModule = symbolic_trace(BasicBlock())
        transformed = transform_all_constraints(symbolic_traced, counter=0)

        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.unsat)


    def test_transpose(self):
        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([1, 2, 3, 4])):
                transpose = x.transpose(0, 1)
                return transpose

        symbolic_traced: torch.fx.GraphModule = symbolic_trace(BasicBlock())
        b = BasicBlock().forward(torch.rand(1, 2, 3, 4))

        transformed = transform_all_constraints(symbolic_traced, counter=0)

        s = z3.Solver()
        s.add(transformed)
        output = z3.Const(2, tensor_type)
        self.assertEqual(s.check(), z3.sat)
        self.assertEqual(s.model()[output].arg(0).arg(1), b.shape[0])
        self.assertEqual(s.model()[output].arg(1).arg(1), b.shape[1])
        self.assertEqual(s.model()[output].arg(2).arg(1), b.shape[2])
        self.assertEqual(s.model()[output].arg(3).arg(1), b.shape[3])

        # change the annotation to Dyn
        for n in symbolic_traced.graph.nodes:
            if n.op == 'placeholder':
                n.type = Dyn

        transformed = transform_all_constraints(symbolic_traced, counter=0)

        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)


    def test_index_select(self):
        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([2050, 1024]), y: Dyn):
                index_select = x.index_select(0, y)
                return index_select
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(BasicBlock())
        # print(symbolic_traced)
        b = BasicBlock().forward(torch.rand(2050, 1024), torch.ones(8).int())
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        index_select = z3.Const(3, tensor_type)

        # the second dimension of the result should not be affected since
        # the index is 0
        self.assertEqual(s.model()[index_select].arg(1).arg(1), b.shape[1])

        replacement_vector = z3.Const(2, tensor_type)

        # we set the vector to Dyn
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        index_select = z3.Const(3, tensor_type)
        s.add(replacement_vector == z3_dyn)
        self.assertEqual(s.check(), z3.sat)

        # this implies that the index at 0 should be dyn
        self.assertEqual(s.model()[index_select].arg(0).arg(0), 0)

    def test_get_attr(self):
        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([1, 2, 3])):
                getattr = x.device
                to = x.to(getattr)
                return to

        symbolic_traced: torch.fx.GraphModule = symbolic_trace(BasicBlock())
        b = BasicBlock().forward(torch.rand(1, 2, 3))
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        attr_res = z3.Const(3, tensor_type)
        assert s.model()[attr_res].arg(0).arg(1) == b.shape[0]
        assert s.model()[attr_res].arg(1).arg(1) == b.shape[1]
        assert s.model()[attr_res].arg(2).arg(1) == b.shape[2]


    def test_expand(self):
        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([1, 4])):
                size = x.size()
                getitem = size[-1]
                expand = x.expand(getitem, 4)
                return expand

        b = BasicBlock().forward(torch.rand(1, 4))

        symbolic_traced: torch.fx.GraphModule = symbolic_trace(BasicBlock())
        transformed = transform_all_constraints(symbolic_traced, counter=0)

        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        expand_res = z3.Const(4, tensor_type)
        assert s.model()[expand_res].arg(0).arg(1) == b.shape[0]
        assert s.model()[expand_res].arg(1).arg(1) == b.shape[1]

        # change the annotation on the input to Dyn.
        # the last dimension should still be 4
        for n in symbolic_traced.graph.nodes:
            if n.op == 'placeholder':
                n.type = Dyn

        transformed = transform_all_constraints(symbolic_traced, counter=0)

        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

        assert s.model()[expand_res].arg(1).arg(1) == b.shape[1]

    def test_getitem_tensor(self):
        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([4, 4])):
                getitem = x[(None, None, slice(None, None, None), slice(None, None, None))]
                return getitem

        B = BasicBlock()
        b = B.forward(torch.rand(4, 4))

        symbolic_traced: torch.fx.GraphModule = symbolic_trace(B)
        transformed = transform_all_constraints(symbolic_traced, counter=0)

        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        get_item_res = z3.Const(2, tensor_type)
        assert s.model()[get_item_res].arg(0).arg(1) == b.shape[0]
        assert s.model()[get_item_res].arg(1).arg(1) == b.shape[1]
        assert s.model()[get_item_res].arg(2).arg(1) == b.shape[2]
        assert s.model()[get_item_res].arg(3).arg(1) == b.shape[3]

        # change the annotation on the input to make sure it propagates
        # to the output
        for n in symbolic_traced.graph.nodes:
            if n.op == 'placeholder':
                n.type = TensorType([Dyn, 4])

        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        # dyn check
        assert s.model()[get_item_res].arg(2).arg(0) == 0


    def test_getitem_tensor2(self):
        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([4, 4])):
                getitem = x[(None, None)]
                return getitem

        B = BasicBlock()
        b = B.forward(torch.rand(4, 4))

        symbolic_traced: torch.fx.GraphModule = symbolic_trace(B)
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        get_item_res = z3.Const(2, tensor_type)
        assert s.model()[get_item_res].arg(0).arg(1) == b.shape[0]
        assert s.model()[get_item_res].arg(1).arg(1) == b.shape[1]
        assert s.model()[get_item_res].arg(2).arg(1) == b.shape[2]
        assert s.model()[get_item_res].arg(3).arg(1) == b.shape[3]


    def test_getitem_tensor_3(self):
        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([4, 4])):
                getitem = x[(None, slice(None, None, None), None, slice(None, None, None))]
                return getitem

        B = BasicBlock()
        b = B.forward(torch.rand(4, 4))
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(B)
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        get_item_res = z3.Const(2, tensor_type)
        assert s.model()[get_item_res].arg(0).arg(1) == b.shape[0]
        assert s.model()[get_item_res].arg(1).arg(1) == b.shape[1]
        assert s.model()[get_item_res].arg(2).arg(1) == b.shape[2]
        assert s.model()[get_item_res].arg(3).arg(1) == b.shape[3]



    def test_layer_norm(self):

        class BasicBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.l = torch.nn.LayerNorm((1024,))

            def forward(self, x: Dyn):
                return self.l(x)

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        transformed = transform_all_constraints(traced, counter=0)

        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

        # make the output a size 1 tensor which should result
        # in the migration of the input

        b = BasicBlock().forward(torch.rand(1024))
        input = z3.Const(1, tensor_type)
        output = z3.Const(2, tensor_type)
        s.add(output == tensor_type.tensor1(D(1, 1024)))
        s.check()
        self.assertEqual(s.model()[input], s.model()[output])
        # input shape = output shape
        self.assertEqual(b.shape[0], s.model()[input].arg(0).arg(1))

        # change annotation to the wrong shape
        for n in graph.nodes:
            if n.op == 'placeholder':
                n.type = TensorType([10, 10])

        traced = GraphModule(ast_rewriter.root, graph, "gm")
        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.unsat)

        # fix the annotation
        for n in graph.nodes:
            if n.op == 'placeholder':
                n.type = TensorType([10, 1024])

        traced = GraphModule(ast_rewriter.root, graph, "gm")
        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        s.check()
        b = BasicBlock().forward(torch.rand(10, 1024)).shape
        self.assertEqual(s.model()[output].arg(0).arg(1), b[0])
        self.assertEqual(s.model()[output].arg(1).arg(1), b[1])


    def test_layer_norm_functional(self):

        class BasicBlock(torch.nn.Module):
            def forward(self, x: Dyn):
                return torch.nn.functional.layer_norm(x, (1024,))

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        transformed = transform_all_constraints(traced, counter=0)

        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

        # make the output a size 1 tensor which should result
        # in the migration of the input

        b = BasicBlock().forward(torch.rand(1024))
        input = z3.Const(1, tensor_type)
        output = z3.Const(2, tensor_type)
        s.add(output == tensor_type.tensor1(D(1, 1024)))
        s.check()
        self.assertEqual(s.model()[input], s.model()[output])
        # input shape = output shape
        self.assertEqual(b.shape[0], s.model()[input].arg(0).arg(1))

    def test_ne_int_long_type_as(self):

        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([Dyn, Dyn]), y: TensorType([Dyn, Dyn])):
                ne_int = torch.ne(x, y).int()
                type_as = ne_int.type_as(y)
                long = type_as.long()
                return long

        symbolic_traced: torch.fx.GraphModule = symbolic_trace(BasicBlock())
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

        # migrate one of the parameters to a fully static shape so we can compare

        input = z3.Const(1, tensor_type)
        input_2 = z3.Const(2, tensor_type)
        s1, s2 = z3.Ints('s1 s2')

        output_long = z3.Const(8, tensor_type)
        s.add(input == tensor_type.tensor2(D(1, 2), D(1, 4)))
        s.add(input_2 == tensor_type.tensor2(D(1, s1), D(1, s2)))

        self.assertEqual(s.check(), z3.sat)
        actual_shape = BasicBlock().forward(torch.rand(2, 4), torch.rand(2, 4)).shape
        self.assertEqual(s.model()[output_long].arg(0).arg(1), actual_shape[0])
        self.assertEqual(s.model()[output_long].arg(1).arg(1), actual_shape[1])


    def test_ne(self):
        s1, s2 = z3.Ints('s1 s2')
        s11, s22 = z3.Ints('s11 s22')
        d1, d2 = D(s11, s1), D(0, s2)

        class BasicBlock(torch.nn.Module):
            def forward(self, x: Dyn, y: Dyn):
                return torch.ne(x, y)

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

        # change the annotations
        for n in graph.nodes:
            if n.name == 'x':
                n.type = TensorType([1, 2])
            if n.name == 'y':
                n.type = TensorType([2, Dyn])

        # resulting type should be TensorType([2, 2])
        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

        # force the second dimension to be Dyn
        # output should still be TensorType([2, 2])
        input = z3.Const(2, tensor_type)
        s.add(input == tensor_type.tensor2(d1, d2))
        self.assertEqual(s.check(), z3.sat)
        B = BasicBlock().forward(torch.rand(1, 2), torch.rand(2, 1))
        output = z3.Const(3, tensor_type)
        self.assertEqual(s.model()[output].arg(0).arg(1), B.shape[0])
        self.assertEqual(s.model()[output].arg(1).arg(1), B.shape[0])


    def test_cumsum(self):
        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([Dyn, 4, 3])):
                t = torch.cumsum(x, 3)
                return t

        symbolic_traced: torch.fx.GraphModule = meta_symbolic_trace(BasicBlock(), meta_args={})
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)

        # should be unsat since the index is not valid for this annotation
        self.assertEqual(s.check(), z3.unsat)

        # modify the annotation to Dyn which should give sat
        for n in symbolic_traced.graph.nodes:
            if n.op == 'placeholder':
                n.type = Dyn

        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

        # # modify the annotation to the right tensor size
        for n in symbolic_traced.graph.nodes:
            if n.op == 'placeholder':
                n.type = TensorType([1, 2, 3, 4])

        # verify that the input is equal to the output
        B = BasicBlock().forward(torch.rand(1, 2, 3, 4))
        res_shape = B.shape
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

        # confirm the output matches the expected tensor
        result = z3.Const(2, tensor_type)
        self.assertEqual(s.model()[result].arg(0).arg(1), res_shape[0])
        self.assertEqual(s.model()[result].arg(1).arg(1), res_shape[1])
        self.assertEqual(s.model()[result].arg(2).arg(1), res_shape[2])
        self.assertEqual(s.model()[result].arg(3).arg(1), res_shape[3])

        # confirm the output is not dyn
        self.assertNotEqual(s.model()[result].arg(0).arg(0).as_long(), 0)
        self.assertNotEqual(s.model()[result].arg(1).arg(0).as_long(), 0)
        self.assertNotEqual(s.model()[result].arg(2).arg(0).as_long(), 0)
        self.assertNotEqual(s.model()[result].arg(3).arg(0).as_long(), 0)


    def test_cumsum_kwargs(self):
        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([Dyn, 4, 3])):
                t = torch.cumsum(x, dim=3)
                return t

        symbolic_traced: torch.fx.GraphModule = meta_symbolic_trace(BasicBlock(), meta_args={})
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)

        # should be unsat since the index is not valid for this annotation
        self.assertEqual(s.check(), z3.unsat)

        # modify the annotation to Dyn which should give sat
        for n in symbolic_traced.graph.nodes:
            if n.op == 'placeholder':
                n.type = Dyn

        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)


    def test_arange(self):
        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([2, 4])):
                size = x.size()
                getitem = size[-1]
                arange = torch.arange(getitem)
                return arange

        B = BasicBlock().forward(torch.rand(2, 4))

        symbolic_traced: torch.fx.GraphModule = meta_symbolic_trace(BasicBlock(), meta_args={})
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        arange_result = z3.Const(5, tensor_type)
        self.assertNotEqual(s.model()[arange_result].arg(0).arg(0).as_long(), 0)
        self.assertEqual(s.model()[arange_result].arg(0).arg(1).as_long(), B.size()[0])

        # change the annotation to Dyn. This will migrate to an arbitrary type
        for n in symbolic_traced.graph.nodes:
            if n.op == 'placeholder':
                n.type = Dyn

        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

        for n in symbolic_traced.graph.nodes:
            if n.op == 'placeholder':
                n.type = TensorType([Dyn, Dyn, Dyn, Dyn])

        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

    def test_scalar_add(self):
        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([2, 4])):
                size = x.size()
                getitem = size[-1]
                arange = torch.arange(getitem)
                add = arange + 1
                return add

        symbolic_traced: torch.fx.GraphModule = meta_symbolic_trace(BasicBlock(), meta_args={})
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

        arange_result = z3.Const(5, tensor_type)
        add_result = z3.Const(6, tensor_type)
        self.assertEqual(s.model()[arange_result], s.model()[add_result])


    def test_regular_add_2(self):
        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([2, 4])):
                to = x.to()
                size = to.size()
                getitem = size[-1]
                add = getitem + 1
                return add

        b = BasicBlock().forward(torch.rand(2, 4))

        symbolic_traced: torch.fx.GraphModule = meta_symbolic_trace(BasicBlock(), meta_args={})
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        res = z3.Int(5)
        self.assertEqual(s.model()[res], b)


    def test_regular_add_3(self):
        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([2, 4])):
                to = x.to()
                size = to.size()
                getitem = size[-1]
                add = 1 + getitem
                return add

        b = BasicBlock().forward(torch.rand(2, 4))

        symbolic_traced: torch.fx.GraphModule = meta_symbolic_trace(BasicBlock(), meta_args={})
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        res = z3.Int(5)
        self.assertEqual(s.model()[res], b)

    def test_embedding(self):
        class BasicBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(256008, 1024, padding_idx=1)

            def forward(self, x: TensorType([2, 4])):
                return self.embedding(x)

        B = BasicBlock().forward(torch.ones([2, 4], dtype=torch.long)).size()
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        embedding_result = z3.Const(2, tensor_type)

        assert s.model()[embedding_result].arg(0).arg(1) == B[0]
        assert s.model()[embedding_result].arg(1).arg(1) == B[1]
        assert s.model()[embedding_result].arg(2).arg(1) == B[2]

        # change the type. This should still be satisfiable
        for n in traced.graph.nodes:
            if n.op == 'placeholder':
                n.type = TensorType([Dyn, Dyn])

        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        assert s.model()[embedding_result].arg(0).arg(0) == 0
        assert s.model()[embedding_result].arg(1).arg(0) == 0
        assert s.model()[embedding_result].arg(2).arg(1) == B[2]

        # change the type to Dyn. Here, we will get an arbitrary migration
        for n in traced.graph.nodes:
            if n.op == 'placeholder':
                n.type = Dyn

        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)

        self.assertEqual(s.check(), z3.sat)


    def test_embedding_2(self):
        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([2, 4]), y: TensorType([Dyn, 1024])):
                return torch.nn.functional.embedding(x, y)

        B = BasicBlock().forward(torch.ones([2, 4], dtype=torch.long), torch.rand(256008, 1024)).size()
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        embedding_result = z3.Const(5, tensor_type)

        assert s.model()[embedding_result].arg(0).arg(1) == B[0]
        assert s.model()[embedding_result].arg(1).arg(1) == B[1]
        assert s.model()[embedding_result].arg(2).arg(1) == B[2]

    def test_size_two_args(self):
        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([Dyn, 2, Dyn])):
                size = x.size(-1)
                return size

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

        d1, d2 = z3.Int(39), z3.Int(2)
        d4, d5 = z3.Int('input_d1'), z3.Int('input_d2')

        # migrate the third dimension
        s.add(d1 != 0)

        self.assertEqual(s.check(), z3.sat)
        input = z3.Const(1, tensor_type)
        s.add(input == tensor_type.tensor3(D(3, 39), D(1, 2), D(d4, d5)))

        # check if the item we got is the right one
        self.assertEqual(s.check(), z3.sat)
        self.assertEqual(s.model()[d5], s.model()[d2])
        self.assertEqual(s.model()[d1], s.model()[d4])

    def test_size_getitem(self):
        class BasicBlock(torch.nn.Module):
            def forward(self, x: Dyn):
                size = x.size()
                getitem = size[-1]
                return getitem


        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)

        self.assertEqual(s.check(), z3.sat)

        # force the input to be of size 4

        s1, s2, s3, s4 = z3.Ints('x1 x2 x3 x4')
        s11, s22, s33, s44 = z3.Ints('x11 x22 x33 x44')
        d1, d2, d3, d4 = D(s11, s1), D(s22, s2), D(s33, s3), D(s44, s4),

        input = z3.Const(1, tensor_type)
        s.add(input == tensor_type.tensor4(d1, d2, d3, d4))

        # check if the model is still SAT
        self.assertEqual(s.check(), z3.sat)

        s1, s2 = z3.Int(23), z3.Int(3)

        # check that the item is correct
        self.assertEqual(s.model()[s1], s.model()[s2])

        # invalid index but should still be SAT because input will be Dyn
        class BasicBlock(torch.nn.Module):
            def forward(self, x: Dyn):
                size = x.size()
                getitem = size[-10]
                return getitem

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)

        self.assertEqual(s.check(), z3.sat)
        s.add(input != z3_dyn)
        self.assertEqual(s.check(), z3.unsat)

    def test_view_mul(self):
        class BasicBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = torch.nn.Embedding(256008, 1024, padding_idx=1)

            def forward(self, x: TensorType([2, 4])):
                size = x.size()
                getitem = size[-1]
                view = x.view(-1, getitem)
                embed_tokens = self.embed_tokens(view)
                mul = embed_tokens * 32.0
                return mul


        # print(B)

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        # print(traced)

        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        # print(s.model())

        embedding_result = z3.Const(6, tensor_type)

        # note that the view output will be: tensor3(dim(0, 0), dim(1, 4), dim(1, 1024))
        # this is due to the reshape constraints. This can be lifted
        # but would require revising the type rules accordingly so we leave it for now
        assert (s.model()[embedding_result].arg(1).arg(1)) == 4
        assert (s.model()[embedding_result].arg(2).arg(1)) == 1024

        mul_result = z3.Const(13, tensor_type)
        assert s.model()[mul_result] == s.model()[embedding_result]

    def test_gt(self):
        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([Dyn, 4])):
                size = x.size()
                getitem_1 = size[-1]
                gt = getitem_1 > 1
                return gt

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        res = z3.Bool(4)
        self.assertEqual(s.model()[res], True)

    def test_view(self):
        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([2, 4])):
                view = x.view(-1, 8)
                return view

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

    def test_lt_tensor(self):
        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([2, 4]), y: Dyn):
                lt = x > y
                return lt

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)


    def test_conditional_wrong_assumption(self):
        """
        Test condition after making the wrong assumption about the input
        """
        class BasicBlock(torch.nn.Module):
            def forward(self, x: Dyn):
                gt = x > 1
                return gt

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())

        # The node we are considering is the gt node
        for n in graph.nodes:
            if n.target == operator.gt:
                node = n

        positive, negative = evaluate_conditional_with_constraints(ast_rewriter.root, graph, node)

        self.assertEqual(positive, z3.sat)
        self.assertEqual(negative, z3.sat)

    def test_conditional(self):
        """
        This test case is for the HFmodels interface.
        A function takes a node and a graph and considers
        the conditional the node represents and its negation
        and solves each formula with the remaining sets of constraints
        Returns:

        """
        class BasicBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = torch.nn.Embedding(256008, 1024, padding_idx=1)

            def forward(self, x: TensorType([Dyn, 4])):
                size = x.size()
                getitem = size[-1]
                view = x.view(-1, getitem)
                embed_tokens = self.embed_tokens(view)
                mul = embed_tokens * 32.0
                getitem_1 = size[-1]
                gt = getitem_1 > 1
                return gt

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())

        # The node we are considering is the gt node
        for n in graph.nodes:
            if n.target == operator.gt:
                node = n

        positive, negative = evaluate_conditional_with_constraints(ast_rewriter.root, graph, node)
        self.assertEqual(positive, z3.sat)
        self.assertEqual(negative, z3.unsat)

        # change the annotation to Dyn
        for n in graph.nodes:
            if n.op == 'placeholder':
                n.type = Dyn

        # here, both should be SAT since the input is Dyn
        positive, negative = evaluate_conditional_with_constraints(ast_rewriter.root, graph, node)

        self.assertEqual(positive, z3.sat)
        self.assertEqual(negative, z3.sat)


        # change the annotation to TensorType[Dyn, Dyn]
        for n in graph.nodes:
            if n.op == 'placeholder':
                n.type = TensorType([Dyn, Dyn])

        # here, both should be SAT as well
        positive, negative = evaluate_conditional_with_constraints(ast_rewriter.root, graph, node)

        self.assertEqual(positive, z3.sat)
        self.assertEqual(negative, z3.sat)


    def test_conditional_2(self):
        """
        This test case is for the HFmodels interface.
        A function takes a node and a graph and considers
        the conditional the node represents and its negation
        and solves each formula with the remaining sets of constraints
        Returns the opposite result of the above testcase

        """
        class BasicBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = torch.nn.Embedding(256008, 1024, padding_idx=1)

            def forward(self, x: TensorType([Dyn, 4])):
                size = x.size()
                getitem = size[-1]
                view = x.view(-1, getitem)
                embed_tokens = self.embed_tokens(view)
                mul = embed_tokens * 32.0
                getitem_1 = size[-1]
                lt = getitem_1 < 1
                return lt

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())

        # The node we are considering is the gt node
        for n in graph.nodes:
            if n.target == operator.lt:
                node = n

        positive, negative = evaluate_conditional_with_constraints(ast_rewriter.root, graph, node)
        self.assertEqual(positive, z3.unsat)
        self.assertEqual(negative, z3.sat)


class ComposeOperationsGradualTypes(unittest.TestCase):

    def test_masked_fill(self):
        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([2, 4])):
                size = x.size()
                getitem = size[-1]
                arange = torch.arange(getitem)
                view = x.view(-1, getitem)
                lt = arange > view
                masked_fill = x.masked_fill_(lt, 0)
                return masked_fill

        B = BasicBlock().forward(torch.rand(2, 4))
        # print(B.shape)

        symbolic_traced: torch.fx.GraphModule = meta_symbolic_trace(BasicBlock(), meta_args={})
        # print(symbolic_traced)
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        masked_fill_res = z3.Const(10, tensor_type)
        self.assertEqual(s.model()[masked_fill_res].arg(0).arg(1).as_long(), B.size()[0])
        self.assertEqual(s.model()[masked_fill_res].arg(1).arg(1).as_long(), B.size()[1])

        # change the annotation to Dyn. This will migrate to an arbitrary type
        for n in symbolic_traced.graph.nodes:
            if n.op == 'placeholder':
                n.type = Dyn

        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

        for n in symbolic_traced.graph.nodes:
            if n.op == 'placeholder':
                n.type = TensorType([Dyn, Dyn, Dyn, Dyn])

        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

    def test_add_reshape_1(self):
        class BasicBlock(torch.nn.Module):
            def forward(self, x: Dyn, y: Dyn):
                return torch.add(torch.reshape(x, (1, 2)), torch.reshape(y, (2, 2)))

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)

        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

    def test_add_reshape_2(self):
        class BasicBlock(torch.nn.Module):
            def forward(self, x: Dyn, y: Dyn):
                return torch.add(torch.reshape(x, (-1, 2)), torch.reshape(y, (2, 2, 2)))

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

    def test_conv_reshape_add_0(self):
        class BasicBlock(torch.nn.Module):
            def __init__(self, in_planes, out_planes, kernel_size, stride, padding, groups, dilation):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(in_channels=in_planes, out_channels=out_planes,
                                             kernel_size=kernel_size, stride=stride,
                                             padding=padding, groups=groups, bias=False, dilation=dilation)

            def forward(self, x: Dyn, y: Dyn):
                return torch.add(self.conv1(torch.reshape(x, (1, 2, 10, 20))), y)

        B = BasicBlock(2, 2, 2, 3, 2, 2, 2)
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        new_transformed_c = transform_all_constraints(traced)
        solver = z3.Solver()
        solver.add(new_transformed_c)
        self.assertEqual(solver.check(), z3.sat)


    def test_conv_reshape_add_0_2(self):
        class BasicBlock(torch.nn.Module):
            def __init__(self, in_planes, out_planes, kernel_size, stride, padding, groups, dilation):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(in_channels=in_planes, out_channels=out_planes,
                                             kernel_size=kernel_size, stride=stride,
                                             padding=padding, groups=groups, bias=False, dilation=dilation)

            def forward(self, x: Dyn, y: TensorType([4, 1])):
                return torch.add(self.conv1(torch.reshape(x, (1, 2, 10, 20))), y)

        B = BasicBlock(2, 2, 2, 3, 2, 2, 2)

        #        4,1
        # 1, 2, 4, 8
        res = B.forward(torch.rand(20, 20), torch.rand(1, 2, 4, 8)).size()
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        new_transformed_c = transform_all_constraints(traced)
        solver = z3.Solver()
        solver.add(new_transformed_c)
        self.assertEqual(solver.check(), z3.sat)


        conv_result = z3.Const(4, tensor_type)
        add_result = z3.Const(9, tensor_type)
        input_2 = z3.Const(2, tensor_type)

        s1, s2, s3, s4 = z3.Ints('x1 x2 x3 x4')
        s11, s22, s33, s44 = z3.Ints('x11 x22 x33 x44')
        d1, d2, d3, d4 = D(s11, s1), D(s22, s2), D(s33, s3), D(s44, s4),


        solver.add(conv_result == tensor_type.tensor4(d1, d2, d3, d4))
        solver.check()
        assert solver.model()[s1].as_long() == res[0]
        assert solver.model()[s2].as_long() == res[1]
        assert solver.model()[s3].as_long() == res[2]
        assert solver.model()[s4].as_long() == res[3]

        solver.add(input_2 == tensor_type.tensor2(D(1, 4), D(1, 1)))
        self.assertEqual(solver.check(), z3.sat)
        solver.add(add_result == tensor_type.tensor4(d1, d2, d3, d4))
        self.assertEqual(solver.check(), z3.sat)

        # first dimension could be anything because we have broadcasting
        assert solver.model()[s1] == res[0]
        assert solver.model()[s2] == res[1]
        assert solver.model()[s3] == res[2]
        assert solver.model()[s4] == res[3]

    def test_conv_reshape_add_0_3(self):
        class BasicBlock(torch.nn.Module):
            def __init__(self, in_planes, out_planes, kernel_size, stride, padding, groups, dilation):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(in_channels=in_planes, out_channels=out_planes,
                                             kernel_size=kernel_size, stride=stride,
                                             padding=padding, groups=groups, bias=False, dilation=dilation)

            def forward(self, x: Dyn, y: TensorType([11, 1])):
                return torch.add(self.conv1(torch.reshape(x, (1, 2, 10, 20))), y)

        B = BasicBlock(2, 2, 2, 3, 2, 2, 2)
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        new_transformed_c = transform_all_constraints(traced)
        solver = z3.Solver()
        solver.add(new_transformed_c)
        self.assertEqual(solver.check(), z3.unsat)


    def test_conv_reshape_add_1(self):
        class BasicBlock(torch.nn.Module):
            def __init__(self, in_planes, out_planes, kernel_size, stride, padding, groups, dilation):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(in_channels=in_planes, out_channels=out_planes,
                                             kernel_size=kernel_size, stride=stride,
                                             padding=padding, groups=groups, bias=False, dilation=dilation)

            def forward(self, x: Dyn, y: TensorType([1, 2, 10, 20])):
                return torch.add(self.conv1(torch.reshape(x, (1, 2, 10, 20))), y)

        B = BasicBlock(2, 2, 2, 3, 2, 2, 2)
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        new_transformed_c = transform_all_constraints(traced)
        solver = z3.Solver()
        solver.add(new_transformed_c)
        self.assertEqual(solver.check(), z3.unsat)


class GradualTypes(unittest.TestCase):
    def test_conv_reshape_unsat(self):

        class BasicBlock(torch.nn.Module):
            def __init__(self, in_planes, out_planes, kernel_size, stride, padding, groups, dilation):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(in_channels=in_planes, out_channels=out_planes,
                                             kernel_size=kernel_size, stride=stride,
                                             padding=padding, groups=groups, bias=False, dilation=dilation)

            def forward(self, x: Dyn):
                return self.conv1(torch.reshape(x, (1, 2, 10)))

        B = BasicBlock(2, 2, 2, 3, 2, 2, 2)
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        new_transformed_c = transform_all_constraints(traced)
        solver = z3.Solver()
        solver.add(new_transformed_c)
        self.assertEqual(solver.check(), z3.unsat)

    def test_conv_reshape0(self):
        class BasicBlock(torch.nn.Module):
            def __init__(self, in_planes, out_planes, kernel_size, stride, padding, groups, dilation):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(in_channels=in_planes, out_channels=out_planes,
                                             kernel_size=kernel_size, stride=stride,
                                             padding=padding, groups=groups, bias=False, dilation=dilation)

            def forward(self, x: Dyn):
                return self.conv1(torch.reshape(x, (1, 2, 10, 20)))

        B = BasicBlock(2, 2, 2, 3, 2, 2, 2)
        res = B.forward(torch.rand(20, 20)).size()
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        new_transformed_c = transform_all_constraints(traced)

        solver = z3.Solver()
        solver.add(new_transformed_c)
        self.assertEqual(solver.check(), z3.sat)
        conv_result = z3.Const(3, tensor_type)

        s1, s2, s3, s4 = z3.Ints('x1 x2 x3 x4')
        s11, s22, s33, s44 = z3.Ints('x11 x22 x33 x44')
        d1, d2, d3, d4 = D(s11, s1), D(s22, s2), D(s33, s3), D(s44, s4),

        solver.add(conv_result == tensor_type.tensor4(d1, d2, d3, d4))
        solver.check()
        # print(solver.model())
        # print(type(solver.model()[s1]))
        assert solver.model()[s1].as_long() == res[0]
        assert solver.model()[s2].as_long() == res[1]
        assert solver.model()[s3].as_long() == res[2]
        assert solver.model()[s4].as_long() == res[3]

        s1, s2, s3, s4 = z3.Ints('y1 y2 y3 y4')
        s11, s22, s33, s44 = z3.Ints('y11 y22 y33 y44')
        d1, d2, d3, d4 = D(s11, s1), D(s22, s2), D(s33, s3), D(s44, s4),

        input = z3.Const(1, tensor_type)
        solver.add(input == tensor_type.tensor4(d1, d2, d3, d4))

        # assert solver.check() == sat
        # solver.add(s11 == 1)
        # solver.add(s22 == 1)
        # solver.add(s33 == 1)
        # solver.add(s44 == 1)
        #
        # print(solver.check())
        # print(solver.model())


    def test_conv_reshape1(self):
        class BasicBlock(torch.nn.Module):
            def __init__(self, in_planes, out_planes, kernel_size, stride, padding, groups, dilation):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(in_channels=in_planes, out_channels=out_planes,
                                             kernel_size=kernel_size, stride=stride,
                                             padding=padding, groups=groups, bias=False, dilation=dilation)

            def forward(self, x: TensorType([20, 20])):
                return self.conv1(torch.reshape(x, (1, -1, 10, 20)))

        B = BasicBlock(2, 2, 2, 3, 2, 2, 2)
        res = B.forward(torch.rand(20, 20)).size()
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        new_transformed_c = transform_all_constraints(traced)

        solver = z3.Solver()
        solver.add(new_transformed_c)
        self.assertEqual(solver.check(), z3.sat)
        conv_result = z3.Const(3, tensor_type)

        s1, s2, s3, s4 = z3.Ints('x1 x2 x3 x4')
        s11, s22, s33, s44 = z3.Ints('x11 x22 x33 x44')
        d1, d2, d3, d4 = D(s11, s1), D(s22, s2), D(s33, s3), D(s44, s4),

        solver.add(conv_result == tensor_type.tensor4(d1, d2, d3, d4))
        solver.check()
        # print(solver.model())
        assert solver.model()[s1].as_long() == res[0]
        assert solver.model()[s2].as_long() == res[1]
        assert solver.model()[s3].as_long() == res[2]
        assert solver.model()[s4].as_long() == res[3]


class TestSingleOperation(unittest.TestCase):

    def test_conv_wrong_example(self):
        class BasicBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(in_channels=2, out_channels=2,
                                             kernel_size=2, stride=2,
                                             padding=2, groups=2, bias=False, dilation=2)

                self.conv2 = torch.nn.Conv2d(in_channels=4, out_channels=2,
                                             kernel_size=2, stride=2,
                                             padding=2, groups=2, bias=False, dilation=2)

                self.relu = torch.nn.ReLU(inplace=True)

            def forward(self, x: Dyn):
                y = self.relu(self.conv1(x))
                z = self.relu(self.conv2(x))
                return z

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced)

        solver3 = z3.Solver()
        solver3.add(transformed)
        print(solver3.check())
        assert solver3.check() == z3.sat

        s1, s2, s3, s4 = z3.Ints('s1 s2 s3 s4')
        s11, s22, s33, s44 = z3.Ints('s11 s22 s33 s44')
        d1, d2, d3, d4 = D(s11, s1), D(s22, s2), D(s33, s3), D(s44, s4),
        x = z3.Const(1, tensor_type)
        solver3.add(x == tensor_type.tensor4(d1, d2, d3, d4))
        assert solver3.check() == z3.sat

        solver3.add(s22 != 0)
        assert solver3.check() == z3.unsat

    def test_conv_dyn(self):

        s1, s2, s3, s4 = z3.Ints('s1 s2 s3 s4')
        e1, e2, e3, e4 = z3.Ints('e1 e2 e3 e4')
        s11, s22, s33, s44 = z3.Ints('s11 s22 s33 s44')
        e11, e22, e33, e44 = z3.Ints('e11 e22 e33 e44')
        d1, d2, d3, d4 = D(s11, s1), D(s22, s2), D(s33, s3), D(s44, s4),
        b1, b2, b3, b4 = D(e11, e1), D(e22, e2), D(e33, e3), D(e44, e4)

        class BasicBlock(torch.nn.Module):
            def __init__(self, in_planes, out_planes, kernel_size, stride, padding, groups, dilation):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(in_channels=in_planes, out_channels=out_planes,
                                             kernel_size=kernel_size, stride=stride,
                                             padding=padding, groups=groups, bias=False, dilation=dilation)

            def forward(self, x: Dyn):
                return self.conv1(x)

        BasicBlock(2, 2, 2, 2, 2, 2, 2).forward(torch.rand(4, 2, 3, 4))

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock(2, 2, 2, 2, 2, 2, 2))
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced)

        solver3 = z3.Solver()
        solver3.add(transformed)
        assert solver3.check() == z3.sat

        x = z3.Const(1, tensor_type)
        y = z3.Const(2, tensor_type)

        solver3.add(x == tensor_type.tensor4(d1, d2, d3, d4),
                    y == tensor_type.tensor4(b1, b2, b3, b4))

        assert solver3.check() == z3.sat
        assert solver3.model()[s1].as_long() == solver3.model()[e1].as_long()
        assert solver3.model()[s11].as_long() == solver3.model()[e11].as_long()

        solver3.add(s2 != 2)
        assert solver3.check() == z3.sat
        assert solver3.model()[s22].as_long() == 0

        solver3.add(s22 != 0)
        self.assertEqual(solver3.check(), z3.unsat)

        solver2 = z3.Solver()
        solver2.add(transformed)
        assert solver2.check() == z3.sat
        solver2.add(x == tensor_type.tensor3(d1, d2, d3))
        self.assertEqual(solver2.check(), z3.unsat)


    def test_add(self):
        s1, s2, s3, s4 = z3.Ints('s1 s2 s3 s4')
        s11, s22, s33, s44 = z3.Ints('s11 s22 s33 s44')
        d1, d2, d3, d4 = D(s11, s1), D(s22, s2), D(s33, s3), D(s44, s4),

        class BasicBlock(torch.nn.Module):
            def forward(self, x: Dyn, y: Dyn):
                return torch.add(x, y)

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)

        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

        # make the tensor be of size 1
        x = z3.Const(1, tensor_type)
        s.add(x == tensor_type.tensor1(D(1, s11)))
        self.assertEqual(s.check(), z3.sat)

        y = z3.Const(2, tensor_type)
        s.add(y == tensor_type.tensor1(D(1, s22)))
        self.assertEqual(s.check(), z3.sat)

        s.add(s11 == 1)  # tensor[1]
        s.add(s22 == 2)  # tensor[2]
        self.assertEqual(s.check(), z3.sat)

        class BasicBlock2(torch.nn.Module):
            def forward(self, x: TensorType((Dyn,)), y: Dyn):
                return torch.add(x, y)

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock2())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        # make the tensor be of size 1
        x = z3.Const(1, tensor_type)
        s.add(x == tensor_type.tensor1(D(1, s11)))
        self.assertEqual(s.check(), z3.sat)
        y = z3.Const(2, tensor_type)
        s.add(y == tensor_type.tensor1(D(1, s22)))
        self.assertEqual(s.check(), z3.sat)
        s.add(s11 == 4)  # tensor[4]
        s.add(s22 == 5)  # tensor[5]
        self.assertEqual(s.check(), z3.unsat)

        class BasicBlock3(torch.nn.Module):
            def forward(self, x: TensorType((Dyn,)), y: Dyn):
                return torch.add(x, y)

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock3())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced)
        s = z3.Solver()
        s.add(transformed)
        x = z3.Const(1, tensor_type)
        s.add(x == tensor_type.tensor2(d1, d2))
        self.assertEqual(s.check(), z3.unsat)

    def test_add_padding(self):
        s1, s2, s3, s4 = z3.Ints('s1 s2 s3 s4')

        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType((Dyn,)), y: TensorType((Dyn, Dyn))):
                return torch.add(x, y)

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)

        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

        x = z3.Const(1, tensor_type)
        s.add(x == tensor_type.tensor1(D(1, s1)))

        self.assertEqual(s.check(), z3.sat)

        # print(s.model())

    def test_add_padding_2(self):
        s1, s2, s3, s4 = z3.Ints('s1 s2 s3 s4')

        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([Dyn, Dyn]), y: TensorType([Dyn])):
                return torch.add(x, y)

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)

        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        # print(s.model())

        x = z3.Const(1, tensor_type)
        s.add(x == tensor_type.tensor2(D(1, s1), D(1, s2)))
        self.assertEqual(s.check(), z3.sat)

        y = z3.Const(2, tensor_type)
        s.add(y == tensor_type.tensor1(D(0, s3)))
        self.assertEqual(s.check(), z3.sat)

        add_result = z3.Const(3, tensor_type)
        broadcast_res1, broadcast_res2 = z3.Const(4, tensor_type), z3.Const(5, tensor_type)

        # print(s.model())

        assert s.model()[broadcast_res1].decl() == tensor_type.tensor2
        assert s.model()[broadcast_res2].decl() == tensor_type.tensor2
        assert s.model()[add_result].decl() == tensor_type.tensor2
        assert s.model()[y].decl() == tensor_type.tensor1

        # print(s.model())

        # prevent broadcasting for that dimension
        s.add(s2 > 1)

        assert s.check()

        # the second dimension of the result is a number, not Dyn.
        # however if the first input dimension had been 1, we would
        # have had dyn in the result, as seen in the next test case
        assert s.model()[add_result].arg(1).arg(0).as_long() != 0

    def test_add_padding_3(self):
        s1, s2, s3, s4 = z3.Ints('s1 s2 s3 s4')

        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([Dyn, 1]), y: TensorType([Dyn])):
                return torch.add(x, y)

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)

        s = z3.Solver()
        s.add(transformed)
        # print(transformed)
        self.assertEqual(s.check(), z3.sat)

        x = z3.Const(1, tensor_type)
        y = z3.Const(2, tensor_type)

        s.add(s2 != 0)
        s.add(x == tensor_type.tensor2(D(0, s1), D(s2, 1)))
        s.add(y == tensor_type.tensor1(D(0, s3)))

        self.assertEqual(s.check(), z3.sat)

        # print(s.model())

        add_result = z3.Const(3, tensor_type)
        assert s.model()[add_result].arg(0).arg(0).as_long() == 0
        assert s.model()[add_result].arg(1).arg(0).as_long() == 0


    def test_add_padding_4(self):
        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([2, 1]), y: TensorType([3])):
                return torch.add(x, y)

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)

        s = z3.Solver()
        s.add(transformed)

        self.assertEqual(s.check(), z3.sat)

        add_result = z3.Const(3, tensor_type)
        assert s.model()[add_result] == tensor_type.tensor2(D(1, 2), D(1, 3))

    def test_add_padding_5(self):
        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([2, 2]), y: TensorType([3])):
                return torch.add(x, y)

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)

        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.unsat)

    def test_add_size_3(self):

        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([Dyn, Dyn, Dyn]), y: TensorType([Dyn, Dyn, Dyn])):
                return torch.add(x, y)

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)

        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

        x = z3.Const(1, tensor_type)
        y = z3.Const(2, tensor_type)

        s1, s2, s3, s4, s5 = z3.Ints('s1 s2 s3 s4 s5')

        s.add(x == tensor_type.tensor3(D(1, s1), D(1, 1), D(1, s2)))
        s.add(y == tensor_type.tensor3(D(1, s3), D(1, s4), D(1, s5)))

        self.assertEqual(s.check(), z3.sat)
        s.add(s2 == 5)
        self.assertEqual(s.check(), z3.sat)
        s.add(s5 == 6)
        self.assertEqual(s.check(), z3.unsat)

    def test_add_padding_6(self):

        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([Dyn]), y: TensorType([Dyn, Dyn, Dyn])):
                return torch.add(x, y)

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

        x = z3.Const(1, tensor_type)
        y = z3.Const(2, tensor_type)

        s1, s2, s3, s4, s5 = z3.Ints('s1 s2 s3 s4 s5')

        s.add(x == tensor_type.tensor1(D(1, s1)))
        s.add(y == tensor_type.tensor3(D(1, s2), D(1, s3), D(1, s4)))

        self.assertEqual(s.check(), z3.sat)

        s.add(s1 == 4)
        s.add(s4 == 5)

        self.assertEqual(s.check(), z3.unsat)

    def test_add_padding_7(self):

        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([Dyn]), y: TensorType([Dyn, Dyn, Dyn, Dyn])):
                return torch.add(x, y)

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        x = z3.Const(1, tensor_type)
        s1, s2, s3, s4, s5 = z3.Ints('s1 s2 s3 s4 s5')
        s.add(x == tensor_type.tensor2(D(s1, s2), D(s2, s3)))
        self.assertEqual(s.check(), z3.unsat)


    def test_add_padding_8(self):

        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([Dyn]), y: TensorType([Dyn, Dyn, Dyn, Dyn])):
                return torch.add(x, y)

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        x = z3.Const(1, tensor_type)
        y = z3.Const(2, tensor_type)

        s1, s2, s3, s4, s5 = z3.Ints('s1 s2 s3 s4 s5')
        s.add(x == tensor_type.tensor1(D(s1, 1)))
        s.add(s1 >= 0)

        self.assertEqual(s.check(), z3.sat)

        s.add(y == tensor_type.tensor4(D(0, s2), D(0, s3), D(0, s4), D(0, s5)))
        self.assertEqual(s.check(), z3.sat)

    def test_add_padding_9(self):

        class BasicBlock(torch.nn.Module):
            def forward(self, x: Dyn, y: TensorType([Dyn, Dyn, Dyn, Dyn])):
                return torch.add(x, y)

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)

        self.assertEqual(s.check(), z3.sat)
        x = z3.Const(1, tensor_type)
        y = z3.Const(2, tensor_type)

        s1, s2, s3, s4, s5, s6, s7 = z3.Ints('s1 s2 s3 s4 s5 s6 s7')
        s.add(x == tensor_type.tensor1(D(s1, s7)))
        s.add(s1 == 1)
        self.assertEqual(s.check(), z3.sat)

        s.add(y == tensor_type.tensor4(D(0, s2), D(0, s3), D(0, s4), D(s6, s5)))
        self.assertEqual(s.check(), z3.sat)

        s.add(s6 == 1)

        self.assertEqual(s.check(), z3.sat)
        s.add(s5 != 1, s7 != 1)
        assert s.check()

        assert s.model()[s5].as_long() == s.model()[s7].as_long()

    def test_conv_static(self):
        s1, s2, s3, s4 = z3.Ints('s1 s2 s3 s4')
        e1, e2, e3, e4 = z3.Ints('e1 e2 e3 e4')
        s11, s22, s33, s44 = z3.Ints('s11 s22 s33 s44')
        e11, e22, e33, e44 = z3.Ints('e11 e22 e33 e44')
        d1, d2, d3, d4 = D(s11, s1), D(s22, s2), D(s33, s3), D(s44, s4),
        b1, b2, b3, b4 = D(e11, e1), D(e22, e2), D(e33, e3), D(e44, e4)

        class BasicBlock(torch.nn.Module):
            def __init__(self, in_planes, out_planes, kernel_size, stride, padding, groups, dilation):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(in_channels=in_planes, out_channels=out_planes,
                                             kernel_size=kernel_size, stride=stride,
                                             padding=padding, dilation=dilation)

            def forward(self, x: TensorType((1, 2, 10, 20))):
                return self.conv1(x)

        ast_rewriter = RewritingTracer()

        B = BasicBlock(2, 2, 2, 3, 2, 2, 2)
        res = B.forward(torch.rand(1, 2, 10, 20)).size()

        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        new_transformed_c = transform_all_constraints(traced)
        solver = z3.Solver()
        solver.add(new_transformed_c)
        self.assertEqual(solver.check(), z3.sat)

        x = z3.Const(1, tensor_type)
        y = z3.Const(2, tensor_type)

        solver.add(x == tensor_type.tensor4(d1, d2, d3, d4))
        solver.add(y == tensor_type.tensor4(b1, b2, b3, b4))
        self.assertEqual(solver.check(), z3.sat)
        # print(solver.model())
        assert solver.model()[e3].as_long() == res[2]
        assert solver.model()[e4].as_long() == res[3]

        B2 = BasicBlock(2, 4, 5, 2, 9, 2, 2)
        res2 = B2.forward(torch.rand(1, 2, 10, 20)).size()

        graph2 = ast_rewriter.trace(B2)
        traced2 = GraphModule(ast_rewriter.root, graph2, "gm")
        new_transformed_c = transform_all_constraints(traced2)
        solver = z3.Solver()
        solver.add(new_transformed_c)

        solver.add(x == tensor_type.tensor4(d1, d2, d3, d4))
        solver.add(y == tensor_type.tensor4(b1, b2, b3, b4))

        self.assertEqual(solver.check(), z3.sat)
        assert solver.model()[e3].as_long() == res2[2]
        assert solver.model()[e4].as_long() == res2[3]

    def test_reshape_dyn(self):
        s11, s22, s33, s44 = z3.Ints('s11 s22 s33 s44')

        class BasicBlock(torch.nn.Module):
            def forward(self, x: Dyn):
                return torch.reshape(x, (2, -1))

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        x = z3.Const(1, tensor_type)
        s.add(x == tensor_type.tensor1(D(1, s11)))
        self.assertEqual(s.check(), z3.sat)
        s.add(z3.Or([s11 == 2, s11 == 4, s11 == 9]))
        self.assertEqual(s.check(), z3.sat)
        s.add(s11 == 9)
        self.assertEqual(s.check(), z3.unsat)


    def test_reshape_annotated(self):
        s1, s2, s3, s4 = z3.Ints('s1 s2 s3 s4')
        s11, s22, s33, s44 = z3.Ints('s11 s22 s33 s44')
        d1, d2, d3, d4 = D(s11, s1), D(s22, s2), D(s33, s3), D(s44, s4),

        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([Dyn])):
                return torch.reshape(x, (2, -1))

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        transformed = transform_all_constraints(traced)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        x = z3.Const(1, tensor_type)
        s.add(x == tensor_type.tensor2(d1, d2))
        self.assertEqual(s.check(), z3.unsat)

    def test_reshape_static_target(self):
        s11, s22, s33, s44 = z3.Ints('s11 s22 s33 s44')

        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([Dyn])):
                return torch.reshape(x, (2, 3))

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        transformed = transform_all_constraints(traced)
        # print(transformed)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        x = z3.Const(1, tensor_type)
        s.add(x == tensor_type.tensor1(D(1, s11)))
        s.check()
        assert s.model()[s11].as_long() == 6
        s.add(s11 != 6)
        self.assertEqual(s.check(), z3.unsat)

    def test_reshape_static_target2(self):
        s11, s22, s33, s44 = z3.Ints('s11 s22 s33 s44')

        class BasicBlock(torch.nn.Module):
            def forward(self, x: Dyn):
                return torch.reshape(x, (2, 3, 1, 1))

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        transformed = transform_all_constraints(traced)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        x = z3.Const(1, tensor_type)
        s.add(x == tensor_type.tensor1(D(1, s11)))
        s.check()
        assert s.model()[s11].as_long() == 6
        s.add(s11 != 6)
        self.assertEqual(s.check(), z3.unsat)


    def test_conv2D_maxpool2d_flatten(self):
        class BasicBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()

                self.conv1 = torch.nn.Conv2d(3, 6, 5)
                self.pool = torch.nn.MaxPool2d(2, 2)
                self.conv2 = torch.nn.Conv2d(6, 16, 5)
                self.fc1 = torch.nn.Linear(5, 120)
                self.pool2 = torch.nn.AdaptiveAvgPool2d((6, 7))

            def forward(self, x : TensorType((4, 3, 32, 32))):
                out = self.conv1(x)
                out = self.pool(out)
                out = self.conv2(out)
                out = self.pool(out)
                out = self.fc1(out)
                out = self.pool2(out)
                out = torch.flatten(out, 1)
                return out

        B = BasicBlock()
        res = B.forward(torch.rand(4, 3, 32, 32)).shape
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        constraints = transform_all_constraints(traced, counter=0)
        solver = z3.Solver()
        solver.add(constraints)
        solver.check()
        input = z3.Const(1, tensor_type)
        solver.add(input == tensor_type.tensor4(D(1, 4), D(1, 3), D(1, 32), D(1, 32)))
        solver.check()
        output = z3.Const(48, tensor_type)
        assert solver.model()[output].arg(0).arg(1) == res[0]
        assert solver.model()[output].arg(1).arg(1) == res[1]

    def test_conv2D_maxpool2d_flatten_unsat(self):
        class BasicBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()

                self.conv1 = torch.nn.Conv2d(3, 6, 5)
                self.pool = torch.nn.MaxPool2d(2, 2)
                self.conv2 = torch.nn.Conv2d(6, 16, 5)
                self.fc1 = torch.nn.Linear(5, 120)
                self.pool2 = torch.nn.AdaptiveAvgPool2d((6, 7))

            def forward(self, x : TensorType((4, 3, 32, 32))):
                out = self.conv1(x)
                out = self.pool(out)
                out = self.conv2(out)
                out = self.pool(out)
                out = self.fc1(out)
                out = self.pool2(out)
                out = torch.flatten(out, 1)
                return out

        B = BasicBlock()
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        constraints = transform_all_constraints(traced, counter=0)
        solver = z3.Solver()
        solver.add(constraints)
        solver.check()
        input = z3.Const(1, tensor_type)
        solver.add(input == tensor_type.tensor4(D(1, 4), D(1, 3), D(1, 32), D(1, 45)))
        self.assertEqual(solver.check(), z3.unsat)

    def test_conv2D_maxpool2d_flatten_dyn(self):
        class BasicBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()

                self.conv1 = torch.nn.Conv2d(3, 6, 5)
                self.pool = torch.nn.MaxPool2d(2, 2)
                self.conv2 = torch.nn.Conv2d(6, 16, 5)
                self.fc1 = torch.nn.Linear(5, 120)
                self.pool2 = torch.nn.AdaptiveAvgPool2d((6, 7))

            def forward(self, x : TensorType((Dyn, 3, 32, 32))):
                out = self.conv1(x)
                out = self.pool(out)
                out = self.conv2(out)
                out = self.pool(out)
                out = self.fc1(out)
                out = self.pool2(out)
                out = torch.flatten(out, 1)
                return out

        B = BasicBlock()
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        constraints = transform_all_constraints(traced, counter=0)
        solver = z3.Solver()
        solver.add(constraints)
        self.assertEqual(solver.check(), z3.sat)

    def test_type_check_flatten(self):
        s1, s2, s3, s4 = z3.Ints('s1 s2 s3 s4')

        class M(torch.nn.Module):
            def forward(self, x: TensorType([2, 3, 4, 5])):
                return torch.flatten(x, start_dim=1, end_dim=3)

        module = M()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        constraints = transform_all_constraints(symbolic_traced, counter=0)
        solver = z3.Solver()
        solver.add(constraints)
        self.assertEqual(solver.check(), z3.sat)
        flatten = z3.Const(2, tensor_type)

        res = M().forward(torch.rand(2, 3, 4, 5)).size()
        assert solver.model()[flatten].arg(0).arg(1) == res[0]
        assert solver.model()[flatten].arg(1).arg(1) == res[1]

        class M(torch.nn.Module):
            def forward(self, x: TensorType([2, 3, Dyn, 5])):
                return torch.flatten(x, start_dim=1, end_dim=3)

        module = M()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        constraints = transform_all_constraints(symbolic_traced, counter=0)
        solver = z3.Solver()
        solver.add(constraints)
        self.assertEqual(solver.check(), z3.sat)
        x = z3.Const(1, tensor_type)
        y = z3.Const(2, tensor_type)

        solver.add(x == tensor_type.tensor4(D(1, 2), D(1, 3), D(0, s1), D(1, 5)))
        self.assertEqual(solver.check(), z3.sat)
        assert solver.model()[y].arg(1).arg(0) == 0


        class M(torch.nn.Module):
            def forward(self, x: TensorType([2, 3, Dyn])):
                return torch.flatten(x, 10, 0)

        module = M()
        # print(module.forward(torch.rand(2,3,5)).shape)
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        constraints = transform_all_constraints(symbolic_traced, counter=0)
        solver = z3.Solver()
        solver.add(constraints)
        self.assertEqual(solver.check(), z3.unsat)

class ConstraintGeneration(unittest.TestCase):

    def test_add_reshape(self):
        class BasicBlock(torch.nn.Module):
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
                super().__init__()
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


class TestInternalConstraints(unittest.TestCase):
    def test_precision(self):

        c1 = BinConstraintT(Dyn, TVar('x'), op_precision)
        transformed, _ = transform_constraint(c1, 0)
        assert transformed == T()

        c2 = BinConstraintT(TensorType([1, Dyn, 3]), TVar('x'), op_precision)
        transformed, counter = transform_constraint(c2, 0)
        assert len(transformed.conjucts) == 7

    def test_matching(self):
        c1 = BinConstraintT(TVar('x'),
                            TensorType([DVar('a'), DVar('b'), DVar('c'), DVar('d')]), op_matching)
        transformed, _ = transform_constraint(c1, 0)
        assert len(transformed.disjuncts) == 2

    def test_consistency(self):
        c1 = BinConstraintT(TVar('x'),
                            TensorType([DVar('a'), DVar('b')]), op_consistency)
        transformed, count = transform_constraint(c1, 0)

        assert len(transformed.disjuncts) == 5
        transformed, count = transform_constraint(transformed, count)
        assert len(transformed.disjuncts) == 5

    # def test_apply_broadcasting(self):
    #     c1 = ApplyBroadcasting(TVar(1), TVar(2), TVar(3), TVar(4))
    #     transformed, count = transform_apply_broadcasting(c1, 5)
    #     assert len(transformed.conjucts) == 41

@skipIfNoTorchVision
class TestResNet(unittest.TestCase):

    def test_resnet50_unsat(self):
        traced = symbolic_trace(models.resnet50())
        for n in traced.graph.nodes:
            n.type = Dyn

        constraints = transform_all_constraints(traced, counter=0)
        solver = z3.Solver()
        solver.add(constraints)
        input = z3.Const(1, tensor_type)
        # input with 3 dimensions
        solver.add(input == tensor_type.tensor3(D(1, 1), D(1, 3), D(1, 224)))
        self.assertEqual(solver.check(), z3.unsat)



    def test_resnet50(self):
        traced = symbolic_trace(models.resnet50())
        for n in traced.graph.nodes:
            n.type = Dyn

        sample_input = torch.randn(1, 3, 224, 224)
        res = models.resnet50().forward(sample_input).size()
        constraints = transform_all_constraints(traced, counter=0)
        solver = z3.Solver()
        solver.add(constraints)
        self.assertEqual(solver.check(), z3.sat)
        linear = z3.Const(650, tensor_type)

        input = z3.Const(1, tensor_type)
        solver.add(input == tensor_type.tensor4(D(1, 1), D(1, 3), D(1, 224), D(1, 224)))
        self.assertEqual(solver.check(), z3.sat)
        assert solver.model()[linear] == tensor_type.tensor2(D(1, res[0]), D(1, res[1]))

    def test_resnet502(self):
        traced = symbolic_trace(models.resnet50())
        for n in traced.graph.nodes:
            n.type = Dyn

        constraints = transform_all_constraints(traced, counter=0)
        solver = z3.Solver()
        solver.add(constraints)
        linear = z3.Const(650, tensor_type)
        input = z3.Const(1, tensor_type)
        batch = z3.Int('b')
        solver.add(input == tensor_type.tensor4(D(1, batch), D(1, 3), D(1, 224), D(1, 224)))
        solver.add(batch > 4)
        solver.check()
        assert solver.model()[batch] == solver.model()[linear].arg(0).arg(1)

    def test_resnet503(self):
        traced = symbolic_trace(models.resnet50())
        for n in traced.graph.nodes:
            n.type = Dyn

        constraints = transform_all_constraints(traced, counter=0)
        solver = z3.Solver()
        solver.add(constraints)
        linear = z3.Const(650, tensor_type)
        input = z3.Const(1, tensor_type)
        batch, d1, d2 = z3.Ints('b d1 d2')
        solver.add(input == tensor_type.tensor4(D(1, batch), D(1, 3), D(1, 224), D(1, 224)))
        solver.add(linear == tensor_type.tensor2(D(1, d1), D(1, d2)))
        self.assertEqual(solver.check(), z3.sat)
        solver.add(batch != d1)
        self.assertEqual(solver.check(), z3.unsat)

@skipIfNoTorchVision
class TestAlexNet(unittest.TestCase):
    def test_alexnet1(self):

        alexnet = models.alexnet()
        symbolic_traced : torch.fx.GraphModule = symbolic_trace(alexnet)

        for n in symbolic_traced.graph.nodes:
            n.type = Dyn

        # print(symbolic_traced)

        res = alexnet.forward(torch.rand(10, 3, 227, 227)).size()
        constraints = transform_all_constraints(symbolic_traced, counter=0)
        solver = z3.Solver()
        solver.add(constraints)
        self.assertEqual(solver.check(), z3.sat)
        input = z3.Const(1, tensor_type)
        conv = z3.Const(2, tensor_type)
        solver.add(input == tensor_type.tensor4(D(1, 10), D(1, 3), D(1, 227), D(1, 227)))
        self.assertEqual(solver.check(), z3.sat)
        assert solver.model()[conv] == tensor_type.tensor4(D(1, 10), D(1, 64), D(1, 56), D(1, 56))

        relu = z3.Const(7, tensor_type)
        assert solver.model()[relu] == tensor_type.tensor4(D(1, 10), D(1, 64), D(1, 56), D(1, 56))

        maxpool = z3.Const(8, tensor_type)
        assert solver.model()[maxpool] == tensor_type.tensor4(D(1, 10), D(1, 64), D(1, 27), D(1, 27))

        maxpool2 = z3.Const(42, tensor_type)
        assert solver.model()[maxpool2] == tensor_type.tensor4(D(1, 10), D(1, 256), D(1, 6), D(1, 6))

        flatten = z3.Const(52, tensor_type)
        assert solver.model()[flatten] == tensor_type.tensor2(D(1, 10), D(1, 9216))

        linear = z3.Const(64, tensor_type)
        assert solver.model()[linear] == tensor_type.tensor2(D(1, 10), D(1, 4096))

        linear2 = z3.Const(109, tensor_type)
        assert solver.model()[linear2] == tensor_type.tensor2(D(1, res[0]), D(1, res[1]))


    def test_alexnet2(self):
        alexnet = models.alexnet()
        symbolic_traced : torch.fx.GraphModule = symbolic_trace(alexnet)

        for n in symbolic_traced.graph.nodes:
            if n.op == 'placeholder':
                n.type = TensorType([Dyn, 4, 227, 227])

        constraints = transform_all_constraints(symbolic_traced, counter=0)
        solver = z3.Solver()
        solver.add(constraints)
        self.assertEqual(solver.check(), z3.unsat)

    def test_alexnet3(self):
        alexnet = models.alexnet()
        symbolic_traced : torch.fx.GraphModule = symbolic_trace(alexnet)

        for n in symbolic_traced.graph.nodes:
            if n.op == 'placeholder':
                n.type = TensorType([Dyn, Dyn, 227, 227])

        constraints = transform_all_constraints(symbolic_traced, counter=0)
        solver = z3.Solver()
        solver.add(constraints)
        self.assertEqual(solver.check(), z3.sat)

    def test_alexnet4(self):
        alexnet = models.alexnet()
        symbolic_traced : torch.fx.GraphModule = symbolic_trace(alexnet)

        for n in symbolic_traced.graph.nodes:
            if n.op == 'placeholder':
                n.type = TensorType([Dyn, Dyn, 227])

        constraints = transform_all_constraints(symbolic_traced, counter=0)
        solver = z3.Solver()
        solver.add(constraints)
        self.assertEqual(solver.check(), z3.unsat)



if __name__ == '__main__':
    unittest.main()
