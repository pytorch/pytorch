# Owner(s): ["oncall: jit"]
# This file is to test that the general mechanics of shape propagation works
# For indivual ops, check out test_symbolic_shape_fns.py

import unittest

import torch
from torch import nn, Tensor
from torch.testing import FileCheck
from torch.testing._internal.jit_utils import JitTestCase
from typing import List, Any

if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_jit.py TESTNAME\n\n"
                       "instead.")

# XXX: still in prototype
class TestSymbolicShapeAnalysis(JitTestCase):
    def setUp(self):
        self.prev_symbolic_shapes_test_enabled = torch._C._jit_symbolic_shapes_test_mode_enabled()
        torch._C._jit_set_symbolic_shapes_test_mode(True)

    def tearDown(self):
        torch._C._jit_set_symbolic_shapes_test_mode(self.prev_symbolic_shapes_test_enabled)

    def test_shape_analysis(self):
        @torch.jit.script
        def foo(x, y):
            return x * y

        inputs = list(foo.graph.inputs())

        def prop_shapes_on_graph(inp0, inp1):
            inputs[0].setType(inputs[0].type().with_sizes(inp0))
            inputs[1].setType(inputs[1].type().with_sizes(inp1))
            torch._C._jit_pass_propagate_shapes_on_graph(foo.graph)

        prop_shapes_on_graph([1, 6, 5], [1, 7, 1, 5])
        FileCheck().check("1, 7, 6, 5").run(foo.graph)

        # None implicitly creates a new symbolic symbol
        prop_shapes_on_graph([None, None], [None, None, None])
        output_shape = foo.graph.findNode("aten::mul").output().type().symbolic_sizes()
        inp0_shape = inputs[0].type().symbolic_sizes()
        inp1_shape = inputs[1].type().symbolic_sizes()

        # output shape dim 0 should be taken from the second inp dim0
        # other two dims we cannot infer and are given a new symbolic shape
        self.assertEqual(output_shape[0], inp1_shape[0])
        self.assertFalse(output_shape[1] in inp0_shape + inp1_shape)
        self.assertFalse(output_shape[2] in inp0_shape + inp1_shape)

        # XXX: symbolic shapes are represented with an increasing counter of unique
        # values, use `_new_symbolic_shape_symbol` api instead of specifying negative
        # dimensions directly so there is no chance of collision between manual number
        # and current counter value.
        sym1 = torch._C._new_symbolic_shape_symbol()
        sym2 = torch._C._new_symbolic_shape_symbol()
        sym3 = torch._C._new_symbolic_shape_symbol()
        prop_shapes_on_graph([sym1, 1, sym3], [1, sym2, sym3])
        output_shape = foo.graph.findNode("aten::mul").output().type().symbolic_sizes()
        self.assertEqual(output_shape[0], sym1)
        self.assertEqual(output_shape[1], sym2)
        self.assertEqual(output_shape[2], sym3)

    def test_shared_shape_graph(self):
        @torch.jit.script
        def foo(x, y):
            return x * y, x / y

        mul_node = foo.graph.findNode("aten::mul")
        div_node = foo.graph.findNode("aten::div")

        mul_graph = torch._C._jit_shape_compute_graph_for_node(mul_node)
        div_graph = torch._C._jit_shape_compute_graph_for_node(div_node)
        self.assertIsNotNone(mul_graph)
        self.assertIs(mul_graph, div_graph)

    def test_write(self):
        @torch.jit.script
        def foo(a, b):
            return a * b

        # broadcast appends cant be removed, so we bail on propagation
        torch._C._jit_pass_propagate_shapes_on_graph(foo.graph)
        FileCheck().check("Tensor = aten::mul").run(foo.graph)

        @torch.jit.script
        def foo(y):
            x = [1, 2, 3, 4]
            x[0] = 5
            return y.view(x)

        torch._C._jit_pass_propagate_shapes_on_graph(foo.graph)
        FileCheck().check("Tensor = aten::view").run(foo.graph)

    def test_if_propagation(self):
        @torch.jit.script
        def foo(i: int, z):
            x = torch.ones([2, 3, 4, 5])
            y = z.view([z.size(i), 3, 2, z.size(i)])
            if i == 4:
                return x
            else:
                return y

        torch._C._jit_pass_constant_propagation(foo.graph)
        torch._C._jit_pass_propagate_shapes_on_graph(foo.graph)
        view = foo.graph.findNode("aten::view")

        def neg_to_one(li):
            return [elem if elem >= 0 else -1 for elem in li]

        self.assertEqual(neg_to_one(view.output().type().symbolic_sizes()), [-1, 3, 2, -1])
        if_out = next(foo.graph.findNode("prim::If").outputs())
        self.assertEqual(neg_to_one(if_out.type().symbolic_sizes()), [-1, 3, -1, -1])


    def test_size_and_sizes(self):
        @torch.jit.script
        def foo(x, y):
            return x.view(y.size(0), 8, y.size(-1))

        @torch.jit.script
        def foo2(x, y):
            return x.view(y.size())

        for graph in [foo.graph, foo2.graph]:
            inputs = list(graph.inputs())
            sym1 = torch._C._new_symbolic_shape_symbol()

            inputs[1].setType(inputs[1].type().with_sizes([5, 8, sym1]))
            torch._C._jit_pass_propagate_shapes_on_graph(graph)
            self.assertEqual(next(graph.outputs()).type().symbolic_sizes(), [5, 8, sym1])

    def test_returning_input_symbolic_shapes(self):
        mm = torch.jit.freeze(torch.jit.script(nn.Conv2d(16, 33, 3, stride=2).eval()))
        inps = list(mm.graph.inputs())
        inps[1].setType(inps[1].type().with_sizes([None, None, None, None]))
        shape_compute_graph = torch._C._jit_pass_propagate_shapes_on_graph_and_build_compute(mm.graph)
        g = shape_compute_graph.partial_eval_shape_graph()
        # to make into a jit function cant have multiple outputs
        g.makeMultiOutputIntoTuple()
        func = torch._C._create_function_from_graph("partial_eval_graph", g)
        out = func([20, 16, 5, 10])
        # first four outputs should be unknown symbolic shapes from input
        self.assertEqual(out[0:4], [20, 16, 5, 10])
        # last two are two new symbolic dims - height and width
        self.assertEqual(out[4:], list(mm(torch.rand([20, 16, 5, 10])).size()[2:]))

    def test_partial_eval_graph_conv(self):
        mm = torch.jit.freeze(torch.jit.script(nn.Conv2d(16, 33, 3, stride=2).eval()))
        shape_compute_graph = torch._C._jit_pass_propagate_shapes_on_graph_and_build_compute(mm.graph)
        output_sizes = mm.graph.findNode("aten::conv2d").output().type().symbolic_sizes()
        # calculating 0, 2 and 3 index
        for i in [0, 2, 3]:
            self.assertTrue(output_sizes[i] < 0)
        self.assertTrue(output_sizes[1] >= 0)
        g = shape_compute_graph.partial_eval_shape_graph()
        # to make into a jit function cant have multiple outputs
        g.makeMultiOutputIntoTuple()
        func = torch._C._create_function_from_graph("partial_eval_graph", g)
        inp = torch.randn(20, 16, 5, 10)
        output = func([20, 16, 5, 10])
        output_eager = list(mm(inp).size())
        for o, oe in zip(output, output_eager[0:1] + output_eager[2:]):
            self.assertEqual(o, oe)

    def checkSymShapeCompute(self, shape_compute_graph, nodes, node_output_sizes, shape_inputs):
        g = shape_compute_graph.partial_eval_shape_graph()
        self.assertTrue(len(list(g.inputs())) == len(shape_inputs))
        output_sym_map = shape_compute_graph.graph_output_to_symbolic_shape_dim()
        # map from sym shape -> index
        sym_shape_to_index = {}
        for index, output in enumerate(g.outputs()):
            sym_shape_to_index[output_sym_map[output]] = index

        g.makeMultiOutputIntoTuple()
        func = torch._C._create_function_from_graph("partial_eval_graph", g)
        sym_outputs = func(*shape_inputs)

        for node, output_shape in zip(nodes, node_output_sizes):
            output_type_sizes = node.output().type().symbolic_sizes()
            for i, sym_shape in enumerate(output_type_sizes):
                if sym_shape >= 0:
                    self.assertEqual(sym_shape, output_shape[i])
                else:
                    sym_shape_index = sym_shape_to_index[sym_shape]
                    self.assertEqual(sym_outputs[sym_shape_index], output_shape[i])

    def test_partial_eval_stitching(self):
        conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        mod = torch.jit.freeze(torch.jit.script(nn.Sequential(conv1, max_pool, conv2).eval()))

        conv1_output = conv1(torch.rand(1, 3, 224, 224))
        max_pool_output = max_pool(conv1_output)
        conv2_output = conv2(max_pool_output)

        shape_compute_graph = torch._C._jit_pass_propagate_shapes_on_graph_and_build_compute(mod.graph)
        nodes = [mod.graph.findNode("aten::max_pool2d")] + list(mod.graph.findAllNodes("aten::conv2d"))
        output_shapes = [max_pool_output.size(), conv1_output.size(), conv2_output.size()]
        self.checkSymShapeCompute(shape_compute_graph, nodes, output_shapes, ([1, 3, 224, 224],))

    def test_refinement_through_graph_stitching(self):
        class TwoConvs(torch.nn.Module):
            def __init__(self):
                super(TwoConvs, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                self.conv2 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

            def forward(self, x):
                a = self.conv1(x)
                b = self.conv2(x)
                return a + b

        mod = torch.jit.freeze(torch.jit.script(TwoConvs()).eval())
        inp_tensor = list(mod.graph.inputs())[1]
        inp_tensor.setType(inp_tensor.type().with_sizes([None, None, None, None]))
        torch._C._jit_pass_propagate_shapes_on_graph(mod.graph)
        outs = list(next(mod.graph.outputs()).node().inputs())
        out1 = outs[0].type().symbolic_sizes()
        out2 = outs[1].type().symbolic_sizes()
        self.assertTrue(out1[2] != out2[2])
        self.assertTrue(out1[3] != out2[3])
        # by joining partial eval graphs of both convs we are able to recognize the output shapes
        # are equivalent
        torch._C._jit_pass_propagate_shapes_on_graph_and_build_compute(mod.graph)
        out1 = outs[0].type().symbolic_sizes()
        out2 = outs[1].type().symbolic_sizes()
        self.assertEqual(out1, out2)

    def test_stitching_multi_output(self):
        max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False, return_indices=True)
        tensor = torch.rand(1, 3, 224, 224)
        mod = torch.jit.trace(max_pool, (tensor,))
        mod = torch.jit.freeze(mod.eval())
        inp = list(mod.graph.inputs())[1]
        inp.setType(inp.type().with_sizes([None, None, None, None]))
        output_tensor = list(mod(tensor)[0].size())
        self.run_pass('lower_all_tuples', mod.graph)
        shape_compute_graph = torch._C._jit_pass_propagate_shapes_on_graph_and_build_compute(mod.graph)
        max_pool_node = mod.graph.findNode("aten::max_pool2d_with_indices")
        outs = list(max_pool_node.outputs())
        self.assertEqual(outs[0].type().symbolic_sizes(), outs[1].type().symbolic_sizes())
        g = shape_compute_graph.partial_eval_shape_graph()
        # to make into a jit function cant have multiple outputs
        g.makeMultiOutputIntoTuple()
        func = torch._C._create_function_from_graph("partial_eval_graph", g)
        mapping = shape_compute_graph.graph_output_to_symbolic_shape_dim()
        output_shape = func(tensor.size())
        # the first 4 dims are input sym dimensions, then the ,
        self.assertEqual(list(output_shape[0:4]), list(tensor.size()))
        self.assertEqual(list(output_shape[4:]), output_tensor[2:])

    def test_sym_ir_parsing(self):
        graph_str1 = """graph(%x.1 : Float(SS(-2), SS(-3))):
                        %3 : int = prim::Constant[value=1]()
                        %4 : Tensor = aten::add(%x.1, %x.1, %3)
                        return (%4)"""
        g = torch._C.parse_ir(graph_str1)
        inp = next(g.inputs())
        out = inp.type().symbolic_sizes()
        self.assertEqual(out, [-2, -3])

    def test_stitching_concat(self):

        @torch.jit.script
        def foo1(a, b, x, y):
            return (a / b) + torch.cat([x, y])

        @torch.jit.script
        def foo2(a, b, x, y):
            return (a / b) + torch.cat([x, y], dim=-2)

        for foo in [foo1, foo2]:
            g = foo.graph
            for inp in foo.graph.inputs():
                inp.setType(inp.type().with_sizes([None, None]))

            shape_compute_graph = torch._C._jit_pass_propagate_shapes_on_graph_and_build_compute(foo.graph)
            nodes = [g.findNode("aten::div")] + [g.findNode("aten::add")] + [g.findNode("aten::cat")]

            inps = [1, 10], [20, 10], [15, 1], [5, 1]
            output_shapes = [[20, 10], [20, 10], [20, 1]]

            self.checkSymShapeCompute(shape_compute_graph, nodes, output_shapes, inps)

    @unittest.skipIf(not hasattr(torch.jit, "_shapes"), "shape functions not loaded in python")
    def test_shape_function_includes(self):
        inp_shape = [1, 16, 5, 10]
        weight_shape = [33, 16, 3, 3]
        bias = None
        stride = [2, 2]
        padding = [0, 0]
        dilation = [1, 1]
        groups = 1
        res = torch.jit._shapes.conv2d(inp_shape, weight_shape, bias, stride, padding, dilation, groups)
        self.assertEqual(res, [1, 33, 2, 4])

        m1_shape = [10, 20]
        m2_shape = [20, 10]
        res = torch.jit._shapes.matmul(m1_shape, m2_shape)
        self.assertEqual(res, [10, 10])

    def test_register_function_error_checking(self):
        # this will error before registering on global map, so
        # no issue in overwriting schema mappings
        @torch.jit.script
        def foo(x, y):
            return x + y

        node = foo.graph.findNode("aten::add")

        @torch.jit.script
        def wrong_input_types(x, y):
            x: List[int] = []
            return x
        with self.assertRaisesRegex(RuntimeError, "Expected supertype of int"):
            torch._C._jit_register_shape_compute_graph_for_node(node, wrong_input_types.graph)

        @torch.jit.script
        def wrong_output_types(x: List[int], y: List[int]):
            x: List[Tensor] = []
            return x

        with self.assertRaisesRegex(RuntimeError, "but got graph_type"):
            torch._C._jit_register_shape_compute_graph_for_node(node, wrong_output_types.graph)

        @torch.jit.script
        def too_many_inputs(x: List[int], y: List[int], z: Any, z2: Any):
            x: List[int] = []
            return x

        with self.assertRaises(RuntimeError) as error:
            torch._C._jit_register_shape_compute_graph_for_node(node, too_many_inputs.graph)

        self.assertTrue("fewer arguments than schema" in str(error.exception))
