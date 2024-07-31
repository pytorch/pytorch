# Owner(s): ["oncall: jit"]

import operator
import unittest
from textwrap import dedent
from typing import Any, List

import torch
from torch import nn, Tensor
from torch.testing import FileCheck
from torch.testing._internal.common_methods_invocations import sample_inputs_cat_concat
from torch.testing._internal.common_utils import make_tensor
from torch.testing._internal.jit_utils import execWrapper, JitTestCase

if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )


# XXX: still in prototype
class TestSymbolicShapeAnalysis(JitTestCase):
    def setUp(self):
        super(JitTestCase, self).setUp()
        self.prev_symbolic_shapes_test_enabled = (
            torch._C._jit_symbolic_shapes_test_mode_enabled()
        )
        torch._C._jit_set_symbolic_shapes_test_mode(True)

    def tearDown(self):
        torch._C._jit_set_symbolic_shapes_test_mode(
            self.prev_symbolic_shapes_test_enabled
        )

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

        self.assertEqual(
            neg_to_one(view.output().type().symbolic_sizes()), [-1, 3, 2, -1]
        )
        if_out = next(foo.graph.findNode("prim::If").outputs())
        self.assertEqual(neg_to_one(if_out.type().symbolic_sizes()), [-1, 3, -1, -1])

    def test_unary_shape_functions(self):
        unary_ops = [
            torch.nn.functional.hardtanh,
        ]
        for fn in unary_ops:
            t = torch.jit.trace(fn, (torch.rand([4, 4])))
            ten_input = next(t.graph.inputs())
            ten_input.setType(ten_input.type().with_sizes([2, 2]))
            torch._C._jit_pass_propagate_shapes_on_graph(t.graph)
            self.assertEqual(next(t.graph.outputs()).type().symbolic_sizes(), [2, 2])

    def test_unary_shape_fns_inplace(self):
        def mul_inplace(x: torch.Tensor):
            y = x.mul_(2)
            return y

        unary_ops = [mul_inplace]
        for fn in unary_ops:
            # t = torch.jit.trace(fn, torch.rand([4, 4]))  # For some reason tracing is erroring out.
            t = torch.jit.script(fn)
            ten_input = next(t.graph.inputs())
            ten_input.setType(ten_input.type().with_sizes([2, 2]))
            torch._C._jit_pass_propagate_shapes_on_graph(t.graph)
            self.assertEqual(next(t.graph.outputs()).type().symbolic_sizes(), [2, 2])

    def test_binary_shape_functions(self):
        binary_ops = [
            operator.__mul__,
            operator.__truediv__,
            operator.__gt__,
            operator.__add__,
        ]

        for fn in binary_ops:
            size_1 = [1, 4, 8]
            size_2 = [4, 1, 8]
            t = torch.jit.trace(fn, (torch.rand([4]), torch.rand([4])))
            inputs = list(t.graph.inputs())
            inputs[0].setType(inputs[0].type().with_sizes(size_1))
            inputs[1].setType(inputs[1].type().with_sizes(size_2))
            torch._C._jit_pass_propagate_shapes_on_graph(t.graph)
            self.assertEqual(next(t.graph.outputs()).type().symbolic_sizes(), [4, 4, 8])

    def test_binary_shape_fns_inplace(self):
        def div_inplace_tensor(x: torch.Tensor, y: torch.Tensor):
            z = x.div_(y)
            return z

        def add_inplace_tensor(x: torch.Tensor, y: torch.Tensor):
            z = x.add_(y)
            return z

        binary_ops = [
            div_inplace_tensor,
            add_inplace_tensor,
        ]

        for fn in binary_ops:
            size_1 = [4, 4, 8]  # x (can't broadcast because it's an inplace op)
            t = torch.jit.script(fn)
            inputs = list(t.graph.inputs())
            inputs[0].setType(inputs[0].type().with_sizes(size_1))
            # Intentionally not populate the type of inputs[1]
            torch._C._jit_pass_propagate_shapes_on_graph(t.graph)
            self.assertEqual(next(t.graph.outputs()).type().symbolic_sizes(), [4, 4, 8])

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
            self.assertEqual(
                next(graph.outputs()).type().symbolic_sizes(), [5, 8, sym1]
            )

    def test_adaptive_avg_pool2d(self):
        inps = [
            [(1, 64, 8, 9), (5, 7)],
            [(1, 64, 10, 9), (7)],
            [(1, 64, 10, 9), (5, None)],
            [(1, 8, 4, 3), (None, None)],
            [(1, 8, 4, 3), (None, 5)],
        ]

        for inp in inps:
            t = torch.randn(*inp[0])
            out_size = torch.nn.functional.adaptive_avg_pool2d(t, inp[1]).size()

            def foo(x):
                return torch.nn.functional.adaptive_avg_pool2d(x, inp[1])

            fn = torch.jit.trace(foo, (t,))
            torch._C._jit_erase_non_input_shape_information(fn.graph)
            torch._C._jit_pass_peephole(fn.graph)
            torch._C._jit_pass_constant_propagation(fn.graph)
            self.checkShapeAnalysis(out_size, fn.graph, assert_propagation=True)

    def test_conv_deconv(self):
        for (
            inp_shape,
            weight_shape,
            bias,
            stride,
            padding,
            output_padding,
            dilation,
            groups,
            mod,
        ) in [
            ([32, 6, 10], [16, 3, 3], None, 2, 2, 1, 1, 2, torch.nn.functional.conv1d),
            (
                [32, 16, 10],
                [16, 3, 3],
                None,
                2,
                2,
                1,
                1,
                2,
                torch.nn.functional.conv_transpose1d,
            ),
            (
                [1, 32, 5, 10],
                [30, 16, 3, 3],
                None,
                [2, 2],
                [0, 0],
                0,
                1,
                2,
                torch.nn.functional.conv2d,
            ),
            (
                [1, 30, 5, 10],
                [30, 16, 3, 3],
                None,
                [2, 2],
                [0, 0],
                0,
                1,
                2,
                torch.nn.functional.conv_transpose2d,
            ),
            (
                [3, 14, 10, 66, 55],
                [2, 7, 7, 4, 4],
                None,
                1,
                1,
                2,
                1,
                2,
                torch.nn.functional.conv3d,
            ),
            (
                [3, 2, 10, 66, 55],
                [2, 7, 7, 4, 4],
                None,
                1,
                1,
                0,
                1,
                2,
                torch.nn.functional.conv_transpose3d,
            ),
        ]:
            inp = torch.rand(inp_shape)
            weight = torch.rand(weight_shape)
            if mod in [
                torch.nn.functional.conv1d,
                torch.nn.functional.conv2d,
                torch.nn.functional.conv3d,
            ]:
                res = mod(inp, weight, bias, stride, padding, dilation, groups).size()
            else:
                res = mod(
                    inp, weight, bias, stride, padding, output_padding, dilation, groups
                ).size()

            def foo(inp, weight):
                if mod in [
                    torch.nn.functional.conv1d,
                    torch.nn.functional.conv2d,
                    torch.nn.functional.conv3d,
                ]:
                    return mod(inp, weight, bias, stride, padding, dilation, groups)
                else:
                    return mod(
                        inp,
                        weight,
                        bias,
                        stride,
                        padding,
                        output_padding,
                        dilation,
                        groups,
                    )

            fn = torch.jit.trace(foo, (inp, weight))
            torch._C._jit_erase_non_input_shape_information(fn.graph)
            torch._C._jit_pass_peephole(fn.graph)
            torch._C._jit_pass_constant_propagation(fn.graph)
            self.checkShapeAnalysis(res, fn.graph, assert_propagation=True)

    def test_arange_shape(self):
        # no opinfo for tensor constructors
        inps = [
            (10,),
            (10, 10),
            (0, 10),
            (0, 1000),
            (1, -1, -1),
            (1, 0, -1),
            (1, 2, 1),
            (0.6, 0.89, 0.1),
            (1, 10, 0.3),
            (1, 10, 4),
            (0.6, 0.7, 0.8),
            (1, 10, 0.3),
            # (True,),  TODO: https://github.com/pytorch/pytorch/issues/63405
            # (False,), TODO: https://github.com/pytorch/pytorch/issues/63405
            (0, 5),
            (0, 5, 2),
            (0, 5 + 1e-6),
            (0, 5 - 1e-6),
            (10, -1 + 1e-6, -1),
            (10, -1, -1),
            (10, -1 - 1e-6, -1),
        ]

        for inp in inps:
            funcs_template = dedent(
                """
            def func():
                return torch.arange({args})
            """
            )

            inp_s = str(inp)[1:-1]  # remove tuple parens
            funcs_str = funcs_template.format(args=inp_s)
            scope = {}
            execWrapper(funcs_str, globals(), scope)
            cu = torch.jit.CompilationUnit(funcs_str)
            self.checkShapeAnalysis(
                list(cu.func().size()),
                cu.func.graph,
                assert_propagation=True,
                constant_prop=False,
            )

    def test_shape_embedding_bag(self):
        # TODO: merge into opinfos, having difficulties there
        with torch.no_grad():

            def make_arg(shape, low=None, high=None):
                return make_tensor(
                    shape,
                    device="cpu",
                    dtype=torch.int64,
                    low=low,
                    high=high,
                    requires_grad=False,
                )

            nn_inps = (
                (
                    make_arg((40,), 0, 9),
                    torch.nn.Embedding(20, embedding_dim=64, max_norm=1.0),
                ),
                (make_arg((2, 4), 0, 9), torch.nn.Embedding(10, 20, sparse=True)),
                (make_arg((0,)), torch.nn.Embedding(0, 0, sparse=True)),
                (make_arg((2, 4), 0, 9), torch.nn.Embedding(10, 0, sparse=True)),
                (make_arg((4,), 0, 21), torch.nn.Embedding(22, 5, max_norm=1.0)),
                (
                    make_arg((2,), 0, 1),
                    torch.nn.Embedding.from_pretrained(
                        torch.arange(6.0).view(2, 3),
                        max_norm=2.0,
                        norm_type=0.5,
                        scale_grad_by_freq=False,
                        sparse=True,
                    ),
                ),
            )

            for inp, module in nn_inps:
                kwargs = {
                    "weight": module.weight.detach(),
                    "padding_idx": module.padding_idx,
                    "max_norm": module.max_norm,
                    "norm_type": module.norm_type,
                    "scale_grad_by_freq": module.scale_grad_by_freq,
                    "sparse": module.sparse,
                }

                out_size = torch.nn.functional.embedding(inp, **kwargs).size()

                def foo(x):
                    return torch.nn.functional.embedding(inp, **kwargs)

                fn = torch.jit.trace(foo, (inp.detach(),), check_trace=False)

                self.checkShapeAnalysis(
                    out_size, fn.graph, assert_propagation=True, constant_prop=False
                )

    def test_shape_concat(self):
        # TODO: unify with opinfo tests, traces of lists dont preserve sizes in IR
        sample_inputs = sample_inputs_cat_concat(None, "cpu", torch.float, False)

        class CatMod(nn.Module):
            __constants__ = ["dim"]

            def __init__(self, dim=0):
                super().__init__()
                self.dim = dim

            def forward(self, x, y):
                return torch.cat([x, y], dim=self.dim)

        for inp in sample_inputs:
            mod = torch.jit.script(CatMod(**inp.kwargs).eval())

            args = inp.input

            # This test is hard-coded only to work with two sample inputs
            # but the OpInfo may have more/less
            if len(args) != 2:
                continue

            out_size = mod(*args).size()
            inps = list(mod.graph.inputs())
            inps[1].setType(inps[1].type().with_sizes(args[0].size()))
            inps[2].setType(inps[2].type().with_sizes(args[1].size()))
            self.checkShapeAnalysis(out_size, mod.graph, assert_propagation=True)

    def assert_shape_equal_scripted(self, script_fn, given_ins):
        expected_res = script_fn(*given_ins)
        g = script_fn.graph
        graph_ins = list(g.inputs())
        self.assertEqual(len(given_ins), len(graph_ins))
        for inp, graph_in in zip(given_ins, graph_ins):
            graph_in.setType(graph_in.type().with_sizes(inp.size()))

        out_sizes = [out.size() for out in expected_res]
        self.checkShapeAnalysis(out_sizes, g, assert_propagation=True)

    def test_convolution_backward(self):
        # No opinfos for ops that are not part of the Python API
        # Also, as the return shapes are the input, weight, and bias shape, there is no point
        # in a really complicated test

        input = torch.randn(
            (16, 16, 8, 8), dtype=torch.float32, device="cpu", requires_grad=True
        )
        weight = torch.randn(
            (8, 4, 3, 3), dtype=torch.float32, device="cpu", requires_grad=True
        )
        out_grad = torch.randn((16, 8, 8, 8), dtype=torch.float32, device="cpu")

        @torch.jit.script
        def conv_bwd(input, weight, grad):
            bias_sizes = [
                8,
            ]
            args = ([1, 1], [1, 1], [1, 1], False, [0, 0], 4, [True, True, True])
            return torch.ops.aten.convolution_backward(
                grad, input, weight, bias_sizes, *args
            )

        self.assert_shape_equal_scripted(conv_bwd, (input, weight, out_grad))

        @torch.jit.script
        def conv_bwd_2(input, weight, grad):
            bias_sizes = None
            args = ([1, 1], [1, 1], [1, 1], False, [0, 0], 4, [True, True, True])
            return torch.ops.aten.convolution_backward(
                grad, input, weight, bias_sizes, *args
            )

        self.assert_shape_equal_scripted(conv_bwd_2, (input, weight, out_grad))

    def test_returning_input_symbolic_shapes(self):
        mm = torch.jit.freeze(torch.jit.script(nn.Conv2d(16, 33, 3, stride=2).eval()))
        inps = list(mm.graph.inputs())
        inps[1].setType(inps[1].type().with_sizes([None, None, None, None]))
        shape_compute_graph = (
            torch._C._jit_pass_propagate_shapes_on_graph_and_build_compute(mm.graph)
        )
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
        shape_compute_graph = (
            torch._C._jit_pass_propagate_shapes_on_graph_and_build_compute(mm.graph)
        )
        output_sizes = (
            mm.graph.findNode("aten::conv2d").output().type().symbolic_sizes()
        )
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

    def checkSymShapeCompute(
        self, shape_compute_graph, nodes, node_output_sizes, shape_inputs
    ):
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
        conv1 = torch.nn.Conv2d(
            3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        max_pool = torch.nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )
        conv2 = nn.Conv2d(
            64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )

        mod = torch.jit.freeze(
            torch.jit.script(nn.Sequential(conv1, max_pool, conv2).eval())
        )

        conv1_output = conv1(torch.rand(1, 3, 224, 224))
        max_pool_output = max_pool(conv1_output)
        conv2_output = conv2(max_pool_output)

        shape_compute_graph = (
            torch._C._jit_pass_propagate_shapes_on_graph_and_build_compute(mod.graph)
        )
        nodes = [mod.graph.findNode("aten::max_pool2d")] + list(
            mod.graph.findAllNodes("aten::conv2d")
        )
        output_shapes = [
            max_pool_output.size(),
            conv1_output.size(),
            conv2_output.size(),
        ]
        self.checkSymShapeCompute(
            shape_compute_graph, nodes, output_shapes, ([1, 3, 224, 224],)
        )

    def test_refinement_through_graph_stitching(self):
        class TwoConvs(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = torch.nn.Conv2d(
                    3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
                )
                self.conv2 = torch.nn.Conv2d(
                    3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
                )

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
        max_pool = torch.nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=1,
            ceil_mode=False,
            return_indices=True,
        )
        tensor = torch.rand(1, 3, 224, 224)
        mod = torch.jit.trace(max_pool, (tensor,))
        mod = torch.jit.freeze(mod.eval())
        inp = list(mod.graph.inputs())[1]
        inp.setType(inp.type().with_sizes([None, None, None, None]))
        output_tensor = list(mod(tensor)[0].size())
        self.run_pass("lower_all_tuples", mod.graph)
        shape_compute_graph = (
            torch._C._jit_pass_propagate_shapes_on_graph_and_build_compute(mod.graph)
        )
        max_pool_node = mod.graph.findNode("aten::max_pool2d_with_indices")
        outs = list(max_pool_node.outputs())
        self.assertEqual(
            outs[0].type().symbolic_sizes(), outs[1].type().symbolic_sizes()
        )
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

            shape_compute_graph = (
                torch._C._jit_pass_propagate_shapes_on_graph_and_build_compute(
                    foo.graph
                )
            )
            nodes = (
                [g.findNode("aten::div")]
                + [g.findNode("aten::add")]
                + [g.findNode("aten::cat")]
            )

            inps = [1, 10], [20, 10], [15, 1], [5, 1]
            output_shapes = [[20, 10], [20, 10], [20, 1]]

            self.checkSymShapeCompute(shape_compute_graph, nodes, output_shapes, inps)

    @unittest.skipIf(
        not hasattr(torch.jit, "_shapes"), "shape functions not loaded in python"
    )
    def test_shape_function_includes(self):
        inp_shape = [1, 16, 5, 10]
        weight_shape = [33, 16, 3, 3]
        bias = None
        stride = [2, 2]
        padding = [0, 0]
        dilation = [1, 1]
        groups = 1
        res = torch.jit._shapes.conv2d(
            inp_shape, weight_shape, bias, stride, padding, dilation, groups
        )
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
            torch._C._jit_register_shape_compute_graph_for_node(
                node, wrong_input_types.graph
            )

        @torch.jit.script
        def wrong_output_types(x: List[int], y: List[int]):
            x: List[Tensor] = []
            return x

        with self.assertRaisesRegex(RuntimeError, "but got graph_type"):
            torch._C._jit_register_shape_compute_graph_for_node(
                node, wrong_output_types.graph
            )

        @torch.jit.script
        def too_many_inputs(x: List[int], y: List[int], z: Any, z2: Any):
            x: List[int] = []
            return x

        with self.assertRaises(RuntimeError) as error:
            torch._C._jit_register_shape_compute_graph_for_node(
                node, too_many_inputs.graph
            )

        self.assertTrue("fewer arguments than schema" in str(error.exception))

    def test_cross_entropy_loss(self):
        @torch.jit.script
        def foo(x, y):
            return torch.ops.aten.cross_entropy_loss(x, y, reduction=0)

        inputs = list(foo.graph.inputs())
        inputs[0].setType(inputs[0].type().with_sizes([8, 2]))
        inputs[1].setType(
            inputs[1]
            .type()
            .with_sizes(
                [
                    8,
                ]
            )
        )
        torch._C._jit_pass_propagate_shapes_on_graph(foo.graph)
        self.assertEqual(
            next(foo.graph.outputs()).type().sizes(),
            [
                8,
            ],
        )

    def test_squeeze_dims(self):
        @torch.jit.script
        def foo(x):
            return torch.ops.aten.squeeze(x, dim=0)

        input = next(foo.graph.inputs())
        input.setType(input.type().with_sizes([1, 5, 8]))
        torch._C._jit_pass_propagate_shapes_on_graph(foo.graph)
        self.assertEqual(next(foo.graph.outputs()).type().symbolic_sizes(), [5, 8])
