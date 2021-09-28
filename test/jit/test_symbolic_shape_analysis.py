import operator
from textwrap import dedent

import torch
from torch.testing import FileCheck
from torch.testing._internal.common_utils import make_tensor
from torch.testing._internal.jit_utils import JitTestCase, execWrapper

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
        FileCheck().check("*, 3, 2, *").check("*, 3, *, *) = prim::If").run(foo.graph)

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
        def div_inplace(x):
            x.div_(2)
            return x

        unary_ops = [
            div_inplace
        ]
        for fn in unary_ops:
            t = torch.jit.trace(fn, (torch.rand([4, 4])))
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
        def div_inplace_tensor(x, y):
            x.div_(y)
            z = x
            return z

        binary_ops = [
            div_inplace_tensor
        ]

        for fn in binary_ops:
            size_1 = [4, 4, 8]  # x (can't broadcast, inplace)
            size_2 = [4, 1, 8]
            t = torch.jit.trace(fn, (torch.rand([4]), torch.rand([4])))
            inputs = list(t.graph.inputs())
            inputs[0].setType(inputs[0].type().with_sizes(size_1))
            inputs[1].setType(inputs[1].type().with_sizes(size_2))
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
            self.assertEqual(next(graph.outputs()).type().symbolic_sizes(), [5, 8, sym1])

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
            funcs_template = dedent('''
            def func():
                return torch.arange({args})
            ''')

            inp_s = str(inp)[1:-1]  # remove tuple parens
            funcs_str = funcs_template.format(args=inp_s)
            scope = {}
            execWrapper(funcs_str, globals(), scope)
            cu = torch.jit.CompilationUnit(funcs_str)
            self.checkShapeAnalysis(list(cu.func().size()), cu.func.graph, assert_propagation=True, constant_prop=False)

    def test_shape_embedding_bag(self):
        # TODO: merge into opinfos, having difficulties there
        with torch.no_grad():
            def make_arg(shape, low=None, high=None):
                return make_tensor(shape, device='cpu', dtype=torch.int64,
                                   low=low, high=high, requires_grad=False)

            nn_inps = (
                (make_arg((40,), 0, 9), torch.nn.Embedding(20, embedding_dim=64, max_norm=1.0)),
                (make_arg((2, 4), 0, 9), torch.nn.Embedding(10, 20, sparse=True)),
                (make_arg(()), torch.nn.Embedding(0, 0, sparse=True)),
                (make_arg((2, 4), 0, 9), torch.nn.Embedding(10, 0, sparse=True)),
                (make_arg((4,), 0, 21), torch.nn.Embedding(22, 5, max_norm=1.0)),
                (make_arg((2,), 0, 1), torch.nn.Embedding.from_pretrained(torch.arange(6.).view(2, 3), max_norm=2.,
                                                                          norm_type=.5, scale_grad_by_freq=False, sparse=True)),
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

                self.checkShapeAnalysis(out_size, fn.graph, assert_propagation=True, constant_prop=False)
