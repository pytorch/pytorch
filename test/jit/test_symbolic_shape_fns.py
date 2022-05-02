# Owner(s): ["oncall: jit"]
# File specifically for testing shape functions in SSA

import operator
from textwrap import dedent

import torch
from torch import nn
from torch.testing._internal.common_methods_invocations import sample_inputs_cat_concat
from torch.testing._internal.common_utils import make_tensor
from torch.testing._internal.jit_utils import JitTestCase, execWrapper

if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )

# XXX: still in prototype


class TestSymbolicShapeFnsBase(JitTestCase):
    def setUp(self):
        self.prev_symbolic_shapes_test_enabled = (
            torch._C._jit_symbolic_shapes_test_mode_enabled()
        )
        torch._C._jit_set_symbolic_shapes_test_mode(True)

    def tearDown(self):
        torch._C._jit_set_symbolic_shapes_test_mode(
            self.prev_symbolic_shapes_test_enabled
        )


class TestSymbolicShapeFnGroups(TestSymbolicShapeFnsBase):
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
            break

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


class TestIndividualShapeFn(TestSymbolicShapeFnsBase):
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
                (make_arg(()), torch.nn.Embedding(0, 0, sparse=True)),
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
                super(CatMod, self).__init__()
                self.dim = dim

            def forward(self, x, y):
                return torch.cat([x, y], dim=self.dim)

        for inp in sample_inputs:
            mod = torch.jit.script(CatMod(**inp.kwargs).eval())

            args = inp.input
            self.assertTrue(len(args) == 2)
            out_size = mod(*args).size()
            inps = list(mod.graph.inputs())
            inps[1].setType(inps[1].type().with_sizes(args[0].size()))
            inps[2].setType(inps[2].type().with_sizes(args[1].size()))
            self.checkShapeAnalysis(out_size, mod.graph, assert_propagation=True)
