import torch
from torch.testing._internal.jit_utils import JitTestCase

from torch.testing import FileCheck
from typing import List


if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_jit.py TESTNAME\n\n"
                       "instead.")

# XXX: still in prototype
class TestSymbolicShapeAnalysis(JitTestCase):
    def test_shape_analysis(self):
        @torch.jit.script
        def broadcast(a: List[int], b: List[int]):
            dimsA = len(a)
            dimsB = len(b)
            ndim = max(dimsA, dimsB)
            expandedSizes : List[int] = []

            for i in range(ndim):
                offset = ndim - 1 - i
                dimA = dimsA - 1 - offset
                dimB = dimsB - 1 - offset
                sizeA = a[dimA] if (dimA >= 0) else 1
                sizeB = b[dimB] if (dimB >= 0) else 1

                if sizeA != sizeB and sizeA != 1 and sizeB != 1:
                    raise Exception("The size of tensor a {} must match the size of tensor b ("
                                    "{}) at non-singleton dimension {}".format(sizeA, sizeB, i))

                expandedSizes.append(sizeB if sizeA == 1 else sizeA)

            return expandedSizes

        @torch.jit.script
        def foo(x, y):
            return x * y

        torch._C._jit_register_operator_shape_function(foo.graph.findNode("aten::mul"), broadcast.graph)
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

    def test_sharing_of_list_len(self):
        # testing generic sharing of logic, a la _convolution and conv2s
        @torch.jit.script
        def adaptive_avg_pool2d(self, out: List[int]):
            assert len(out) == 2
            out2 : List[int] = []
            for elem in out:
                out2.append(elem)
            return out2

        @torch.jit.script
        def foo(x, out: List[int]):
            return torch.nn.functional.adaptive_avg_pool2d(x, out)

        self.run_pass("inline", foo.graph)
        torch._C._jit_register_operator_shape_function(foo.graph.findNode("aten::adaptive_avg_pool2d"), adaptive_avg_pool2d.graph)
        torch._C._jit_pass_propagate_shapes_on_graph(foo.graph)
        FileCheck().check("Tensor(*, *)").check_same("adaptive_avg_pool2d").run(foo.graph)
