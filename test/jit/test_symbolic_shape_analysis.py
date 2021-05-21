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

        test_cases = [
            ([1, 6, 5], [1, 7, 1, 5], "1, 7, 6, 5"),
            ([1, 6, 5], [1, None, 1, 5], "1, *, 6, 5"),
            ([None, None], [None, None, None], "*, *, *"),
        ]

        for inp0, inp1, result in test_cases:
            inputs[0].setType(inputs[0].type().with_sizes(inp0))
            inputs[1].setType(inputs[1].type().with_sizes(inp1))
            torch._C._jit_pass_propagate_shapes_on_graph(foo.graph)
            FileCheck().check(result).run(foo.graph)
