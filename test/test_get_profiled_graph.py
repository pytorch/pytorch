import torch
from torch.testing._internal.common_utils import enable_profiling_mode

import unittest

class TestProfiledGraph(unittest.TestCase):

    def test_get_profiled_graph(self):

        def foo(a):
            b = a + 1
            c = a * b
            return c

        with enable_profiling_mode():
            ja = torch.jit.script(foo)
            # profile once to get shapes
            a = torch.ones(1)
            ja(a)
            g = ja._profiled_graph
            for n in g.nodes():
                for o in n.outputs():

                    # tensor with a fully specified shape info
                    # these are defined in python_ir.cpp
                    # or can be inspected with dir()
                    if o.type().kind() == "TensorType":
                        assert o.type().scalarType() == 'Float'
                        assert o.type().sizes() == [1]


if __name__ == '__main__':
    unittest.main()
