from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import cpp_extension # noqa

import unittest



class TestConsumeOp(unittest.TestCase):
    def test_jit_consume_op(self):
        iters = 6

        def foo(x):
            for i in range(iters):
                result = torch.ops.operator_benchmark._consume(torch.sum(x))
            return result

        r = torch.jit.trace(foo, (torch.rand(2, 2)))

        graph = str(r.graph)
        occurance = graph.count("aten::sum")

        x = torch.rand(2, 2)
        value = r(x)
        self.assertEqual(value, torch.sum(x))
        self.assertEqual(occurance, iters)
