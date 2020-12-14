import unittest

import cpp_extension  # noqa
import torch


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

    def test_jit_consume_op_for_list_input(self):
        iters = 6

        def foo(x):
            for i in range(iters):
                result = torch.ops.operator_benchmark._consume(torch.chunk(x, 2))
            return result

        r = torch.jit.trace(foo, torch.rand(2, 2))

        graph = str(r.graph)
        occurance = graph.count("aten::chunk")

        x = torch.rand(2, 2)
        value = r(x)

        self.assertTrue(
            all([torch.allclose(t1, t2) for t1, t2 in zip(value, torch.chunk(x, 2))])
        )
        self.assertEqual(occurance, iters)
