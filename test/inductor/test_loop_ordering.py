# Owner(s): ["module: inductor"]
import functools
import operator

import torch
from torch._inductor import ir
from torch._inductor.debug import DebugContext
from torch._inductor.graph import GraphLowering
from torch._inductor.scheduler import LoopOrder, Scheduler, SchedulerNode
from torch._inductor.virtualized import ops, V
from torch.fx import symbolic_trace
from torch.testing._internal.common_utils import TestCase


class LoopOrderingTest(TestCase):
    device = torch.device("cuda:0")

    def _create_scheduler_node(self, shape):
        with DebugContext():
            stride = (shape[1], 1)  # contiguous
            ninput = 1

            def inner_fn(iter_vars):
                x, y = iter_vars
                idx = x * stride[0] + y * stride[1]
                a = ops.load("input0", idx)
                return ops.sin(a)

            pw_irnode = ir.Pointwise.create(
                device=self.device, dtype=torch.float32, inner_fn=inner_fn, ranges=shape
            )
            V.set_graph_handler(GraphLowering(symbolic_trace(lambda x: x)))
            V.graph.graph_inputs = {
                f"input{i}": ir.TensorBox.create(
                    ir.InputBuffer(
                        f"input{i}",
                        ir.FixedLayout(self.device, torch.float32, shape, stride),
                    )
                )
                for i in range(ninput)
            }
            V.graph.graph_outputs = [pw_irnode]
            pw_irnode.realize()

            scheduler = V.graph.scheduler = Scheduler(V.graph.buffers)

            snode = SchedulerNode(
                scheduler,
                pw_irnode.data.data,
                scheduler.get_backend(self.device).group_fn,
            )
            return snode

    def test_loop_ordering_and_can_merge(self):
        r"""
        For a pointwise kernel operating on an input contiguous tensor of
        shape [a, b], if the loop ordering is [a, b], then we can merge
        these 2 dimensions. However, if we pick loop order [b, a],
        we would NOT be able to merge these 2 dimensions.
        """
        shape = (2, 3)
        snode = self._create_scheduler_node(shape)
        loop_order = LoopOrder.permute(snode, snode._body, snode._sizes, [0, 1])
        snode.apply_loop_order(loop_order)
        snode.merge_loops()
        # loop get merged when loop order is [0, 1]
        self.assertTrue(snode._sizes, ([functools.reduce(operator.mul, shape)], []))

    def test_loop_ordering_and_cannot_merge(self):
        shape = (2, 3)
        snode = self._create_scheduler_node(shape)
        loop_order = LoopOrder.permute(snode, snode._body, snode._sizes, [1, 0])
        snode.apply_loop_order(loop_order)
        snode.merge_loops()

        # we can not merge the loop when the loop order is [1, 0]
        self.assertTrue(snode._sizes, (list(shape), []))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
