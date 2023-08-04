# Owner(s): ["module: inductor"]
import functools
import itertools
import math
import operator
from unittest.mock import patch

import torch
from torch._inductor import config, ir, test_operators
from torch._inductor.codegen.triton import TritonScheduling
from torch._inductor.debug import DebugContext
from torch._inductor.graph import GraphLowering
from torch._inductor.scheduler import (
    FusedSchedulerNode,
    LoopOrder,
    Scheduler,
    SchedulerNode,
)
from torch._inductor.utils import add_scheduler_init_hook, run_and_get_code
from torch._inductor.virtualized import ops, V
from torch.fx import symbolic_trace
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    TestCase,
)


@instantiate_parametrized_tests
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

    def test_loop_ordering_can_merge(self):
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

    def test_loop_ordering_cannot_merge(self):
        shape = (2, 3)
        snode = self._create_scheduler_node(shape)
        loop_order = LoopOrder.permute(snode, snode._body, snode._sizes, [1, 0])
        snode.apply_loop_order(loop_order)
        snode.merge_loops()

        # we can not merge the loop when the loop order is [1, 0]
        self.assertTrue(snode._sizes, (list(shape), []))

    @parametrize(
        "shape",
        [
            (5, 7),
            (3, 5, 7),
            (3, 5, 7, 9),
            (3, 5, 7, 9, 11),
            (3, 5, 7, 9, 11, 13),
            (3, 5, 7, 9, 11, 13, 15),
        ],
    )
    def test_select_loop_orders(self, shape):
        def f(x):
            x = x + 1
            x = test_operators.realize(x)
            x = x * 2
            return x

        x = torch.randn(shape).cuda()

        called = False

        def post_fn(scheduler, nodes):
            nonlocal called
            called = True
            self.assertEqual(
                len(scheduler.nodes),
                1,
                "We should have a single FusedSchedulerNode after fusion",
            )
            self.assertTrue(isinstance(scheduler.nodes[0], FusedSchedulerNode))

            add_node, mul_node = (scheduler.create_scheduler_node(n) for n in nodes)

            add_loop_orders = add_node.possible_loop_orders()
            mul_loop_orders = mul_node.possible_loop_orders()
            nchoices = math.factorial(len(shape))
            if nchoices > config.loop_ordering_search_limit:
                # fallback to quadratic enumeration of orders
                nchoices = len(shape) * (len(shape) - 1) // 2 + 3

            self.assertEqual(len(add_loop_orders), nchoices)
            self.assertEqual(len(mul_loop_orders), nchoices)

            for add_idx, mul_idx in itertools.product(range(nchoices), range(nchoices)):
                add_order = add_loop_orders[add_idx]
                mul_order = mul_loop_orders[mul_idx]

                (add_write_dep,) = add_order.read_writes.writes
                (mul_read_dep,) = mul_order.read_writes.reads

                if add_idx == mul_idx:
                    self.assertTrue(add_order.sizes == mul_order.sizes)
                    self.assertTrue(add_write_dep == mul_read_dep)
                else:
                    self.assertFalse(add_order.sizes == mul_order.sizes)
                    self.assertFalse(add_write_dep == mul_read_dep)

            self.assertEqual(
                nchoices,
                len(FusedSchedulerNode.select_loop_orders((add_node, mul_node))),
            )

        with add_scheduler_init_hook(pre_fn=None, post_fn=post_fn):
            torch.compile(f, fullgraph=True)(x)

        self.assertTrue(called)

    @patch.object(torch._dynamo.config, "automatic_dynamic_shapes", False)
    @parametrize(
        "shape",
        [
            (2, 3),
            (16, 32),
            (32, 32),
            (32, 64),
        ],
    )
    def test_tiling_apbt(self, shape):
        def f(a, b):
            return a + b.t()

        assert len(shape) == 2
        a = torch.randn(shape).cuda()
        b = torch.randn(list(reversed(shape))).cuda()

        called = False
        numel = shape[0] * shape[1]

        def hook_fn(scheduler, nodes):
            nonlocal called
            called = True

            self.assertTrue(len(nodes) == 1)
            snode = scheduler.create_scheduler_node(nodes[0])
            candidate_tilings = TritonScheduling.candidate_tilings(snode)

            # only the read to argument 'b' need to be tiled
            self.assertTrue(len(candidate_tilings) == 1)
            self.assertTrue(
                V.graph.sizevars.size_hints(candidate_tilings[0].tiling) == shape
            )
            # b may get an argname like arg3_1 if it's not the first graph
            # being compiled.
            # self.assertTrue(candidate_tilings[0].name == "arg1_1")
            score = numel
            for s in shape:
                if s % 32 == 0:
                    score *= 2
            self.assertTrue(candidate_tilings[0].score == score)

            tiling = TritonScheduling.select_tiling([snode], numel, 1)
            self.assertTrue(V.graph.sizevars.size_hints(tiling) == (*shape, 1))

        with add_scheduler_init_hook(None, hook_fn):
            torch.compile(f, fullgraph=True)(a, b)
        self.assertTrue(called)

    @patch.object(torch._dynamo.config, "automatic_dynamic_shapes", False)
    @parametrize(
        "shape",
        [
            (2, 3),
            (16, 32),
            (32, 32),
            (32, 64),
        ],
    )
    def test_tiling_atpb(self, shape):
        """
        Unlink 'a + b.t()', 'a.t() + b' will return an non contiguous tensor.
        This results in 2 candidate tilings, one for a and one for the result.
        """

        def f(a, b):
            return a.t() + b

        assert len(shape) == 2
        a = torch.randn(list(reversed(shape))).cuda()
        b = torch.randn(shape).cuda()

        called = False
        numel = shape[0] * shape[1]

        def hook_fn(scheduler, nodes):
            nonlocal called
            called = True

            self.assertTrue(len(nodes) == 1)
            snode = scheduler.create_scheduler_node(nodes[0])
            candidate_tilings = TritonScheduling.candidate_tilings(snode)

            # only the read to argument 'b' need to be tiled
            self.assertTrue(len(candidate_tilings) == 2)
            self.assertTrue(
                V.graph.sizevars.size_hints(candidate_tilings[0].tiling) == shape
            )
            self.assertTrue(
                V.graph.sizevars.size_hints(candidate_tilings[1].tiling) == shape
            )
            score = numel
            for s in shape:
                if s % 32 == 0:
                    score *= 2
            self.assertTrue(candidate_tilings[0].score == score)
            # candidate 1 is for write MemoryDep and we double score for write
            self.assertTrue(candidate_tilings[1].score == 2 * score)

            tiling = TritonScheduling.select_tiling([snode], numel, 1)
            self.assertTrue(V.graph.sizevars.size_hints(tiling) == (*shape, 1))

        with add_scheduler_init_hook(None, hook_fn):
            torch.compile(f, fullgraph=True)(a, b)
        self.assertTrue(called)

    def test_tiling_three(self):
        """
        b and c has conflict tiling requirements actually.
        3d tiling should be able to make both happy.
        The 2 tiling candidates have the same score, so right now
        a random one is picked. (it's a random one since we use Set
        to store read/write dependencies and the order is non-determinitic)
        """

        def f(a, b, c):
            return a + b.permute(1, 2, 0) + c.permute(2, 0, 1)

        shape = (10, 10, 10)
        a = torch.randn(shape).cuda()
        b = torch.randn(shape).cuda()
        c = torch.randn(shape).cuda()

        called = False

        def hook_fn(scheduler, nodes):
            nonlocal called
            called = True

            self.assertTrue(len(nodes) == 1)
            snode = scheduler.nodes[0]
            choices = TritonScheduling.candidate_tilings(snode)
            self.assertTrue(len(choices) == 2)
            self.assertTrue(choices[0].tiling != choices[1].tiling)
            self.assertTrue(choices[0].score == choices[1].score)

            selected = TritonScheduling.select_tiling(
                [snode], functools.reduce(operator.mul, shape), 1
            )
            self.assertTrue(selected == (10, 100, 1) or selected == (100, 10, 1))

        with add_scheduler_init_hook(None, hook_fn):
            torch.compile(f, fullgraph=True)(a, b, c)

        self.assertTrue(called)

    @staticmethod
    def lite_init_scheduler(scheduler, nodes):
        scheduler.backends = {}
        scheduler.available_buffer_names = {
            *V.graph.graph_inputs.keys(),
            *V.graph.constants.keys(),
        }

        scheduler.nodes = [scheduler.create_scheduler_node(n) for n in nodes]
        scheduler.compute_predecessors()

    def test_outer_reduction_fusion(self):
        def f(x):
            x = x.sin()
            x = test_operators.realize(x)
            return x.sum(dim=0)

        x = torch.randn(128, 256).cuda()
        called = False

        def hook_fn(scheduler, nodes):
            nonlocal called
            called = True

            self.lite_init_scheduler(scheduler, nodes)
            snodes = scheduler.nodes
            self.assertTrue(scheduler.can_fuse(snodes[0], snodes[1]))

        with add_scheduler_init_hook(hook_fn):
            torch.compile(f, fullgraph=True)(x)

        self.assertTrue(called)

    def test_reshape_fusion(self):
        def f(x, y):
            x = x + y
            x = test_operators.realize(x)
            s0, s1, s2, s3, s4 = x.size()
            x = x.permute(1, 0, 4, 3, 2).reshape(s0 * s1, s2 * s3 * s4)
            return x

        x = torch.randn(9, 8, 7, 6, 5, device=self.device)
        y = torch.randn(9, 8, 7, 6, 5, device=self.device)

        called = False

        def hook_fn(scheduler, nodes):
            nonlocal called
            called = True

            self.lite_init_scheduler(scheduler, nodes)
            snodes = scheduler.nodes

            sin_snode, reshape_snode = snodes
            self.assertTrue(str(reshape_snode.node).count("ModularIndexing") == 5)

            self.assertTrue(len(reshape_snode.node.get_size()) == 2)
            self.assertTrue(len(reshape_snode._sizes[0]) == 5)

            # we can fuse sin_snode and reshape_snode since we split the var
            # ranges for the reshape_snode.
            self.assertTrue(scheduler.can_fuse(sin_snode, reshape_snode))

        with add_scheduler_init_hook(hook_fn):
            actual = torch.compile(f, fullgraph=True)(x, y)
            ref = f(x, y)
            self.assertTrue(torch.allclose(actual, ref))

        self.assertTrue(called)

    @parametrize(
        "loop_range",
        [
            32,
            64,
            128,
            129,
            256,
        ],
    )
    def test_chain_fusion(self, loop_range):
        def f(x):
            for i in range(loop_range):
                x = x + 1
                if i != loop_range - 1:
                    x = test_operators.realize(x)
            return x

        x = torch.rand(10, 10).cuda()

        called = False

        def hook_fn(scheduler, nodes):
            nonlocal called
            called = True
            self.assertTrue(len(nodes) == loop_range)
            # loop_range nodes are fused into a single node
            expected_num_fused = (
                loop_range + config.max_fusion_size - 1
            ) // config.max_fusion_size
            self.assertTrue(
                len(scheduler.nodes) == expected_num_fused,
                f"Number of fused scheduler node: {len(scheduler.nodes)}, expected {expected_num_fused}",
            )
            self.assertTrue(
                sum(len(n.get_nodes()) for n in scheduler.nodes) == loop_range
            )

        with add_scheduler_init_hook(None, hook_fn):
            actual, (code,) = run_and_get_code(torch.compile(f, fullgraph=True), x)
            ref = f(x)
            self.assertTrue(torch.allclose(actual, ref))

        self.assertTrue(called)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
