# Owner(s): ["module: inductor"]

import contextlib

import numpy as np

import torch
from torch._dynamo.testing import rand_strided
from torch._dynamo.utils import same

from torch._inductor import config as inductor_config, ir, metrics
from torch._inductor.codegen.triton import TritonScheduling
from torch._inductor.graph import GraphLowering
from torch._inductor.scheduler import SchedulerNode
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.test_operators import realize
from torch._inductor.virtualized import V
from torch.testing._internal.inductor_utils import HAS_CUDA


class MockScheduler:
    available_buffer_names = ()

    @staticmethod
    def get_backend(cls, *args):
        return TritonScheduling(cls)


class ImplDetailTest(TestCase):
    _exit_stack = None

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        gm = torch.fx.symbolic_trace(lambda: 0)
        graph = GraphLowering(gm)
        graph.scheduler = MockScheduler
        cls._exit_stack = contextlib.ExitStack()
        cls._exit_stack.enter_context(V.set_graph_handler(graph))

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        cls._exit_stack.close()

    @staticmethod
    def _get_snode_body_sym_prefix(snode):
        body = snode._body
        prefix = ""

        for var in body.var_ranges:
            prefix = str(var)[0]
            break

        assert prefix
        return prefix

    @staticmethod
    def _create_computed_buffer_ax2(sizes=(32, 64), strides=None):
        """
        Create a ComputedBuffer for 'a x 2'
        """
        if strides is None:
            strides = ir.FlexibleLayout.contiguous_strides(sizes)

        box_a = ir.TensorBox.create(
            ir.Buffer(
                "a", ir.FixedLayout(torch.device("cuda"), torch.float32, sizes, strides)
            )
        )
        box_a_loader = box_a.make_loader()

        def inner_fn(index):
            return box_a_loader(index) * 2

        buf = ir.Pointwise.create(
            device=box_a.get_device(),
            dtype=box_a.get_dtype(),
            inner_fn=inner_fn,
            ranges=box_a.get_size(),
        )
        buf.realize()
        computed_buf = buf.data.data
        computed_buf.decide_layout()
        return computed_buf

    def test_reorder_twice(self):
        """
        This may happen in practice if we pick a order when fusing A and B.
        Then we pick another order for AB when we fusion C into it.

        E.g. happens for BertForMaskedLM.
        """

        buf = self._create_computed_buffer_ax2()
        snode = SchedulerNode(V.graph.scheduler, buf)
        snode.apply_new_loop_order([1, 0])
        prefix1 = self._get_snode_body_sym_prefix(snode)
        self.assertTrue(prefix1 == "z")
        snode.apply_new_loop_order([1, 0])
        prefix2 = self._get_snode_body_sym_prefix(snode)
        self.assertTrue(prefix2 == "z")

    def test_reorder_and_merge_loops(self):
        sizes = (1024, 2048)
        strides = (1, 1024)
        buf = self._create_computed_buffer_ax2(sizes, strides)
        old_sizes, old_body = buf.simplify_and_reorder()

        # Make sure loop reordering happens here
        self.assertTrue(tuple(old_sizes[0]) == tuple(reversed(sizes)), f"{old_sizes=}")
        new_sizes, new_body = SchedulerNode._merge_loops(old_sizes, old_body)
        self.assertTrue(tuple(new_sizes[0]) == (np.prod(sizes),), f"{new_sizes=}")


@inductor_config.patch(
    {
        "benchmark_kernel": True,
        "triton.unique_kernel_names": True,
    }
)
class LoopOrderingTest(TestCase):
    def do_acc_test(self, f, *args):
        expect = f(*args)
        actual = torch.compile(f)(*args)
        self.assertTrue(same(expect, actual, tol=1e-3))

    def setUp(self):
        super().setUp()
        metrics.reset()

    def test_for_reordering_reindex(self):
        """
        ComputedBuffer.iter_reoredering_reindex can cause some fusion
        opportunitiies being skipped.

        In this test case, Inductor generates 2 triton kernels before.
        By removing ComputedBuffer.iter_reoredering_reindex, we can fuse those
        two kernels into a single one.
        """

        def f(x, y):
            """
            Add a matmul since inductor may force layout for output.
            """
            return (x.sum(dim=-1) + 1) @ y

        A, B = 20, 30
        # Make the first 2 dimension not able to merge on purpose so that
        # ComputedBuffer.iter_reoredering_reindex will be updated.
        x = rand_strided([A, A, B], [B, B * A + 300, 1], device="cuda")
        y = torch.randn(A, A)

        self.do_acc_test(f, x, y)
        self.assertEqual(1, metrics.generated_kernel_count)
        expected_num_bytes = 0
        expected_num_bytes += A * A * B + A * A  # for the fused reduction
        expected_num_bytes += A * A * 3  # for matmul
        expected_num_bytes *= x.itemsize
        self.assertEqual(expected_num_bytes, metrics.num_bytes_accessed)

    def test_apbt_realize(self):
        M = 1024
        N = 2048

        def f(x, y):
            """
            There will be 2 kernels being generated without loop ordering after fusion:
              https://gist.github.com/shunting314/44df83f71de2c110232c50ac6638ed69
            """
            x = realize(x * 2)
            y = realize(y * 3)
            return x + y

        x = torch.randn(M, N)
        y = torch.randn(N, M).t()

        self.do_acc_test(f, x, y)
        self.assertEqual(1, metrics.generated_kernel_count)

    def test_sum_and_t(self):
        N = 1024

        def f(x):
            return x.sum(dim=-1), x.t().contiguous()

        x = torch.randn(N, N)
        self.do_acc_test(f, x)
        self.assertEqual(1, metrics.generated_kernel_count)


if __name__ == "__main__":
    if HAS_CUDA:
        torch.set_default_device("cuda")
        run_tests()
