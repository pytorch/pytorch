# Owner(s): ["module: inductor"]

import contextlib

import numpy as np

import torch
from torch import nn
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

if HAS_CUDA:
    torch.set_default_device("cuda")


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
        self.assertTrue(prefix1 == "y")
        snode.apply_new_loop_order([1, 0])
        prefix2 = self._get_snode_body_sym_prefix(snode)
        self.assertTrue(prefix2 == "y")

    def test_reorder_and_merge_loops(self):
        sizes = (1024, 2048)
        strides = (1, 1024)
        buf = self._create_computed_buffer_ax2(sizes, strides)
        old_sizes, old_body = buf.simplify_and_reorder()

        # Make sure loop reordering happens here
        self.assertTrue(tuple(old_sizes[0]) == tuple(reversed(sizes)), f"{old_sizes=}")
        new_body = old_body.merge_loops()
        new_sizes = new_body.sizes
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

        x = torch.randn(N, N * 2)
        self.do_acc_test(f, x)
        self.assertEqual(1, metrics.generated_kernel_count)

    def test_pw_outer_red(self):
        def f(x):
            x = realize(x + 1)
            return x.sum(dim=[0, 1])

        # make the first 2 dimension small so we don't split the reduction
        x = torch.randn(2, 4, 512)
        self.do_acc_test(f, x)
        self.assertEqual(1, metrics.generated_kernel_count)

    def test_pw_outer_red_2(self):
        """
        The pointwise kernel is a fused kernel
        """

        def f(x):
            x = realize(x + 1)
            x = realize(x - 2)
            x = realize(x * 3)
            return x.sum(dim=[0, 1])

        # make the first 2 dimension small so we don't split the reduction
        x = torch.randn(2, 4, 512)
        self.do_acc_test(f, x)
        self.assertEqual(1, metrics.generated_kernel_count)

    @inductor_config.patch(split_reductions=False)
    def test_different_reduction_order(self):
        """
        We should not reorder loops in this case. Since reordering loops does
        not help!
        """

        def f(x):
            return x.sum(dim=0), x.sum(dim=1)

        x = torch.randn(1024, 2048)
        self.do_acc_test(f, x)
        self.assertEqual(2, metrics.generated_kernel_count)
        self.assertEqual(0, metrics.num_loop_reordering)

    def test_keep_fake_dep(self):
        """
        In this model, there are fake dependencies (StarDep) between Scatter
        and a following mutation kernel that computes the gradients of
        the embedding tables.

        When we do loop reordering for the mutation kernel, we re-analyze
        the node's dependencies. But the analysis result does not contains
        those fake dependencies. Have to add them back manually.
        """
        V = 2048
        hidden_size = 64
        max_seqlen = 512
        batch_size = 8

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.word_embeddings = nn.Embedding(V, hidden_size)
                self.position_embeddings = nn.Embedding(max_seqlen, hidden_size)
                self.layer_norm = nn.LayerNorm(hidden_size)

            def forward(self, input_ids, labels, position_ids):
                emb = self.word_embeddings(input_ids) + self.position_embeddings(
                    position_ids
                )
                return self.layer_norm(emb)

        m = Model()

        @torch.compile
        def f(*args):
            m(*args).sum().backward()

        input_ids = torch.randint(0, V, (batch_size, max_seqlen))
        labels = torch.randint(0, V, (batch_size, max_seqlen))
        position_ids = torch.arange(max_seqlen)[None, :]
        # Make sure this line does not raise exceptions. If we miss
        # fake dependencies after loop reordering, we may get exception that
        # some buffer is used before being defined.
        f(input_ids, labels, position_ids)

    def test_different_broadcast_shapes(self):
        def f(x, y, c):
            return x + c, y + c

        x = torch.randn(4, 256, 1024)
        y = torch.randn(2, 512, 1024)
        c = torch.randn(1024)
        self.do_acc_test(f, x, y, c)

        # The two kernels are not fused due to c is broadcasted
        self.assertEqual(2, metrics.generated_kernel_count)

    def test_view(self):
        """
        Passing this test relies that we compare normalized MemoryDep.
        Normlaization here means merging contiguous loops.

        To make loop reordering work, we don't merge loops when creating
        SchedulerNode. Thus we need explicitly normalize MemoryDep when
        we check if two MemeoryDep matches.
        """

        def f(x):
            y = x.sin()
            x = realize(x.view(10, 10))
            return x, y

        x = torch.randn(100)
        self.do_acc_test(f, x)
        self.assertEqual(1, metrics.generated_kernel_count)


if __name__ == "__main__":
    if HAS_CUDA:
        run_tests()
