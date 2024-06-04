# Owner(s): ["module: inductor"]

import contextlib

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

    @staticmethod
    def _create_computed_buffer():
        box_a = ir.TensorBox.create(
            ir.Buffer(
                "a",
                ir.FixedLayout(torch.device("cuda"), torch.float32, [32, 64], [64, 1]),
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
        return buf.data.data

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

    def test_reorder_twice(self):
        """
        This may happen in practice if we pick a order when fusing A and B.
        Then we pick another order for AB when we fusion C into it.

        E.g. happens for BertForMaskedLM.
        """
        buf = self._create_computed_buffer()
        snode = SchedulerNode(V.graph.scheduler, buf)
        prefix0 = self._get_snode_body_sym_prefix(snode)
        snode.apply_new_loop_order([1, 0])
        prefix1 = self._get_snode_body_sym_prefix(snode)
        self.assertEqual(prefix0, prefix1, f"{prefix0} v.s. {prefix1}")
        snode.apply_new_loop_order([1, 0])
        prefix2 = self._get_snode_body_sym_prefix(snode)
        self.assertEqual(prefix0, prefix2, f"{prefix0} v.s. {prefix2}")


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
