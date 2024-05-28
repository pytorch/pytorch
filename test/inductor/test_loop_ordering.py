# Owner(s): ["module: inductor"]
import contextlib

import torch
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

        expect = f(x, y)
        actual = torch.compile(f)(x, y)
        self.assertTrue(torch.allclose(expect, actual, atol=1e-3, rtol=1e-3))
        self.assertEqual(1, metrics.generated_kernel_count)


if __name__ == "__main__":
    if HAS_CUDA:
        torch.set_default_device("cuda")
        run_tests()
