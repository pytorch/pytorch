# Owner(s): ["module: inductor"]
import contextlib
import unittest

import torch

from torch._inductor.graph import GraphLowering
from torch._inductor.ir import Buffer, FixedLayout, Pointwise
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.virtualized import ops, V

from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA


class TestDependencies(InductorTestCase):
    def _create_buffer(self, name, shape, dtype=torch.float32):
        return Buffer(name, FixedLayout(torch.device("cuda:0"), dtype, shape))

    def setUp(self):
        super().setUp()

        class DummyModule(torch.nn.Module):
            def forward(self, x):
                return x * 2

        self._gm = torch.fx.symbolic_trace(DummyModule())
        self._graph = GraphLowering(self._gm)

        self._stack = contextlib.ExitStack()
        self._stack.enter_context(V.set_graph_handler(self._graph))

    def tearDown(self):
        self._stack.close()
        super().tearDown()

    @unittest.skipIf(not HAS_CUDA, "CUDA-only test")
    def test_bucketize_dependencies(self):
        offsets = self._create_buffer("offsets", (1025,), torch.int32)

        def inner_fn(index):
            idx = index[0]
            return ops.bucketize(
                values=idx,
                offsets_name=offsets.get_name(),
                offsets_size=offsets.get_size()[0],
                indexing_dtype=torch.int32,
                right=True,
            )

        pointwise = Pointwise.create(
            device=torch.device("cuda:0"),
            dtype=torch.int32,
            inner_fn=inner_fn,
            ranges=[1024 * 4],
        )

        self.assertEqual(len(pointwise.get_reads()), 1)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_CPU or HAS_CUDA:
        run_tests("sympy")
