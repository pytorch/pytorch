# Owner(s): ["module: inductor"]

import math

import torch
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.fx_passes.sorted_segment_scatter import (
    sorted_segment_index_add,
    sorted_segment_scatter_pass,
)
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import (
    DeterministicGuard,
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_CUDA_AND_TRITON


aten = torch.ops.aten


def _index_put(base, idx, src):
    return aten.index_put.default(base, [idx], src, True)


class TestSortedSegmentScatter(TestCase):
    hw_classification = HardwareClassification.CUDA

    def _indices(self, kind, m, n):
        if kind == "uniform":
            return torch.randint(0, n, (m,), device=GPU_TYPE)
        if kind == "zeros":
            return torch.zeros(m, dtype=torch.long, device=GPU_TYPE)
        if kind == "negative":
            return torch.randint(-n, 0, (m,), device=GPU_TYPE)
        return torch.sort(torch.randint(0, n, (m,), device=GPU_TYPE)).values

    @parametrize("dtype", [torch.float32, torch.bfloat16])
    @parametrize("kind", ["uniform", "zeros", "negative", "sorted"])
    @parametrize("shape", [(1025, 64), (100003, 96), (65536, 300), (32768, 2, 128)])
    def test_segment_index_add(self, dtype, kind, shape):
        m, trailing = shape[0], shape[1:]
        n = 501
        width = 1
        for t in trailing:
            width *= t
        idx = self._indices(kind, m, n)
        src = torch.randn(m, *trailing, device=GPU_TYPE, dtype=dtype)
        out = sorted_segment_index_add(src.reshape(m, width), idx, n).view(n, *trailing)
        ref = torch.zeros(n, *trailing, device=GPU_TYPE, dtype=torch.float64)
        ref.index_put_((idx,), src.double(), accumulate=True)
        # relaxed atomics run in a varying order; fp32 rounding grows with run length
        atol = max(1e-4, 1e-6 * math.sqrt(m))
        torch.testing.assert_close(out.double(), ref, rtol=1e-5, atol=atol)

    def test_masked_tail_does_not_write_before_out(self):
        # fully masked tail blocks must not flush to out[-1]
        for m in (1025, 2**16 + 5):
            idx = torch.randint(0, 100, (m,), device=GPU_TYPE)
            src = torch.randn(m, 64, device=GPU_TYPE)
            big = torch.full((101, 64), -0.0, device=GPU_TYPE)
            out = big[1:]
            from torch._inductor.fx_passes.sorted_segment_scatter import (
                _sorted_segment_sum_kernel,
            )

            skey, perm = torch.sort(idx.int())
            _sorted_segment_sum_kernel[((m + 1023) // 1024, 1)](
                src, perm, skey, out, m, 64, 32, 64, 32, num_warps=2
            )
            self.assertTrue(torch.signbit(big[0]).all())

    def _rewrites(self, base, idx, src, min_index_size=1000):
        gm = make_fx(_index_put, tracing_mode="fake")(base, idx, src)
        with config.patch(sorted_segment_scatter_min_index_size=min_index_size):
            n = sorted_segment_scatter_pass(gm.graph)
        for node in gm.graph.nodes:
            if node.op == "call_function":
                self.assertIn("val", node.meta, str(node))
        return n

    def test_pass_gates(self):
        m, n = 4096, 501
        idx = torch.randint(0, n, (m,), device=GPU_TYPE)
        mk = lambda dt: (  # noqa: E731
            torch.zeros(n, 64, device=GPU_TYPE, dtype=dt),
            idx,
            torch.randn(m, 64, device=GPU_TYPE, dtype=dt),
        )
        self.assertEqual(self._rewrites(*mk(torch.float32)), 1)
        self.assertEqual(self._rewrites(*mk(torch.float16)), 1)
        self.assertEqual(self._rewrites(*mk(torch.float64)), 0)
        base, _, src = mk(torch.float32)
        self.assertEqual(self._rewrites(base, idx.cpu(), src), 0)
        # narrow rows, large tables and short indices are left to the default lowering
        self.assertEqual(
            self._rewrites(base[:, :32], idx, torch.randn(m, 32, device=GPU_TYPE)), 0
        )
        big = torch.zeros(65537, 64, device=GPU_TYPE)
        self.assertEqual(self._rewrites(big[:65536], idx, src), 1)
        self.assertEqual(self._rewrites(big, idx, src), 0)
        self.assertEqual(self._rewrites(base, idx, src, min_index_size=m), 1)
        self.assertEqual(self._rewrites(base, idx, src, min_index_size=m + 1), 0)
        # a [1, D] source broadcasts over the index
        self.assertEqual(self._rewrites(base, idx, src[:1]), 0)
        with DeterministicGuard(True):
            self.assertEqual(self._rewrites(*mk(torch.float32)), 0)
        with config.patch(cpp_wrapper=True):
            self.assertEqual(self._rewrites(*mk(torch.float32)), 0)
        with config.patch(fx_wrapper=True):
            self.assertEqual(self._rewrites(*mk(torch.float32)), 0)

    @parametrize("dtype", [torch.float32, torch.bfloat16])
    @config.patch(
        sorted_segment_scatter_enabled=True, sorted_segment_scatter_min_index_size=1000
    )
    def test_compile_matches_eager(self, dtype):
        def fn(base, idx, src):
            return base.index_put((idx,), src, accumulate=True) * 2

        m, n = 70001, 501
        base = torch.randn(n, 256, device=GPU_TYPE, dtype=dtype)
        idx = torch.randint(0, n, (m,), device=GPU_TYPE)
        src = torch.randn(m, 256, device=GPU_TYPE, dtype=dtype)
        torch._dynamo.reset()
        counters.clear()
        out, code = run_and_get_code(torch.compile(fn), base, idx, src)
        self.assertEqual(counters["inductor"]["sorted_segment_scatter_applied"], 1)
        self.assertIn("sorted_segment_index_add", code[0])
        ref = (base.double().index_put((idx,), src.double(), accumulate=True) * 2).to(
            dtype
        )
        tol = 1e-4 if dtype == torch.float32 else 2e-2
        torch.testing.assert_close(out, ref, rtol=tol, atol=tol)


instantiate_parametrized_tests(TestSortedSegmentScatter)

if __name__ == "__main__":
    if HAS_CUDA_AND_TRITON:
        run_tests()
