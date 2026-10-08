# Owner(s): ["module: inductor"]

import unittest

import torch
from torch._inductor.codegen.simd import SIMDScheduling
from torch._inductor.test_case import run_tests, TestCase


# candidate_tilings is only reached by the SIMD (triton) backend, so a CPU-only
# run would leave the cache trivially empty and assert nothing.
@unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
class TilingCacheRetentionTest(TestCase):
    """`candidate_tilings` must not outlive the graph it memoised on.

    Its lru_cache key holds a SchedulerNode, which reaches the compiled
    GraphModule through ComputedBuffer.origins -> fx Node -> Graph. A retained
    entry therefore pins every parameter of the model, long after compilation
    has finished.
    """

    def setUp(self):
        super().setUp()
        SIMDScheduling.candidate_tilings.cache_clear()
        self.addCleanup(SIMDScheduling.candidate_tilings.cache_clear)

    def test_cache_is_empty_after_compile(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 8),
        ).cuda()
        compiled = torch.compile(model)
        compiled(torch.randn(32, 64, device="cuda"))

        self.assertEqual(
            SIMDScheduling.candidate_tilings.cache_info().currsize,
            0,
            "candidate_tilings retained entries after codegen; each one pins "
            "the whole GraphModule and every parameter hanging off it",
        )


if __name__ == "__main__":
    run_tests()
