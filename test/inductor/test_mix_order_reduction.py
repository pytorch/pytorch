# Owner(s): ["module: inductor"]

import torch
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.inductor_utils import HAS_GPU, GPU_TYPE
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
import torch._inductor.config as inductor_config
from torch._inductor import metrics
from torch._dynamo.utils import same

@instantiate_parametrized_tests
class MixOrderReductionTest(TestCase):
    def setUp(self):
        super().setUp()
        metrics.reset()

    @parametrize("swap", (False, True))
    @inductor_config.patch(split_reductions=False)
    def test_mix_order_sum(self, swap):
        def f(x):
            if swap:
                return x.sum(dim=0), x.sum(dim=1)
            else:
                return x.sum(dim=1), x.sum(dim=0)

        M, N = 32768, 768
        dtype = torch.float
        x = torch.randn(M, N, dtype=dtype, device=GPU_TYPE)
        
        opt_f = torch.compile(f)
        
        ref = f(x)
        act = opt_f(x)

        self.assertTrue(same(ref, act, tol=1e-3), f"ref:\n{ref}\nact:\n{act}")
        self.assertEqual(1, metrics.generated_kernel_count)

if __name__ == "__main__":
    if HAS_GPU:
        run_tests()
