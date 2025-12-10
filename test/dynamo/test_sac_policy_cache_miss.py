"""
Test to verify that SAC (Selective Activation Checkpointing) policy changes
result in cache misses when caching is enabled.

This test verifies that:
1. Different SAC policies produce different cache keys (cache miss)
2. Same SAC policies produce cache hits
"""

from functools import partial

import torch
import torch._dynamo
import torch._dynamo.test_case
from torch._dynamo.utils import counters
from torch._functorch import config as functorch_config
from torch._inductor import config as inductor_config
from torch._inductor.utils import fresh_cache
from torch.testing._internal.common_utils import run_tests
from torch.utils.checkpoint import checkpoint, CheckpointPolicy



def policy_save_mm(ctx, op, *args, **kwargs):
    """Policy that saves mm and recomputes everything else"""
    if op == torch.ops.aten.mm.default:
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.MUST_RECOMPUTE


def policy_save_add(ctx, op, *args, **kwargs):
    """Policy that saves add and recomputes everything else"""
    if op == torch.ops.aten.add.Tensor:
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.MUST_RECOMPUTE


def gn(x, y):
    """Function to checkpoint"""
    a = torch.mm(x, y)
    b = torch.add(a, x)
    return b


class TestSACPolicyCacheMiss(torch._dynamo.test_case.TestCase):
    def setUp(self):
        super().setUp()
        counters.clear()
        torch._dynamo.reset()

    def tearDown(self):
        super().tearDown()
        counters.clear()
        torch._dynamo.reset()

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch(
        {"enable_autograd_cache": True, "strict_autograd_cache": True}
    )
    def test_different_sac_policies_cause_cache_miss(self):
        ctx_fn_save_mm = partial(
            torch.utils.checkpoint.create_selective_checkpoint_contexts, policy_save_mm
        )

        ctx_fn_save_add = partial(
            torch.utils.checkpoint.create_selective_checkpoint_contexts, policy_save_add
        )

        @torch.compile(backend="inductor")
        def fn_with_checkpoint(x, y, ctx_fn):
            return checkpoint(gn, x, y, use_reentrant=False, context_fn=ctx_fn)

        x = torch.randn(4, 4)
        y = torch.randn(4, 4)

        with fresh_cache():
            # First run with policy that saves mm
            fn_with_checkpoint(x, y, ctx_fn_save_mm)

            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)

            # Clear dynamo but keep cache
            torch._dynamo.reset()

            # Second run with different policy (saves add instead of mm)
            fn_with_checkpoint(x, y, ctx_fn_save_add)

            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch(
        {"enable_autograd_cache": True, "strict_autograd_cache": True}
    )
    def test_same_sac_policy_causes_cache_hit(self):
        ctx_fn = partial(
            torch.utils.checkpoint.create_selective_checkpoint_contexts, policy_save_mm
        )

        @torch.compile(backend="inductor")
        def fn_with_checkpoint(x, y):
            return checkpoint(gn, x, y, use_reentrant=False, context_fn=ctx_fn)

        x = torch.randn(4, 4)
        y = torch.randn(4, 4)

        with fresh_cache():
            # First run
            fn_with_checkpoint(x, y)

            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)

            # Clear dynamo but keep cache
            torch._dynamo.reset()

            # Second run with same policy - should hit cache
            fn_with_checkpoint(x, y)

            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)


if __name__ == "__main__":
    run_tests()
