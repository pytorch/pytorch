# Owner(s): ["module: functorch"]
"""End-to-end test for pytorch/pytorch#180614 (fix option 2: reconstruct).

Inductor must wait an AsyncCollectiveTensor nested inside DTensor._local_tensor.
A correct fix must (a) call wait_tensor at runtime AND (b) not crash Dynamo
guard construction.
"""

import os
import unittest

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


@unittest.skipIf(not torch.distributed.is_available(), "torch.distributed required")
class TestNestedActEndToEnd(TestCase):
    def test_inductor_waits_nested_act_in_dtensor(self):
        import torch.distributed as dist
        import torch.distributed._functional_collectives as fc
        from torch.distributed._functional_collectives import AsyncCollectiveTensor
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import DTensor, Replicate

        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29552")
        dist.init_process_group("gloo", rank=0, world_size=1)
        try:
            wait_tensor_calls: list[str] = []
            orig_wait_tensor = fc.wait_tensor

            def counting_wait_tensor(t):
                wait_tensor_calls.append("wait_tensor")
                return orig_wait_tensor(t)

            fc.wait_tensor = counting_wait_tensor

            mesh = init_device_mesh("cpu", (1,))

            @torch.compile(backend="inductor", fullgraph=True)
            def fn(x):
                return x + 1

            # Compile + warmup, then isolate runtime.
            dt = DTensor.from_local(torch.randn(4, 4), mesh, [Replicate()])
            dt._local_tensor = AsyncCollectiveTensor(dt._local_tensor.clone())
            fn(dt)

            wait_tensor_calls.clear()

            dt2 = DTensor.from_local(torch.randn(4, 4), mesh, [Replicate()])
            dt2._local_tensor = AsyncCollectiveTensor(dt2._local_tensor.clone())
            self.assertIsInstance(dt2._local_tensor, AsyncCollectiveTensor)
            self.assertFalse(dt2._local_tensor.completed)

            out = fn(dt2)

            print(f"\nruntime wait_tensor calls: {len(wait_tensor_calls)}")
            # After a correct run the runtime wrapper resolves the nested ACT,
            # so _local_tensor is now a plain (waited) tensor.
            resolved = not isinstance(dt2._local_tensor, AsyncCollectiveTensor)
            print(f"nested ACT resolved to plain tensor: {resolved}")

            self.assertGreaterEqual(
                len(wait_tensor_calls),
                1,
                "BUG #180614: Inductor ran the compiled kernel without calling "
                "wait_tensor on the nested AsyncCollectiveTensor.",
            )
            self.assertTrue(torch.isfinite(out.to_local()).all())
        finally:
            fc.wait_tensor = orig_wait_tensor
            dist.destroy_process_group()


if __name__ == "__main__":
    run_tests()
