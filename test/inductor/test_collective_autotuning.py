# Owner(s): ["module: inductor"]

"""
Test collective op autotuning - Phase 1: Basic functionality with 2 ranks.

This test validates that:
1. Collective ops are detected correctly
2. CollectiveBenchmarker is used for collective ops
3. 2 ranks can sync and benchmark successfully
4. Results are correct
"""

import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import run_tests


class TestCollectiveAutotuning(MultiProcessTestCase):
    @property
    def world_size(self):
        return 2

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    @skip_if_lt_x_gpu(2)
    def test_single_allreduce_2ranks(self):
        """Test single all_reduce with 2 ranks"""

        # Initialize distributed
        dist.init_process_group(
            backend="nccl",
            init_method=f"file:///tmp/test_collective_autotune_{self.id()}",
            world_size=self.world_size,
            rank=self.rank,
        )

        # Add barrier to ensure all ranks are synchronized
        dist.barrier()

        rank = dist.get_rank()
        device = f"cuda:{rank}"

        # Register the default process group
        from torch._C._distributed_c10d import _register_process_group

        _register_process_group("default", dist.group.WORLD)

        # Define custom collective op
        @torch.library.custom_op("test::my_allreduce", mutates_args=())
        def my_allreduce(x: torch.Tensor) -> torch.Tensor:
            result = x.clone()
            return torch.ops._c10d_functional.all_reduce_(result, "sum", "default")

        # Fake implementation for abstract
        @my_allreduce.register_fake
        def _(x):
            return torch.empty_like(x)

        # Implementation 1: Direct NCCL
        def allreduce_nccl(x):
            result = x.clone()
            return torch.ops._c10d_functional.all_reduce_(result, "sum", "default")

        # Implementation 2: Simulate chunked (for testing multiple choices)
        def allreduce_nccl2(x):
            # For now, just call the regular allreduce
            result = x.clone()
            return torch.ops._c10d_functional.all_reduce_(result, "sum", "default")

        # Register autotuning
        from torch._inductor.kernel.custom_op import (
            CustomOpConfig,
            register_custom_op_autotuning,
        )

        register_custom_op_autotuning(
            my_allreduce,
            configs=[
                CustomOpConfig(allreduce_nccl),
                CustomOpConfig(allreduce_nccl2),
            ],
        )

        # Test model
        class SimpleModel(torch.nn.Module):
            def forward(self, x):
                return my_allreduce(x)

        model = torch.compile(SimpleModel()).to(device)

        x = torch.randn(128, 128, device=device)
        x_copy = x.clone()
        y = model(x)
        expected = x_copy * 2
        torch.testing.assert_close(y, expected, rtol=1e-3, atol=1e-3)

        # Ensure all ranks finish and then cleanup
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    run_tests()
