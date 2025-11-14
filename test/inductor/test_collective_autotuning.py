# Owner(s): ["module: inductor"]

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

    @skip_if_lt_x_gpu(4)
    def test_all_gather_4ranks(self):
        """Test all_gather with 4 ranks"""

        # Initialize distributed
        dist.init_process_group(
            backend="nccl",
            init_method=f"file:///tmp/test_collective_allgather_4ranks_{self.id()}",
            world_size=self.world_size,
            rank=self.rank,
        )

        rank = dist.get_rank()
        device = f"cuda:{rank}"

        # Register the default process group
        from torch._C._distributed_c10d import _register_process_group

        _register_process_group("default", dist.group.WORLD)

        # Define custom all_gather op
        @torch.library.custom_op("test::my_allgather_4ranks", mutates_args=())
        def my_allgather(x: torch.Tensor) -> torch.Tensor:
            output = torch.ops._c10d_functional.all_gather_into_tensor(
                x.contiguous(), 4, "default"
            )
            return output

        # Fake implementation
        @my_allgather.register_fake
        def _(x):
            return torch.empty(
                x.size(0) * 4, *x.size()[1:], dtype=x.dtype, device=x.device
            )

        # Implementation function
        def allgather_impl(x):
            output = torch.ops._c10d_functional.all_gather_into_tensor(
                x.contiguous(), 4, "default"
            )
            return output

        # Register autotuning
        from torch._inductor.kernel.custom_op import (
            CustomOpConfig,
            register_custom_op_autotuning,
        )

        register_custom_op_autotuning(
            my_allgather,
            configs=[
                CustomOpConfig(allgather_impl),
            ],
        )

        # Test model
        class GatherModel(torch.nn.Module):
            def forward(self, x):
                return my_allgather(x)

        model = torch.compile(GatherModel()).to(device)

        # Run
        x = torch.ones(32, 64, device=device) * (rank + 1)
        y = model(x)

        # Verify shape
        expected_shape = (32 * 4, 64)
        self.assertEqual(y.shape, expected_shape)

        if rank == 0:
            print(
                f"[4 ranks] All-gather test passed! Output shape: {y.shape}, World size: {self.world_size}"
            )

        dist.destroy_process_group()

    @skip_if_lt_x_gpu(2)
    def test_equivalent_allreduce_strategies(self):
        """
        Test autotuning between mathematically equivalent all_reduce strategies.

        Strategy 1: sum all_reduce
        Strategy 2: avg all_reduce * world_size

        Both compute sum(x_i) but may have different performance characteristics.
        """
        dist.init_process_group(
            backend="nccl",
            init_method=f"file:///tmp/test_equiv_ar_{self.id()}",
            world_size=self.world_size,
            rank=self.rank,
        )

        dist.barrier()

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = f"cuda:{rank}"

        from torch._C._distributed_c10d import _register_process_group

        _register_process_group("default", dist.group.WORLD)

        @torch.library.custom_op("test::equiv_ar", mutates_args=())
        def equiv_ar(x: torch.Tensor) -> torch.Tensor:
            result = x.clone()
            return torch.ops._c10d_functional.all_reduce_(result, "sum", "default")

        @equiv_ar.register_fake
        def _(x):
            return torch.empty_like(x)

        def sum_allreduce(x: torch.Tensor) -> torch.Tensor:
            result = x.clone()
            return torch.ops._c10d_functional.all_reduce_(result, "sum", "default")

        def avg_allreduce_scaled(x: torch.Tensor) -> torch.Tensor:
            result = x.clone()
            result = torch.ops._c10d_functional.all_reduce_(result, "avg", "default")
            return result * world_size

        from torch._inductor.kernel.custom_op import (
            CustomOpConfig,
            register_custom_op_autotuning,
        )

        register_custom_op_autotuning(
            equiv_ar,
            configs=[
                CustomOpConfig(sum_allreduce),
                CustomOpConfig(avg_allreduce_scaled),
            ],
        )

        class EquivAllReduceModel(torch.nn.Module):
            def forward(self, x):
                return equiv_ar(x)

        model = torch.compile(EquivAllReduceModel()).to(device)

        torch.manual_seed(42)
        x = torch.randn(128, 128, device=device)
        dist.broadcast(x, src=0)

        _ = model(x)

        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    run_tests()
