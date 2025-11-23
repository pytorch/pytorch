"""
Test collective op autotuning with multiple ranks (4 and 8 ranks).

This test validates that:
1. Collective ops work correctly with 4 ranks
2. Collective ops work correctly with 8 ranks
3. Autotuning works correctly across more ranks
4. Results are correct
"""

import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import run_tests


class TestCollectiveAutotuning4Ranks(MultiProcessTestCase):
    """Test collective autotuning with 4 ranks"""

    @property
    def world_size(self):
        return 4

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    @skip_if_lt_x_gpu(4)
    def test_single_allreduce_4ranks(self):
        """Test single all_reduce with 4 ranks"""

        # Initialize distributed
        dist.init_process_group(
            backend="nccl",
            init_method=f"file:///tmp/test_collective_autotune_4ranks_{self.id()}",
            world_size=self.world_size,
            rank=self.rank,
        )

        rank = dist.get_rank()
        device = f"cuda:{rank}"

        # Register the default process group
        from torch._C._distributed_c10d import _register_process_group

        _register_process_group("default", dist.group.WORLD)

        # Define custom collective op
        @torch.library.custom_op("test::my_allreduce_4ranks", mutates_args=())
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
        def allreduce_chunked(x, chunk_size=1024):
            # For now, just call the regular allreduce
            result = x.clone()
            return torch.ops._c10d_functional.all_reduce_(result, "sum", "default")

        # Register autotuning
        from torch._inductor.kernel.custom_op import (
            register_custom_op_autotuning,
            CustomOpConfig,
        )

        register_custom_op_autotuning(
            my_allreduce,
            configs=[
                CustomOpConfig(allreduce_nccl),
                CustomOpConfig(allreduce_chunked, chunk_size=1024),
                CustomOpConfig(allreduce_chunked, chunk_size=2048),
            ],
        )

        # Test model
        class SimpleModel(torch.nn.Module):
            def forward(self, x):
                return my_allreduce(x)

        model = torch.compile(SimpleModel()).to(device)

        # Run
        x = torch.randn(128, 128, device=device)
        x_copy = x.clone()
        y = model(x)

        # Verify: sum across 4 ranks
        expected = x_copy * 4
        torch.testing.assert_close(y, expected, rtol=1e-3, atol=1e-3)

        if rank == 0:
            print(f"[4 ranks] Single allreduce test passed! World size: {self.world_size}")

        dist.destroy_process_group()

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
            register_custom_op_autotuning,
            CustomOpConfig,
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


class TestCollectiveAutotuning8Ranks(MultiProcessTestCase):
    """Test collective autotuning with 8 ranks"""

    @property
    def world_size(self):
        return 8

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    @skip_if_lt_x_gpu(8)
    def test_single_allreduce_8ranks(self):
        """Test single all_reduce with 8 ranks"""

        # Initialize distributed
        dist.init_process_group(
            backend="nccl",
            init_method=f"file:///tmp/test_collective_autotune_8ranks_{self.id()}",
            world_size=self.world_size,
            rank=self.rank,
        )

        rank = dist.get_rank()
        device = f"cuda:{rank}"

        # Register the default process group
        from torch._C._distributed_c10d import _register_process_group

        _register_process_group("default", dist.group.WORLD)

        # Define custom collective op
        @torch.library.custom_op("test::my_allreduce_8ranks", mutates_args=())
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

        # Implementation 2: Simulate chunked
        def allreduce_chunked(x, chunk_size=1024):
            result = x.clone()
            return torch.ops._c10d_functional.all_reduce_(result, "sum", "default")

        # Register autotuning with more configurations
        from torch._inductor.kernel.custom_op import (
            register_custom_op_autotuning,
            CustomOpConfig,
        )

        register_custom_op_autotuning(
            my_allreduce,
            configs=[
                CustomOpConfig(allreduce_nccl),
                CustomOpConfig(allreduce_chunked, chunk_size=512),
                CustomOpConfig(allreduce_chunked, chunk_size=1024),
                CustomOpConfig(allreduce_chunked, chunk_size=2048),
            ],
        )

        # Test model
        class SimpleModel(torch.nn.Module):
            def forward(self, x):
                return my_allreduce(x)

        model = torch.compile(SimpleModel()).to(device)

        # Run with larger tensor to stress-test
        x = torch.randn(256, 256, device=device)
        x_copy = x.clone()
        y = model(x)

        # Verify: sum across 8 ranks
        expected = x_copy * 8
        torch.testing.assert_close(y, expected, rtol=1e-3, atol=1e-3)

        if rank == 0:
            print(f"[8 ranks] Single allreduce test passed! World size: {self.world_size}")

        dist.destroy_process_group()

    @skip_if_lt_x_gpu(8)
    def test_reduce_scatter_8ranks(self):
        """Test reduce_scatter with 8 ranks"""

        # Initialize distributed
        dist.init_process_group(
            backend="nccl",
            init_method=f"file:///tmp/test_collective_rs_8ranks_{self.id()}",
            world_size=self.world_size,
            rank=self.rank,
        )

        rank = dist.get_rank()
        device = f"cuda:{rank}"

        # Register the default process group
        from torch._C._distributed_c10d import _register_process_group

        _register_process_group("default", dist.group.WORLD)

        # Define custom reduce_scatter op
        @torch.library.custom_op("test::my_reduce_scatter_8ranks", mutates_args=())
        def my_reduce_scatter(x: torch.Tensor) -> torch.Tensor:
            # reduce_scatter_tensor API: (input, reduce_op, group_size, group_name)
            output = torch.ops._c10d_functional.reduce_scatter_tensor(
                x.contiguous(), "sum", 8, "default"
            )
            return output

        # Fake implementation
        @my_reduce_scatter.register_fake
        def _(x):
            output_size = x.size(0) // 8
            return torch.empty(
                output_size, *x.size()[1:], dtype=x.dtype, device=x.device
            )

        # Implementation function
        def reduce_scatter_impl(x):
            output = torch.ops._c10d_functional.reduce_scatter_tensor(
                x.contiguous(), "sum", 8, "default"
            )
            return output

        # Register autotuning
        from torch._inductor.kernel.custom_op import (
            register_custom_op_autotuning,
            CustomOpConfig,
        )

        register_custom_op_autotuning(
            my_reduce_scatter,
            configs=[
                CustomOpConfig(reduce_scatter_impl),
            ],
        )

        # Test model
        class ReduceScatterModel(torch.nn.Module):
            def forward(self, x):
                return my_reduce_scatter(x)

        model = torch.compile(ReduceScatterModel()).to(device)

        # Run
        x = torch.ones(256, 64, device=device)  # 256 = 32 * 8
        y = model(x)

        # Verify shape
        expected_shape = (32, 64)
        self.assertEqual(y.shape, expected_shape)

        # Verify values (sum of 8 ranks worth of ones)
        expected = torch.ones(32, 64, device=device) * 8
        torch.testing.assert_close(y, expected, rtol=1e-3, atol=1e-3)

        if rank == 0:
            print(
                f"[8 ranks] Reduce-scatter test passed! Output shape: {y.shape}, World size: {self.world_size}"
            )

        dist.destroy_process_group()


if __name__ == "__main__":
    run_tests()
