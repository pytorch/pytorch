# Owner(s): ["module: inductor"]

import sys

import torch
import torch.distributed as dist


if not dist.is_available() or not dist.is_nccl_available():
    print("c10d NCCL not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import run_tests


class TestCollectiveAutotuning2Ranks(MultiProcessTestCase):
    """Test collective autotuning with 2 ranks"""

    @property
    def world_size(self):
        return 2

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    @skip_if_lt_x_gpu(2)
    def test_equivalent_allreduce_strategies(self):
        """
        Test autotuning between mathematically equivalent all_reduce strategies.

        Strategy 1: sum all_reduce
        Strategy 2: avg all_reduce * world_size
        """
        dist.init_process_group(
            backend="nccl",
            init_method=f"file:///tmp/test_equiv_allreduce_{self.id()}",
            world_size=self.world_size,
            rank=self.rank,
        )

        dist.barrier()

        rank = dist.get_rank()
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
            return result * self.world_size

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


class TestCollectiveAutotuning4Ranks(MultiProcessTestCase):
    """Test collective autotuning with 4 ranks"""

    @property
    def world_size(self):
        return 4

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    @skip_if_lt_x_gpu(4)
    def test_vllm_style_allreduce(self):
        """
        Test vLLM-style custom allreduce with buffer copy pattern.

        vLLM uses custom allreduce optimized for small tensors (<8MB).
        Two implementations simulate vLLM's registered=False mode vs standard NCCL.
        """
        dist.init_process_group(
            backend="nccl",
            init_method=f"file:///tmp/test_vllm_allreduce_{self.id()}",
            world_size=self.world_size,
            rank=self.rank,
        )

        dist.barrier()

        rank = dist.get_rank()
        device = f"cuda:{rank}"

        from torch._C._distributed_c10d import _register_process_group

        _register_process_group("default", dist.group.WORLD)

        @torch.library.custom_op("test::vllm_allreduce", mutates_args=())
        def vllm_allreduce(x: torch.Tensor) -> torch.Tensor:
            result = x.clone()
            return torch.ops._c10d_functional.all_reduce_(result, "sum", "default")

        @vllm_allreduce.register_fake
        def _(x):
            return torch.empty_like(x)

        def vllm_buffer_copy_allreduce(x: torch.Tensor) -> torch.Tensor:
            """
            vLLM registered=False: flatten -> copy to IPC buffer -> allreduce -> reshape

            vLLM code:
                inp_size = inp.numel() * inp.element_size()
                self.buffer_ptrs[self.rank][:inp_size].copy_(inp.view(-1))
                ops.all_reduce(self._ptr, inp, out, self.buffer_ptrs[self.rank], self.max_size)
            """
            original_shape = x.shape
            flat_x = x.contiguous().view(-1)
            buffer_copy = flat_x.clone()
            result = torch.ops._c10d_functional.all_reduce_(
                buffer_copy, "sum", "default"
            )
            return result.view(original_shape)

        def nccl_allreduce_direct(x: torch.Tensor) -> torch.Tensor:
            """Standard NCCL allreduce without buffer copy."""
            result = x.clone()
            return torch.ops._c10d_functional.all_reduce_(result, "sum", "default")

        from torch._inductor.kernel.custom_op import (
            CustomOpConfig,
            register_custom_op_autotuning,
        )

        register_custom_op_autotuning(
            vllm_allreduce,
            configs=[
                CustomOpConfig(vllm_buffer_copy_allreduce),
                CustomOpConfig(nccl_allreduce_direct),
            ],
        )

        class VLLMAllReduceModel(torch.nn.Module):
            def forward(self, x):
                return vllm_allreduce(x)

        model = torch.compile(VLLMAllReduceModel()).to(device)

        torch.manual_seed(42 + rank)
        x = torch.randn(128, 256, device=device)

        y = model(x)
        self.assertEqual(y.shape, x.shape)
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    run_tests()
