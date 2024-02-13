# Owner(s): ["oncall: distributed"]

from typing import Callable

import torch
import torch.distributed as dist
import torch.nn as nn

from torch.distributed._composable.fsdp import fully_shard
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    FSDPTest,
    patch_all_gather,
    patch_reduce_scatter,
)
from torch.testing._internal.common_utils import get_cycles_per_ms, run_tests


class TestFullyShardOverlap(FSDPTest):
    @property
    def world_size(self) -> int:
        return min(2, torch.cuda.device_count())

    @skip_if_lt_x_gpu(2)
    def test_fully_shard_training_overlap(self):
        class LinearWithSleep(nn.Module):
            def __init__(self, dim: int, sleep_ms: int):
                super().__init__()
                self.weight = nn.Parameter(torch.randn((dim, dim)))
                self.sleep_ms = sleep_ms

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return nn.functional.relu(Matmul.apply(x, self.weight, self.sleep_ms))

        torch.manual_seed(42)

        # Use non-trivial comm. time but still shorter than compute time
        dim, num_linears, compute_sleep_ms, comm_sleep_ms = (4, 3, 25, 10)
        model = nn.Sequential(
            *[LinearWithSleep(dim, compute_sleep_ms) for _ in range(num_linears)]
        )
        for lin in model:
            fully_shard(lin, reshard_after_forward=True)
        fully_shard(model, reshard_after_forward=True)

        orig_all_gather_into_tensor = dist.all_gather_into_tensor
        orig_reduce_scatter = dist.reduce_scatter_tensor
        comm_stream = torch.cuda.Stream()

        def delay_collective():
            # Share a stream so that all-gather and reduce-scatter block each
            # other like in `ProcessGroupNCCL`
            comm_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(comm_stream):
                torch.cuda._sleep(int(comm_sleep_ms * get_cycles_per_ms()))
            torch.cuda.current_stream().wait_stream(comm_stream)

        def delayed_all_gather(*args, **kwargs):
            delay_collective()
            return orig_all_gather_into_tensor(*args, **kwargs)

        def delayed_reduce_scatter(*args, **kwargs):
            delay_collective()
            return orig_reduce_scatter(*args, **kwargs)

        inp = torch.randn((2, dim), device="cuda")
        loss = model(inp).sum()  # warmup CUDA and allocator
        loss.backward()

        def fwd():
            with patch_all_gather(delayed_all_gather):
                model(inp)

        fwd_time = self._time_fn(fwd)
        buffer_ms = 2  # CPU delays and copies
        expected_fwd_time = comm_sleep_ms + num_linears * compute_sleep_ms + buffer_ms
        # Forward: only 1st all-gather is exposed
        self.assertLessEqual(fwd_time, expected_fwd_time)

        def fwd_bwd():
            with patch_all_gather(delayed_all_gather), patch_reduce_scatter(
                delayed_reduce_scatter
            ):
                loss = model(inp).sum()
                loss.backward()

        fwd_bwd_time = self._time_fn(fwd_bwd)
        # Backward: only 1st all-gather and last reduce-scatter are exposed;
        # double the backward compute since computing two gradients per layer
        expected_bwd_time = (
            comm_sleep_ms * 2 + num_linears * 2 * compute_sleep_ms + buffer_ms * 2
        )
        self.assertLessEqual(fwd_bwd_time, expected_fwd_time + expected_bwd_time)

    def _time_fn(self, fn: Callable):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        dist.barrier()
        torch.cuda.synchronize()
        start_event.record()
        fn()
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        return elapsed_time


class Matmul(torch.autograd.Function):
    # Use CUDA sleeps to emulate compute time
    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor, sleep_ms: int):
        ctx.save_for_backward(input, weight)
        ctx.sleep_ms = sleep_ms
        torch.cuda._sleep(int(sleep_ms * get_cycles_per_ms()))
        return input @ weight

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (input, weight) = ctx.saved_tensors
        torch.cuda._sleep(int(2 * ctx.sleep_ms * get_cycles_per_ms()))
        grad_input = grad_output @ weight.T
        grad_weight = input.T @ grad_output
        return grad_input, grad_weight, None


if __name__ == "__main__":
    run_tests()
