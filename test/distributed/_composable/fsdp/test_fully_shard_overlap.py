# Owner(s): ["oncall: distributed"]

import copy
import functools
import unittest
from typing import Callable

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor.experimental import implicit_replication
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    FSDPTest,
    get_devtype,
    patch_all_gather,
    patch_reduce_scatter,
)
from torch.testing._internal.common_utils import get_cycles_per_ms, run_tests, TEST_HPU


device_type = torch.device(get_devtype())
device_module = torch.get_device_module(device_type)


class TestFullyShardOverlap(FSDPTest):
    """
    NOTE: Testing stream overlap in PyTorch CI is tricky.

    One approach is to use CUDA sleeps to emulate kernels in each stream;
    however, ``torch.cuda._sleep`` requires inputs in units of cycles. The
    ``get_cycles_per_ms`` function to convert from ms to cycles is computed
    once and cached thereafter, which means that if there is variation later,
    the cached value may not be accurate. This leads to flakiness in CI.

    To address this, we relax the tests as simple sanity checks that the
    overlapped times are less than a non-overlapped baseline, but we do not
    test that the overlapped time is less than a precisely calculated time.
    """

    @property
    def world_size(self) -> int:
        return min(2, torch.get_device_module(device_type).device_count())

    @skip_if_lt_x_gpu(2)
    @unittest.skipIf(TEST_HPU, "Sleep is not supported on HPU")
    def test_fully_shard_training_overlap(self):
        torch.manual_seed(42)

        # Use non-trivial comm. time but still shorter than compute time
        dim, num_linears, compute_sleep_ms, comm_sleep_ms = (4, 3, 25, 10)
        model = nn.Sequential(
            *[LinearWithSleep(dim, compute_sleep_ms) for _ in range(num_linears)]
        )
        ref_model = copy.deepcopy(model).to(device_type)
        for lin in model:
            assert len(list(lin.parameters())) == 1, "Expects only one weight"
            fully_shard(lin, reshard_after_forward=True)
        fully_shard(model, reshard_after_forward=True)

        orig_all_gather_into_tensor = dist.all_gather_into_tensor
        orig_reduce_scatter_tensor = dist.reduce_scatter_tensor
        comm_stream = torch.get_device_module(device_type).Stream()

        def delay_collective():
            # Share a stream so that all-gather and reduce-scatter block each
            # other like in `ProcessGroupNCCL`
            comm_stream.wait_stream(
                torch.get_device_module(device_type).current_stream()
            )
            with torch.get_device_module(device_type).stream(comm_stream):
                torch.get_device_module(device_type)._sleep(
                    int(comm_sleep_ms * get_cycles_per_ms())
                )
            torch.get_device_module(device_type).current_stream().wait_stream(
                comm_stream
            )

        def delayed_all_gather(*args, **kwargs):
            delay_collective()
            return orig_all_gather_into_tensor(*args, **kwargs)

        def delayed_reduce_scatter(*args, **kwargs):
            delay_collective()
            return orig_reduce_scatter_tensor(*args, **kwargs)

        inp = torch.randn((2, dim), device=device_type.type)
        loss = model(inp).sum()  # warmup CUDA and allocator
        loss.backward()

        def ref_fwd():
            with patch_all_gather(delayed_all_gather):
                # Run dummy all-gathers per weight (which is one FSDP group)
                for lin in ref_model:
                    dummy_ag_output = torch.empty_like(lin.weight)
                    dummy_ag_input = torch.chunk(dummy_ag_output, self.world_size)[
                        self.rank
                    ]
                    dist.all_gather_into_tensor(dummy_ag_output, dummy_ag_input)
                return ref_model(inp)

        def fwd():
            with patch_all_gather(delayed_all_gather):
                model(inp)

        ref_fwd_time = self._time_fn(ref_fwd)
        fwd_time = self._time_fn(fwd)
        # Forward: only 1st all-gather is exposed
        # NOTE: Do not enforce the expected forward time due to flakiness in CI
        # expected_fwd_time = comm_sleep_ms + num_linears * compute_sleep_ms + buffer_ms
        self.assertLessEqual(fwd_time, ref_fwd_time)

        def ref_fwd_bwd():
            with patch_all_gather(delayed_all_gather):
                # Run dummy all-gathers per weight (which is one FSDP group)
                for lin in ref_model:
                    dummy_ag_output = torch.empty_like(lin.weight)
                    dummy_ag_input = torch.chunk(dummy_ag_output, self.world_size)[
                        self.rank
                    ]
                    dist.all_gather_into_tensor(dummy_ag_output, dummy_ag_input)
                loss = ref_model(inp).sum()
                # Run dummy all-gathers per weight again since we are
                # resharding after forward
                for lin in ref_model:
                    dummy_ag_output = torch.empty_like(lin.weight)
                    dummy_ag_input = torch.chunk(dummy_ag_output, self.world_size)[
                        self.rank
                    ]
                    dist.all_gather_into_tensor(dummy_ag_output, dummy_ag_input)
                loss.backward()
                # Run dummy reduce-scatters per weight
                for lin in ref_model:
                    dummy_rs_input = torch.empty_like(lin.weight)
                    dummy_rs_output = torch.chunk(dummy_rs_input, self.world_size)[
                        self.rank
                    ]
                    dist.reduce_scatter_tensor(dummy_rs_output, dummy_rs_input)

        def fwd_bwd():
            with (
                patch_all_gather(delayed_all_gather),
                patch_reduce_scatter(delayed_reduce_scatter),
            ):
                loss = model(inp).sum()
                loss.backward()

        ref_fwd_bwd_time = self._time_fn(ref_fwd_bwd)
        fwd_bwd_time = self._time_fn(fwd_bwd)
        # Backward: only 1st all-gather and last reduce-scatter are exposed;
        # double the backward compute since computing two gradients per layer
        # NOTE: Do not enforce the expected forward-backward time due to
        # flakiness in CI
        # expected_bwd_time = (
        #     comm_sleep_ms * 2 + num_linears * 2 * compute_sleep_ms + buffer_ms * 2
        # )
        self.assertLessEqual(fwd_bwd_time, ref_fwd_bwd_time)

    @skip_if_lt_x_gpu(2)
    @unittest.skipIf(TEST_HPU, "Sleep is not supported on HPU")
    def test_fully_shard_post_optim_event_overlap(self):
        torch.manual_seed(42)

        # Use non-trivial comm. time but still shorter than compute time
        dim, compute_sleep_ms, comm_sleep_ms = (4, 25, 10)
        # Define the model to have a high-compute linear followed by a
        # low-compute linear, where only the low-compute linear uses FSDP
        model = nn.Sequential(
            LinearWithSleep(dim, compute_sleep_ms), nn.Linear(dim, dim)
        ).to(device_type)
        fully_shard(model[1], reshard_after_forward=False)
        optim = torch.optim.AdamW(model.parameters(), lr=1e-2)

        orig_all_gather_into_tensor = dist.all_gather_into_tensor

        def delayed_all_gather(*args, **kwargs):
            torch.get_device_module(device_type)._sleep(
                int(comm_sleep_ms * get_cycles_per_ms())
            )
            return orig_all_gather_into_tensor(*args, **kwargs)

        inp = torch.randn((2, dim), device=device_type)

        def run_train_steps(num_iters: int, use_post_optim_event: bool):
            for _ in range(num_iters):
                optim.zero_grad()
                with patch_all_gather(delayed_all_gather):
                    loss = model(inp).sum()
                loss.backward()
                with implicit_replication():
                    optim.step()
                if use_post_optim_event:
                    post_optim_event = (
                        torch.get_device_module(device_type)
                        .current_stream()
                        .record_event()
                    )
                    model[1].set_post_optim_event(post_optim_event)

        run_train_steps(1, False)  # warmup CUDA and allocator
        num_iters = 5
        baseline_time = self._time_fn(
            functools.partial(run_train_steps, num_iters, False)
        )
        test_time = self._time_fn(functools.partial(run_train_steps, num_iters, True))

        buffer_ms = 4  # CPU delays and copies
        # Baseline: FSDP all-gather is exposed since the FSDP module waits for
        # the current stream and hence the high-compute linear
        self.assertLessEqual(
            baseline_time,
            num_iters * (3 * compute_sleep_ms + comm_sleep_ms + buffer_ms),
        )
        # Test: FSDP all-gather is overlapped with the high-compute linear
        # since the FSDP module only waits for the post-optim event (except on
        # the 1st iteration when no event has been recorded)
        expected_test_time = (
            num_iters * (3 * compute_sleep_ms + buffer_ms) + comm_sleep_ms
        )
        self.assertLessEqual(test_time, expected_test_time)
        # Since `get_cycles_per_ms` uses lru cache, there may be some variance
        # between the initially determined cycles vs. the current cycles per
        # ms, so we relax the baseline check to just that it is greater than
        # the test time rather than the expected test time
        self.assertGreater(baseline_time, test_time)

    def _time_fn(self, fn: Callable):
        start_event = device_module.Event(enable_timing=True)
        end_event = device_module.Event(enable_timing=True)
        dist.barrier()
        device_module.synchronize()
        start_event.record()
        fn()
        end_event.record()
        device_module.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        return elapsed_time


class Matmul(torch.autograd.Function):
    # Use CUDA sleeps to emulate compute time
    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor, sleep_ms: int):
        ctx.save_for_backward(input, weight)
        ctx.sleep_ms = sleep_ms
        torch.get_device_module(device_type)._sleep(int(sleep_ms * get_cycles_per_ms()))
        return input @ weight

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (input, weight) = ctx.saved_tensors
        torch.get_device_module(device_type)._sleep(
            int(2 * ctx.sleep_ms * get_cycles_per_ms())
        )
        grad_input = grad_output @ weight.T
        grad_weight = input.T @ grad_output
        return grad_input, grad_weight, None


class LinearWithSleep(nn.Module):
    def __init__(self, dim: int, sleep_ms: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn((dim, dim)))
        self.sleep_ms = sleep_ms

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.relu(Matmul.apply(x, self.weight, self.sleep_ms))


if __name__ == "__main__":
    run_tests()
