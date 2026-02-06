# Owner(s): ["oncall: distributed"]

import sys
import time
import unittest
from statistics import mean
from unittest.mock import patch

import torch
import torch.nn as nn
from torch import distributed as dist, Event
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTestContinuous
from torch.testing._internal.common_utils import (
    get_cycles_per_ms,
    run_tests,
    TEST_HPU,
    TEST_WITH_DEV_DBG_ASAN,
    TEST_XPU,
    xfailIf,
)


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class Layer(nn.Module):
    def __init__(self, compute_cycles, has_params: bool):
        super().__init__()
        self.sleep_cycles = compute_cycles
        self.optional_param = None
        if has_params:
            self.optional_param = nn.Parameter(torch.rand(1))

    def forward(self, x):
        # Get 2 events.
        self.e1 = Event(enable_timing=True)
        self.e2 = Event(enable_timing=True)

        # Record the fake forward compute time.
        self.e1.record()
        if self.sleep_cycles > 0:
            if torch.cuda.is_available():
                torch.cuda._sleep(self.sleep_cycles)
        if self.optional_param is not None:
            x = x + self.optional_param  # force the param to be part of the graph
        self.e2.record()
        return x

    def get_time(self):
        # return the recorded duration.
        return self.e1.elapsed_time(self.e2)


def _create_model(compute_cycles, has_params: bool):
    # Use `limit_all_gathers=False` since the timing being tested relies on the
    # CPU running ahead of the GPU
    model = FSDP(
        nn.Sequential(
            FSDP(Layer(compute_cycles, has_params), limit_all_gathers=False),
            FSDP(Layer(compute_cycles, has_params), limit_all_gathers=False),
            FSDP(Layer(compute_cycles, has_params), limit_all_gathers=False),
            FSDP(Layer(compute_cycles, has_params), limit_all_gathers=False),
        ),
        limit_all_gathers=False,
    ).to(device_type)
    return model


class Min10:
    def __init__(self) -> None:
        self.data = []

    def add(self, new_data):
        if len(self.data) < 10:
            self.data.append(new_data)
        else:
            self.data = sorted(self.data)
            if new_data < self.data[-1]:
                self.data[-1] = new_data

    def avg(self):
        return mean(self.data)


class TestForwardOverlapWorldSizeOne(FSDPTestContinuous):
    @property
    def world_size(self):
        return 1

    def _dist_train(self):
        rank = self.rank
        world_size = self.world_size
        # Save the original torch.distributed.all_gather_into_tensor function since we will
        # patch it to include an artificial delay.
        orig_all_gather = torch.distributed.all_gather_into_tensor

        def run(compute_cycles, all_gather_cycles):
            has_params = all_gather_cycles > 0
            model = _create_model(compute_cycles, has_params)

            # Get the input and sets the input's requires_grad to True because
            # we have a fake compute in the forward pass.
            batch = torch.rand(1).to(device_type)
            batch.requires_grad = True

            # Run one dummy iteration to trigger the execution order validation
            # all-gathers
            out = model(batch)
            out.backward()
            model.zero_grad(set_to_none=True)

            # We run 20 iterations but only collect timing data from the minimal 10
            # data points because nondeterministic system events can disturb the timing.
            cpu_iter = Min10()
            cpu_wait = Min10()
            gpu_compute = Min10()
            gpu_total = Min10()
            for _ in range(20):
                # Get two events for measuring the overall time.
                e1 = Event(enable_timing=True)
                e2 = Event(enable_timing=True)

                cpu_start = time.process_time()

                all_gather_called = False

                def _delayed_all_gather(*args, **kwargs):
                    nonlocal all_gather_called
                    all_gather_called = True
                    if torch.cuda.is_available():
                        torch.cuda._sleep(all_gather_cycles)
                    if not orig_all_gather:
                        raise AssertionError("Expected orig_all_gather to be truthy")
                    return orig_all_gather(*args, **kwargs)

                # forward pass
                #
                # Even though both e1 & e2 are on the compute stream, since
                # compute depends on all_gather, e2-e1 includes all_gather time.
                e1.record()
                with patch(
                    "torch.distributed.all_gather_into_tensor", _delayed_all_gather
                ):
                    out = model(batch)
                    if has_params and world_size > 1:
                        self.assertTrue(all_gather_called)
                    else:
                        self.assertFalse(all_gather_called)
                e2.record()

                # backward pass
                out.backward()
                model.zero_grad(set_to_none=True)

                cpu_iter_time = time.process_time() - cpu_start

                # wait for gpu
                out.item()
                cpu_wait_for_gpu_time = time.process_time() - cpu_start - cpu_iter_time

                # get sum of the compute time
                times = []
                for mod in model.modules():
                    if not isinstance(mod, Layer):
                        continue
                    times.append(mod.get_time())

                # get gpu compute + all_gather time
                overall_gpu_time = e1.elapsed_time(e2)

                cpu_iter.add(cpu_iter_time)
                cpu_wait.add(cpu_wait_for_gpu_time)
                gpu_compute.add(sum(times))
                gpu_total.add(overall_gpu_time)

            del model
            return {
                "cpu_iter": cpu_iter.avg(),
                "cpu_wait": cpu_wait.avg(),
                "gpu_compute": gpu_compute.avg(),
                "gpu_total": gpu_total.avg(),
            }

        sleep_cycles = int(100 * get_cycles_per_ms())

        e1 = run(0, 0)  # no compute, no all-gather
        e2 = run(0, sleep_cycles)  # no compute, only all-gather
        e3 = run(sleep_cycles, 0)  # only compute, no all-gather
        e4 = run(sleep_cycles, sleep_cycles)  # both compute and all-gather
        debug_string = f"\nrank{rank}:\n  e1: {e1}\n  e2: {e2}\n  e3: {e3}\n  e4: {e4}"
        print(debug_string)

        # Check the cpu/gpu timing. CPU should run ahead of GPU. Therefore, cpu-gpu
        # wait should be long, except when there is no real work on GPU.
        #
        # If the assertions fail below, we likely have a cpu-gpu wait in the forward/backward pass.
        # e4["cpu_iter"] may not be short as cpu may take some time to queue both compute and all-gather.
        short = [
            e1["cpu_iter"],
            e2["cpu_iter"],
            e3["cpu_iter"],
            e1["cpu_wait"],
        ]
        long = [e3["cpu_wait"], e4["cpu_wait"]]
        if world_size == 1:
            short.append(e2["cpu_wait"])  # all gather should not be happening.
        else:
            long.append(
                e2["cpu_wait"]
            )  # all gather should happen and prolong the cpu-gpu wait.
        for s in short:
            for l in long:
                # 10X longer is a safe margin, since the GPU work timing is around 100X more
                # of that of the CPU.
                self.assertTrue(s * 10 < l)

        # Check the GPU timing.
        short = [e1["gpu_compute"], e1["gpu_total"], e2["gpu_compute"]]
        long = [e3["gpu_compute"], e3["gpu_total"], e4["gpu_compute"], e4["gpu_total"]]
        if world_size == 1:
            short.append(e2["gpu_total"])  # all gather should not be happening.
        else:
            long.append(
                e2["gpu_total"]
            )  # all gather should happen and prolong the cpu-gpu wait.
        for s in short:
            for l in long:
                # 10X longer is a safe margin, since the time is around 100X longer
                # when there is work on GPU vs. no work.
                self.assertTrue(s * 10 < l)

        # Check the GPU overlapping when there is all-gather.
        if world_size > 1:
            compute_only = e3["gpu_compute"]
            all_gather_only = e2["gpu_total"]
            both = e4["gpu_total"]
            self.assertTrue(compute_only + all_gather_only > 1.1 * both)

    @unittest.skipIf(TEST_HPU, "HPU doesn't has HW sleep API support, skipping")
    @xfailIf(TEST_XPU)  # https://github.com/intel/torch-xpu-ops/issues/1504
    @skip_if_lt_x_gpu(2)
    def test_forward_overlap(self):
        self._dist_train()


class TestForwardOverlapWorldSizeTwo(TestForwardOverlapWorldSizeOne):
    @property
    def world_size(self):
        return 2


devices = ("cuda", "hpu", "xpu")
instantiate_device_type_tests(
    TestForwardOverlapWorldSizeOne, globals(), only_for=devices, allow_xpu=True
)
if __name__ == "__main__":
    run_tests()
