# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-unsafe

from perf.perf_test_helpers import (
    PerfParams,
    PerfResult,
    PerfTimer,
    print_perf_header,
    print_perf_result,
    sync_device,
)

import torch
import torch.comms


def run_barrier_perf(
    comm: torch.comms.TorchComm,
    params: PerfParams,
    device: torch.device,
) -> None:
    rank = comm.get_rank()
    num_ranks = comm.get_size()

    if rank == 0:
        mode = "Asynchronous" if params.async_op else "Synchronous"
        print(f"\n=== {mode} Barrier Performance ===")
    print_perf_header(rank)

    # Barrier has no message size, run once
    # Warmup
    for _ in range(params.warmup_iterations):
        work = comm.barrier(params.async_op)
        if params.async_op:
            work.wait()

    # Synchronize all ranks before measurement
    comm.barrier(False)

    # Measure
    timer = PerfTimer()
    sync_device(device)
    timer.start()

    for i in range(params.measure_iterations):
        work = comm.barrier(params.async_op)
        if params.async_op:
            work.wait()

        if params.iteration_window > 0 and (i + 1) % params.iteration_window == 0:
            sync_device(device)

    sync_device(device)
    timer.stop()

    # Calculate statistics
    total = timer.elapsed_us()
    avg_time = total / params.measure_iterations

    # Barrier has no data transfer, so bus bandwidth is 0
    result = PerfResult(
        message_size_bytes=0,
        num_ranks=num_ranks,
        iterations=params.measure_iterations,
        total_time_us=total,
        avg_time_us=avg_time,
        min_time_us=avg_time,
        max_time_us=avg_time,
        bus_bw_gbps=0.0,
    )

    print_perf_result(result, rank)
