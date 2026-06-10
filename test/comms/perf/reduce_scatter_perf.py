# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-unsafe

from perf.perf_test_helpers import (
    create_tensor,
    PerfParams,
    PerfResult,
    PerfTimer,
    print_perf_header,
    print_perf_result,
    sync_device,
)

import torch
import torch.comms


def run_reduce_scatter_perf(
    comm: torch.comms.TorchComm,
    params: PerfParams,
    device: torch.device,
) -> None:
    rank = comm.get_rank()
    num_ranks = comm.get_size()

    if rank == 0:
        mode = "Asynchronous" if params.async_op else "Synchronous"
        print(f"\n=== {mode} ReduceScatter Performance ===")
    print_perf_header(rank)

    element_size = torch.tensor([], dtype=params.dtype).element_size()

    msg_size = params.min_size
    while msg_size <= params.max_size:
        num_elements = msg_size // element_size
        if num_elements == 0:
            num_elements = 1
        output_tensor = create_tensor(num_elements, rank, device, params.dtype)

        # Create input tensor list
        input_list = [
            create_tensor(num_elements, rank, device, params.dtype)
            for _ in range(num_ranks)
        ]

        # Warmup
        for _ in range(params.warmup_iterations):
            work = comm.reduce_scatter(
                output_tensor, input_list, torch.comms.ReduceOp.SUM, params.async_op
            )
            if params.async_op:
                work.wait()

        # Synchronize all ranks before measurement
        comm.barrier(False)

        # Measure
        timer = PerfTimer()
        sync_device(device)
        timer.start()

        for i in range(params.measure_iterations):
            work = comm.reduce_scatter(
                output_tensor, input_list, torch.comms.ReduceOp.SUM, params.async_op
            )
            if params.async_op:
                work.wait()

            if params.iteration_window > 0 and (i + 1) % params.iteration_window == 0:
                sync_device(device)

        sync_device(device)
        timer.stop()

        # Calculate statistics
        total = timer.elapsed_us()
        avg_time = total / params.measure_iterations

        # ReduceScatter bus bandwidth: (n-1) / n * total_size / time
        total_size = num_elements * num_ranks * element_size
        algo_bw = total_size / avg_time  # bytes/us = MB/s
        bus_bw_factor = (num_ranks - 1) / num_ranks
        bus_bw_gbps = algo_bw * bus_bw_factor / 1000.0  # Convert to GB/s

        result = PerfResult(
            message_size_bytes=num_elements * num_ranks * element_size,
            num_ranks=num_ranks,
            iterations=params.measure_iterations,
            total_time_us=total,
            avg_time_us=avg_time,
            min_time_us=avg_time,
            max_time_us=avg_time,
            bus_bw_gbps=bus_bw_gbps,
        )

        print_perf_result(result, rank)

        msg_size *= params.size_scaling_factor
