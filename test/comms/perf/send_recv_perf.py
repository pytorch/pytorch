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


def _do_ping_pong(
    comm: torch.comms.TorchComm,
    tensor: torch.Tensor,
    rank: int,
    peer: int,
    async_op: bool,
) -> None:
    """Execute one ping-pong iteration between rank 0 and rank 1."""
    if rank == 0:
        work = comm.send(tensor, peer, async_op)
        if async_op:
            work.wait()
        work2 = comm.recv(tensor, peer, async_op)
        if async_op:
            work2.wait()
    elif rank == 1:
        work = comm.recv(tensor, peer, async_op)
        if async_op:
            work.wait()
        work2 = comm.send(tensor, peer, async_op)
        if async_op:
            work2.wait()


def run_send_recv_perf(
    comm: torch.comms.TorchComm,
    params: PerfParams,
    device: torch.device,
) -> None:
    rank = comm.get_rank()
    num_ranks = comm.get_size()

    if num_ranks < 2:
        if rank == 0:
            print("SendRecv test requires at least 2 ranks, skipping")
        return

    if rank == 0:
        mode = "Asynchronous" if params.async_op else "Synchronous"
        print(f"\n=== {mode} SendRecv Performance ===")
    print_perf_header(rank)

    peer = 1 if rank == 0 else 0
    element_size = torch.tensor([], dtype=params.dtype).element_size()

    msg_size = params.min_size
    while msg_size <= params.max_size:
        num_elements = msg_size // element_size
        if num_elements == 0:
            num_elements = 1
        tensor = create_tensor(num_elements, rank, device, params.dtype)

        # Warmup
        for _ in range(params.warmup_iterations):
            _do_ping_pong(comm, tensor, rank, peer, params.async_op)

        comm.barrier(False)

        # Measure ping-pong latency
        timer = PerfTimer()
        sync_device(device)
        timer.start()

        for i in range(params.measure_iterations):
            _do_ping_pong(comm, tensor, rank, peer, params.async_op)

            if params.iteration_window > 0 and (i + 1) % params.iteration_window == 0:
                sync_device(device)

        sync_device(device)
        timer.stop()

        total = timer.elapsed_us()
        avg_time = total / params.measure_iterations

        # SendRecv bus bandwidth: size / one-way time
        one_way_time = avg_time / 2
        algo_bw = (num_elements * element_size) / one_way_time  # bytes/us = MB/s
        bus_bw_gbps = algo_bw / 1000.0  # Convert to GB/s

        result = PerfResult(
            message_size_bytes=num_elements * element_size,
            num_ranks=2,
            iterations=params.measure_iterations,
            total_time_us=total,
            avg_time_us=avg_time / 2,
            min_time_us=avg_time / 2,
            max_time_us=avg_time / 2,
            bus_bw_gbps=bus_bw_gbps,
        )

        print_perf_result(result, rank)

        msg_size *= params.size_scaling_factor
