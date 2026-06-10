#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Flight Recorder Verification Script with Debug Server Integration

This script demonstrates how to use the FlightRecorderHook with torchcomms
to record collective operations and dump traces for debugging, along with
the PyTorch debug server for real-time debugging capabilities.

The traces can be analyzed using the PyTorch flight_recorder trace analyzer:
    python -m torch.distributed.flight_recorder.fr_trace <trace_dir>

Debug server endpoints available at http://localhost:<DEBUG_SERVER_PORT>:
    /torchcomms_fr_trace   - Flight recorder trace viewer
    /torchcomms_fr_trace_json - Flight recorder trace as JSON

Usage:
    # Set the dump directory (traces will be written as <prefix><rank>)
    export TORCHCOMM_FR_DUMP_TEMP_FILE=/tmp/flight_recorder_traces/rank_

    torchrun --nproc_per_node=2 verify_flight_recorder.py

    Or with a specific backend:
    TEST_BACKEND=nccl torchrun --nproc_per_node=2 verify_flight_recorder.py

    Enable debug server (rank 0 only by default):
    TEST_DEBUG_SERVER=1 torchrun --nproc_per_node=2 verify_flight_recorder.py

    Specify debug server port:
    TEST_DEBUG_SERVER=1 DEBUG_SERVER_PORT=25999 torchrun --nproc_per_node=2 verify_flight_recorder.py
"""

import os
import time
from datetime import timedelta

import torch
from torch.comms import new_comm, ReduceOp
from torch.comms.hooks import FlightRecorderHook
from torch.distributed.debug import start_debug_server


def main() -> None:
    # Get backend from environment or default to gloo
    backend = os.environ.get("TEST_BACKEND", "gloo")
    device = torch.device(os.environ.get("TEST_DEVICE", "cuda"))

    # Debug server configuration
    enable_debug_server = os.environ.get("TEST_DEBUG_SERVER", "0") == "1"
    debug_server_port = int(os.environ.get("DEBUG_SERVER_PORT", "25999"))

    # Initialize TorchComm (this starts the distributed backend)
    comm = new_comm(
        backend=backend,
        device=device,
        name="main_comm",
        timeout=timedelta(seconds=300),
    )

    rank = comm.get_rank()
    world_size = comm.get_size()

    # Calculate device ID
    num_devices = torch.accelerator.device_count()
    device_id = rank % num_devices
    target_device = torch.device(f"{device.type}:{device_id}")

    print(f"Rank {rank}/{world_size}: Running on device {device_id}")

    # Start debug server on ALL ranks
    # The frontend HTTP server runs on rank 0, but the backend worker
    # must run on every rank to register with the store
    if enable_debug_server:
        print(f"Rank {rank}: Starting debug server (port {debug_server_port})")
        start_debug_server(port=debug_server_port)
        if rank == 0:
            print(
                f"Rank {rank}: Debug server frontend at http://localhost:{debug_server_port}"
            )

    # Create FlightRecorderHook
    recorder = FlightRecorderHook(max_entries=100)
    recorder.register_with_comm(comm)

    # Create a tensor with rank-specific data
    tensor = torch.full(
        (1024,),
        float(rank + 1),
        dtype=torch.float32,
        device=target_device,
    )

    print(f"Rank {rank}: Before AllReduce: {tensor[0].item()}")

    # Perform multiple collective operations
    for _ in range(5):
        comm.all_reduce(tensor, ReduceOp.SUM, async_op=False)

    # Broadcast from rank 0
    comm.broadcast(tensor, root=0, async_op=False)

    # Synchronize CUDA stream
    torch.accelerator.current_stream().synchronize()

    print(f"Rank {rank}: After AllReduce: {tensor[0].item()}")

    # Dump traces using dump_file API
    # The output location is controlled by TORCHCOMM_FR_DUMP_TEMP_FILE env var
    # Files are written as <TORCHCOMM_FR_DUMP_TEMP_FILE><rank>
    recorder.dump_file(rank)

    dump_prefix = os.environ.get(
        "TORCHCOMM_FR_DUMP_TEMP_FILE", "~/.cache/torchcomm_fr_trace_"
    )
    trace_file = f"{dump_prefix}{rank}"
    print(f"Rank {rank}: Flight recorder trace dumped to {trace_file}")

    if rank == 0:
        dump_dir = os.path.dirname(dump_prefix) or "."
        print(f"\n=== Flight Recorder traces saved to: {dump_dir} ===")
        print("To analyze the traces, run:")
        print(f"  python -m torch.distributed.flight_recorder.fr_trace -j {dump_dir}")

    # Keep debug server running for manual verification
    if enable_debug_server:
        print(
            f"\nRank {rank}: Debug server running at http://localhost:{debug_server_port}"
        )
        print("Press Ctrl+C to stop the server and exit...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print(f"\nRank {rank}: Stopping debug server...")

    # Cleanup
    comm.finalize()


if __name__ == "__main__":
    main()
