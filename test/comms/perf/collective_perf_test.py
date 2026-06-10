# Copyright (c) Meta Platforms, Inc. and affiliates.
# Owner(s): ["oncall: distributed"]
# pyre-unsafe

import os
import sys

from perf.all_gather_perf import run_all_gather_perf
from perf.all_gather_single_perf import run_all_gather_single_perf
from perf.all_reduce_perf import run_all_reduce_perf
from perf.all_to_all_perf import run_all_to_all_perf
from perf.all_to_all_single_perf import run_all_to_all_single_perf
from perf.barrier_perf import run_barrier_perf
from perf.broadcast_perf import run_broadcast_perf
from perf.gather_perf import run_gather_perf
from perf.perf_test_helpers import (
    dtype_to_string,
    parse_dtype,
    PerfParams,
    print_usage,
    validate_params,
)
from perf.reduce_perf import run_reduce_perf
from perf.reduce_scatter_perf import run_reduce_scatter_perf
from perf.reduce_scatter_single_perf import run_reduce_scatter_single_perf
from perf.scatter_perf import run_scatter_perf
from perf.send_recv_perf import run_send_recv_perf

import torch
import torch.comms


# Map collective names to their perf functions
COLLECTIVE_RUNNERS = {
    "all_reduce": run_all_reduce_perf,
    "all_gather": run_all_gather_perf,
    "all_gather_single": run_all_gather_single_perf,
    "reduce_scatter": run_reduce_scatter_perf,
    "reduce_scatter_single": run_reduce_scatter_single_perf,
    "all_to_all": run_all_to_all_perf,
    "all_to_all_single": run_all_to_all_single_perf,
    "broadcast": run_broadcast_perf,
    "reduce": run_reduce_perf,
    "scatter": run_scatter_perf,
    "gather": run_gather_perf,
    "send_recv": run_send_recv_perf,
    "barrier": run_barrier_perf,
}


def parse_args(args: list) -> tuple[str, PerfParams, str | None]:
    """Parse command-line arguments and return (collective, params, error)."""
    collective = "all"
    params = PerfParams()

    i = 0
    while i < len(args):
        arg = args[i]

        if arg in ("--help", "-h"):
            return collective, params, "help"
        elif arg == "--async":
            params.async_op = True
        elif arg == "--warmup" and i + 1 < len(args):
            i += 1
            params.warmup_iterations = int(args[i])
        elif arg == "--iters" and i + 1 < len(args):
            i += 1
            params.measure_iterations = int(args[i])
        elif arg == "--window" and i + 1 < len(args):
            i += 1
            params.iteration_window = int(args[i])
        elif arg == "--min-size" and i + 1 < len(args):
            i += 1
            params.min_size = int(args[i])
        elif arg == "--max-size" and i + 1 < len(args):
            i += 1
            params.max_size = int(args[i])
        elif arg == "--size-scaling-factor" and i + 1 < len(args):
            i += 1
            params.size_scaling_factor = int(args[i])
        elif arg == "--dtype" and i + 1 < len(args):
            i += 1
            params.dtype = parse_dtype(args[i])
        elif not arg.startswith("-"):
            collective = arg

        i += 1

    return collective, params, None


def run_collectives(
    collective: str,
    comm: torch.comms.TorchComm,
    params: PerfParams,
    device: torch.device,
) -> None:
    """Run the specified collective performance test(s)."""
    if collective == "all":
        for runner in COLLECTIVE_RUNNERS.values():
            runner(comm, params, device)
    elif collective in COLLECTIVE_RUNNERS:
        COLLECTIVE_RUNNERS[collective](comm, params, device)


def main() -> int:
    collective, params, parse_error = parse_args(sys.argv[1:])

    if parse_error == "help":
        print_usage(sys.argv[0])
        return 0

    error = validate_params(collective, params)
    if error:
        print(f"Error: {error}\n", file=sys.stderr)
        print_usage(sys.argv[0])
        return 1

    device = torch.device(os.environ.get("TEST_DEVICE", "cuda"))

    test_backend = os.environ.get("TEST_BACKEND")
    if not test_backend:
        print("Error: TEST_BACKEND environment variable is not set", file=sys.stderr)
        return 1

    hints = {}
    fast_init_mode = os.environ.get("TEST_FAST_INIT_MODE")
    if fast_init_mode:
        hints["fastInitMode"] = fast_init_mode

    comm = torch.comms.new_comm(
        test_backend,
        device,
        name="collective_perf_test",
        hints=hints if hints else None,
    )
    rank = comm.get_rank()
    num_ranks = comm.get_size()

    if rank == 0:
        print("TorchComms Collective Performance Test")
        print("======================================")
        print(f"Backend: {comm.get_backend()}")
        print(f"Ranks: {num_ranks}")
        print(f"Collective: {collective}")
        print(f"Mode: {'async' if params.async_op else 'sync'}")
        print(f"Dtype: {dtype_to_string(params.dtype)}")
        print(f"Warmup: {params.warmup_iterations}")
        print(f"Iterations: {params.measure_iterations}")
        print(f"Window: {params.iteration_window}")
        print(
            f"Size range: {params.min_size} - {params.max_size} "
            f"bytes (x{params.size_scaling_factor})"
        )
        print()

    run_collectives(collective, comm, params, device)

    if rank == 0:
        print("\nPerformance test completed.")

    comm.finalize()

    return 0


if __name__ == "__main__":
    sys.exit(main())
