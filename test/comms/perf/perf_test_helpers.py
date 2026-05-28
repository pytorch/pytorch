# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-unsafe

import time
from dataclasses import dataclass, field

import torch


@dataclass
class PerfResult:
    message_size_bytes: int = 0
    num_ranks: int = 0
    iterations: int = 0
    total_time_us: float = 0.0
    avg_time_us: float = 0.0
    min_time_us: float = 0.0
    max_time_us: float = 0.0
    bus_bw_gbps: float = 0.0


@dataclass
class PerfParams:
    async_op: bool = False
    warmup_iterations: int = 5
    measure_iterations: int = 1000
    # Number of iterations between stream synchronizations during measurement.
    # If 0, only synchronize after all iterations complete.
    iteration_window: int = 0
    # Message size range in bytes (powers of 2)
    min_size: int = 4  # 4 bytes
    max_size: int = 67108864  # 64 MB
    # Scaling factor for message sizes (default 2 = powers of 2)
    size_scaling_factor: int = 2
    # Data type
    dtype: torch.dtype = field(default_factory=lambda: torch.float32)


class PerfTimer:
    def __init__(self):
        self._start_time = 0.0
        self._end_time = 0.0
        self._running = False

    def start(self):
        self._start_time = time.perf_counter()
        self._running = True

    def stop(self):
        self._end_time = time.perf_counter()
        self._running = False

    def reset(self):
        self._running = False

    def elapsed_us(self) -> float:
        return (self._end_time - self._start_time) * 1e6

    def elapsed_ms(self) -> float:
        return self.elapsed_us() / 1000.0


def print_perf_header(rank: int) -> None:
    if rank != 0:
        return
    print(
        f"{'SendMsgSize(B)':<15}{'Ranks':<10}{'Iters':<10}"
        f"{'Avg(us)':<15}{'Min(us)':<15}{'Max(us)':<15}{'BusBw(GB/s)':<15}"
    )
    print("-" * 95)


def print_perf_result(result: PerfResult, rank: int) -> None:
    if rank != 0:
        return
    print(
        f"{result.message_size_bytes:<15}{result.num_ranks:<10}{result.iterations:<10}"
        f"{result.avg_time_us:<15.2f}{result.min_time_us:<15.2f}{result.max_time_us:<15.2f}"
        f"{result.bus_bw_gbps:<15.2f}"
    )


def create_tensor(
    num_elements: int,
    rank: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return torch.ones(num_elements, dtype=dtype, device=device) * float(rank + 1)


def sync_device(device: torch.device) -> None:
    """Synchronize the device stream if it's a CUDA device."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def print_usage(program_name: str) -> None:
    print(f"""Usage: {program_name} <collective> [options]

Collectives:
  all_reduce             - AllReduce collective
  all_gather             - AllGather collective (tensor list output)
  all_gather_single      - AllGather collective (single tensor output)
  reduce_scatter         - ReduceScatter collective (tensor list input)
  reduce_scatter_single  - ReduceScatter (single tensor input)
  all_to_all             - AllToAll collective (tensor list)
  all_to_all_single      - AllToAll collective (single tensor)
  broadcast              - Broadcast collective
  reduce                 - Reduce collective
  scatter                - Scatter collective
  gather                 - Gather collective
  send_recv              - Send/Recv point-to-point (ping-pong)
  barrier                - Barrier collective
  all                    - Run all collectives

Options:
  --async                - Run async mode (default: sync)
  --warmup <n>           - Number of warmup iterations (default: 5)
  --iters <n>            - Number of measurement iterations (default: 1000)
  --window <n>           - Iterations between stream syncs (default: 0)
  --min-size <n>         - Min message size in bytes (default: 4)
  --max-size <n>         - Max message size in bytes (default: 67108864)
  --size-scaling-factor <n> - Size multiplier between tests (default: 2)
  --dtype <type>         - Data type: float32, float16, bfloat16, float64,
                           int32, int64 (default: float32)
  --help, -h             - Show this help message
""")


def parse_dtype(dtype_str: str) -> torch.dtype:
    dtype_map = {
        "float": torch.float32,
        "float32": torch.float32,
        "fp32": torch.float32,
        "half": torch.float16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "double": torch.float64,
        "float64": torch.float64,
        "fp64": torch.float64,
        "int": torch.int32,
        "int32": torch.int32,
        "long": torch.int64,
        "int64": torch.int64,
    }
    if dtype_str not in dtype_map:
        raise ValueError(f"Unknown dtype: {dtype_str}")
    return dtype_map[dtype_str]


def dtype_to_string(dtype: torch.dtype) -> str:
    dtype_map = {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float64: "float64",
        torch.int32: "int32",
        torch.int64: "int64",
    }
    if dtype not in dtype_map:
        raise ValueError(f"Unknown dtype: {dtype}")
    return dtype_map[dtype]


def validate_params(collective: str, params: PerfParams) -> str:
    valid_collectives = [
        "all_reduce",
        "allreduce",
        "all_gather",
        "allgather",
        "all_gather_single",
        "allgathersingle",
        "reduce_scatter",
        "reducescatter",
        "reduce_scatter_single",
        "reducescattersingle",
        "all_to_all",
        "alltoall",
        "all_to_all_single",
        "alltoallsingle",
        "broadcast",
        "reduce",
        "scatter",
        "gather",
        "send_recv",
        "sendrecv",
        "barrier",
        "all",
    ]

    if collective not in valid_collectives:
        return f"Unknown collective '{collective}'"

    if params.size_scaling_factor < 2:
        return "size_scaling_factor must be at least 2"

    if params.min_size <= 0:
        return "min_size must be positive"

    if params.max_size < params.min_size:
        return "max_size must be >= min_size"

    # Validate that max_size is reachable from min_size via scaling factor
    size = params.min_size
    while size < params.max_size:
        size *= params.size_scaling_factor
    if size != params.max_size and params.min_size != params.max_size:
        return "max_size must be min_size * size_scaling_factor^n for some integer n"

    # Validate dtype divides sizes evenly
    element_size = torch.tensor([], dtype=params.dtype).element_size()
    if params.min_size % element_size != 0:
        return (
            f"min_size must be divisible by dtype element size ({element_size} bytes)"
        )
    if params.max_size % element_size != 0:
        return (
            f"max_size must be divisible by dtype element size ({element_size} bytes)"
        )

    if params.warmup_iterations < 0:
        return "warmup_iterations must be non-negative"
    if params.measure_iterations <= 0:
        return "measure_iterations must be positive"
    if params.iteration_window < 0:
        return "iteration_window must be non-negative"

    return ""  # Valid
